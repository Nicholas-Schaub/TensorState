"""GROUP BY benchmark: current Rust ``_lex_sort`` vs DuckDB out-of-core.

Context (Plane AIQ-33): the core analytical operation on captured states is
"count unique microstates -> frequency distribution -> entropy", i.e. a
``GROUP BY microstate COUNT(*)``. Today tensorstate computes the frequency
distribution by lex-sorting bit-packed uint8 rows in a Rust extension
(``_TensorState_rs._lex_sort``) and taking ``np.diff(bin_edges)``. This script
asks whether DuckDB (out-of-core hash aggregation, automatic spill-to-disk)
would be a better backend for the state-space sizes typical of early CNN
layers.

What it measures, for synthetic bit-packed uint8 microstate matrices of shape
``(n_rows, bytes_per_row)`` with a controllable number of distinct microstates:

  * lex_sort   -- ``_lex_sort(states, n)`` + ``np.diff(edges)`` (current path).
  * duckdb-blob -- one BLOB column (the packed row), ``GROUP BY blob COUNT(*)``.
  * duckdb-cols -- N uint8 columns, ``GROUP BY c0..cN COUNT(*)``.

For each it records mean wall-clock (ms) and peak RSS delta (MiB, measured in a
child process so allocator/arena effects do not bleed across methods). It also
verifies all three produce the same multiset of unique-state counts.

Usage::

    cd /polus1/schaubnj/ngrf/tensorstate
    unset CONDA_PREFIX
    uv pip install duckdb pyarrow   # benchmark-only deps
    .venv/bin/python benches/duckdb_vs_lexsort.py

Flags:
    --max-rows N   cap the largest n_rows (default: run all incl. 1e7).
    --reps N       timed reps per method (default 3).
"""

import argparse
import functools
import multiprocessing as mp
import resource
import sys
import time

import numpy as np
import TensorState._TensorState_rs as ts

try:
    import duckdb
    import pyarrow as pa
except ImportError:
    print(
        "Benchmark deps missing. Run `uv pip install duckdb pyarrow`.",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def make_states(n_rows: int, bytes_per_row: int, n_distinct: int, seed: int = 0):
    """Build a bit-packed uint8 microstate matrix with bounded cardinality.

    A palette of ``n_distinct`` random byte-rows is drawn once, then rows are
    sampled from the palette with a Zipf-ish (1/rank) weighting so a handful of
    microstates dominate -- which is what real early-CNN layers look like
    (a few common firing patterns, a long tail of rare ones).

    Args:
        n_rows: number of state observations (rows).
        bytes_per_row: ceil(neurons / 8).
        n_distinct: size of the microstate palette (true group cardinality is
            <= this; could be smaller if the palette has dup draws).
        seed: RNG seed.

    Returns:
        ``(n_rows, bytes_per_row)`` C-contiguous uint8 array.
    """
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(n_distinct, bytes_per_row), dtype=np.uint8)
    ranks = np.arange(1, n_distinct + 1)
    weights = 1.0 / ranks
    weights /= weights.sum()
    idx = rng.choice(n_distinct, size=n_rows, p=weights)
    return np.ascontiguousarray(palette[idx])


# ---------------------------------------------------------------------------
# The three implementations. Each returns a *sorted* counts array so the
# multiset of group sizes can be compared across methods.
# ---------------------------------------------------------------------------
def counts_lex_sort(states: np.ndarray) -> np.ndarray:
    """Current production path: Rust lex-sort + diff of bin edges."""
    edges, _index = ts._lex_sort(states, states.shape[0])
    counts = np.diff(edges)
    counts.sort()
    return counts


def _arrow_blob_table(states: np.ndarray) -> "pa.Table":
    """One BLOB column: each row packed as a contiguous byte string."""
    raw = states.tobytes()  # row-major; rows are contiguous
    w = states.shape[1]
    # pa.binary(w) fixed-size-binary view over the raw buffer (zero-ish copy).
    buf = pa.py_buffer(raw)
    arr = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(w), states.shape[0], [None, buf]
    )
    return pa.table({"s": arr})


def counts_duckdb_blob(
    states: np.ndarray, con: "duckdb.DuckDBPyConnection"
) -> np.ndarray:
    """DuckDB GROUP BY on a single fixed-size-binary (BLOB) column."""
    tbl = _arrow_blob_table(states)  # noqa: F841 -- referenced by SQL below
    res = con.execute("SELECT count(*) AS c FROM tbl GROUP BY s").fetch_arrow_table()
    counts = np.array(res.column("c").to_numpy(), copy=True)
    counts.sort()
    return counts


def _arrow_cols_table(states: np.ndarray) -> "pa.Table":
    """N uint8 columns, one per packed byte."""
    cols = {f"c{j}": pa.array(states[:, j]) for j in range(states.shape[1])}
    return pa.table(cols)


def counts_duckdb_cols(
    states: np.ndarray, con: "duckdb.DuckDBPyConnection"
) -> np.ndarray:
    """DuckDB GROUP BY across N uint8 columns."""
    tbl = _arrow_cols_table(states)  # noqa: F841 -- referenced by SQL below
    group_cols = ", ".join(f"c{j}" for j in range(states.shape[1]))
    res = con.execute(
        f"SELECT count(*) AS c FROM tbl GROUP BY {group_cols}"
    ).fetch_arrow_table()
    counts = np.array(res.column("c").to_numpy(), copy=True)
    counts.sort()
    return counts


# ---------------------------------------------------------------------------
# Timing + peak-memory in an isolated child process
# ---------------------------------------------------------------------------
def _peak_rss_mib() -> float:
    """Peak RSS of this process so far, in MiB (Linux: ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _child(method: str, n_rows: int, bpr: int, n_distinct: int, reps: int, q):
    """Run one method in a fresh process; report (mean_ms, peak_mem_delta_mib)."""
    states = make_states(n_rows, bpr, n_distinct)
    con = None
    if method.startswith("duckdb"):
        con = duckdb.connect(":memory:")
        # Bound RAM so spill-to-disk engages on the big cases; this is the
        # property AIQ-33 cares about. 2 GiB is generous for a single layer.
        con.execute("SET memory_limit='2GB'")
        con.execute("PRAGMA threads=4")

    if method == "lex_sort":
        fn = functools.partial(counts_lex_sort, states)
    elif method == "duckdb-blob":
        fn = functools.partial(counts_duckdb_blob, states, con)
    elif method == "duckdb-cols":
        fn = functools.partial(counts_duckdb_cols, states, con)
    else:
        raise ValueError(method)

    result = fn()  # warmup + correctness sample
    base = _peak_rss_mib()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ms = (time.perf_counter() - t0) / reps * 1000.0
    peak = _peak_rss_mib() - base
    # Send back a small fingerprint of the counts for cross-method validation.
    fp = (int(result.size), int(result.sum()), tuple(int(x) for x in result[-5:]))
    q.put((ms, peak, fp))


def run_method(method, n_rows, bpr, n_distinct, reps):
    """Spawn a child for one (method, size) cell and collect its result."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(method, n_rows, bpr, n_distinct, reps, q))
    p.start()
    try:
        ms, peak, fp = q.get(timeout=1800)
    except Exception:  # noqa: BLE001 -- bench guard: any child failure -> skip cell
        p.terminate()
        return None
    p.join()
    return ms, peak, fp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=float, default=1e7)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    row_sizes = [int(n) for n in (1e5, 1e6, 1e7) if n <= args.max_rows]
    # bytes-per-row -> neurons: 1->8, 8->64, 64->512
    byte_sizes = [1, 8, 64]
    n_distinct = 2048  # realistic group cardinality for an early CNN layer

    methods = ["lex_sort", "duckdb-blob", "duckdb-cols"]

    header = (
        f"{'n_rows':>9} {'bytes':>5} {'neurons':>7} | "
        f"{'lex ms':>9} {'lexMiB':>7} | "
        f"{'blob ms':>9} {'blobMiB':>8} | "
        f"{'cols ms':>9} {'colsMiB':>8} | {'match':>6}"
    )
    print(f"distinct-microstate palette = {n_distinct}, reps = {args.reps}")
    print(header)
    print("-" * len(header))

    for n_rows in row_sizes:
        for bpr in byte_sizes:
            neurons = bpr * 8
            results = {}
            for m in methods:
                results[m] = run_method(m, n_rows, bpr, n_distinct, args.reps)

            # Cross-validate: same number of groups, same total, same tail.
            fps = [results[m][2] for m in methods if results[m] is not None]
            match = "yes" if len(set(fps)) == 1 and len(fps) == len(methods) else "NO"

            def cell(m, results=results):
                r = results[m]
                return (float("nan"), float("nan")) if r is None else (r[0], r[1])

            l_ms, l_mb = cell("lex_sort")
            b_ms, b_mb = cell("duckdb-blob")
            c_ms, c_mb = cell("duckdb-cols")
            print(
                f"{n_rows:>9} {bpr:>5} {neurons:>7} | "
                f"{l_ms:>9.1f} {l_mb:>7.0f} | "
                f"{b_ms:>9.1f} {b_mb:>8.0f} | "
                f"{c_ms:>9.1f} {c_mb:>8.0f} | {match:>6}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
