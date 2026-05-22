//! `_lex_sort`: radix-style byte-wise lex-sort over the rows of a 2-D u8 array.
//!
//! Port of the Cython `__byte_sort` + `__lex_sort` pair. The implementation
//! is purely scalar in the Cython original; this port preserves the
//! algorithmic structure (in-place index permutation, recursive descent
//! over columns from least-significant byte to most).

use ndarray::{Array1, ArrayView2};

/// Sort `index[start..end]` so that rows are partitioned by the value of
/// column `col`. Writes per-value counts to `counts[0..256]`.
fn byte_sort(states: ArrayView2<'_, u8>, index: &mut [i64], start: usize, end: usize, col: usize, counts: &mut [usize; 256]) {
    counts.fill(0);

    // Count occurrences of each byte value in the column.
    for i in start..end {
        let row = index[i] as usize;
        counts[states[(row, col)] as usize] += 1;
    }

    // Compute offsets and an end pointer per value.
    let mut offsets: [usize; 256] = [0; 256];
    let mut next_offset: [usize; 256] = [0; 256];
    let mut remaining: [u8; 256] = [0; 256];
    let mut num_partitions: usize = 0;
    let mut total: usize = 0;
    for v in 0..256 {
        let c = counts[v];
        if c > 0 {
            offsets[v] = total;
            total += c;
            remaining[num_partitions] = v as u8;
            num_partitions += 1;
        }
        next_offset[v] = total;
    }

    // Swap indices into their value-partitioned positions.
    if num_partitions > 0 {
        for partition_idx in 0..(num_partitions - 1) {
            let val = remaining[partition_idx] as usize;
            while offsets[val] < next_offset[val] {
                let ind = offsets[val];
                let row = index[start + ind] as usize;
                let v = states[(row, col)] as usize;
                if v == val {
                    offsets[val] += 1;
                    continue;
                }
                let other_offset = offsets[v];
                offsets[v] += 1;
                index.swap(start + ind, start + other_offset);
            }
        }
    }
}

/// Recursive lex-sort over columns, descending from MSB to LSB. Returns
/// the next free position in `bin_edges`.
fn lex_sort_recursive(
    states: ArrayView2<'_, u8>,
    index: &mut [i64],
    start: usize,
    end: usize,
    col: usize,
    bin_edges: &mut Vec<i64>,
) {
    let mut counts: [usize; 256] = [0; 256];

    if col > 0 {
        byte_sort(states, index, start, end, col, &mut counts);
        let mut total: usize = 0;
        for v in 0..256 {
            let c = counts[v];
            if c == 0 {
                continue;
            }
            if c == 1 {
                let last = *bin_edges.last().unwrap();
                bin_edges.push(last + 1);
                total += 1;
                continue;
            }
            lex_sort_recursive(
                states,
                index,
                start + total,
                start + total + c,
                col - 1,
                bin_edges,
            );
            total += c;
        }
    } else {
        byte_sort(states, index, start, end, col, &mut counts);
        for v in 0..256 {
            let c = counts[v];
            if c > 0 {
                let last = *bin_edges.last().unwrap();
                bin_edges.push(last + c as i64);
            }
        }
    }
}

/// Lex-sort the rows of `states[..state_count]` so identical rows are
/// adjacent. Returns `(bin_edges, index)`:
///
/// - `index` is a permutation of `0..state_count` such that
///   `states[index, :]` is sorted lex-ordered (last column primary).
/// - `bin_edges` partitions `index` into runs of identical rows.
///   `bin_edges[0] == 0` and `bin_edges[-1] == state_count`.
pub fn lex_sort(states: ArrayView2<'_, u8>, state_count: usize) -> (Array1<i64>, Array1<i64>) {
    let mut index: Vec<i64> = (0..state_count as i64).collect();
    let mut bin_edges: Vec<i64> = vec![0];

    if state_count == 0 || states.ncols() == 0 {
        return (Array1::from(bin_edges), Array1::from(index));
    }

    let last_col = states.ncols() - 1;
    lex_sort_recursive(states, &mut index, 0, state_count, last_col, &mut bin_edges);

    (Array1::from(bin_edges), Array1::from(index))
}
