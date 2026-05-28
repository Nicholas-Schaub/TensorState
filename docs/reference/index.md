# API Reference

Auto-generated documentation from the public TensorState API. Sections:

- **[Layers](layers.md)** — the `StateCaptureHook` (PyTorch forward hook) and
  its abstract base.
- **[States](states.md)** — `compress_states`, `decompress_states`,
  `sort_states`. CPU bit-packing is in the Rust extension; GPU is a Triton
  kernel.
- **[TensorState](tensorstate.md)** — top-level functions: `attach`, `match`,
  `layers`, `layer`, `entropy`, `efficiency`, `network_efficiency`, `aIQ`,
  `reset_efficiency_model`, `remove_state_layers`, `zero_info`.
  The legacy `build_efficiency_model` still works and dispatches to the
  same machinery as `attach`; it will be removed in a future release.
- **[Dependency](dependency.md)** — graph-based layer-dependency tracking
  used by the apoptotic pruning pipeline (added in v0.5).
