# API Reference

Auto-generated documentation from the public TensorState API. Sections:

- **[Layers](layers.md)** — the `StateCaptureHook` (PyTorch forward hook) and
  its abstract base.
- **[States](states.md)** — `compress_states`, `decompress_states`,
  `sort_states`. CPU bit-packing is in Cython; GPU is a Triton kernel.
- **[TensorState](tensorstate.md)** — top-level functions: `entropy`, `aIQ`,
  `network_efficiency`, `build_efficiency_model`, `reset_efficiency_model`,
  `zero_info`.
- **[Dependency](dependency.md)** — graph-based layer-dependency tracking
  used by the apoptotic pruning pipeline (added in v0.5).
