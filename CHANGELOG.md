# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffolding: dual MIT/Apache-2.0 license, CI (GitHub Actions:
  native test matrix, x86_64 cross-check, WASM via `wasm-pack test --node`,
  MSRV check, `cargo-deny`), `cargo-fuzz` differential-fuzzing harness,
  `CONTRIBUTING.md`, this changelog.
- MSRV set to 1.89 (verified empirically — the AVX-512F intrinsics are the
  binding constraint; NEON/SIMD128/rayon all work much earlier).
- Register-blocked micro-kernel: `Kernel::dot4`/`Kernel::axpy4` process 4
  query rows at once, sharing each streamed `K`/`V` row's vector loads across
  four independent FMA accumulator chains (BLIS/OpenBLAS-style register
  tiling), implemented for all five kernels (scalar, AVX2, AVX-512F, NEON,
  SIMD128) and wired into the QK^T/PV loops of `v1`/`v2`/`v3` with a
  scalar-row fallback for the `block_size_q % 4` remainder. Validated first
  as an isolated NEON micro-benchmark (1.74x at `d_head=64`, 1.29x at
  `d_head=128`) before being wired in.
- `Kernel::sub_exp_inplace` fuses the online-softmax subtract-max and `exp`
  steps into a single SIMD pass per kernel (previously two separate
  traversals of the score row).
- `tests/correctness.rs::block_size_q_not_a_multiple_of_four`, covering the
  register-blocked micro-kernel's scalar-row remainder path.

### Changed

- `run_v1`/`run_v2`/`run_v3` now use Rayon's `for_each_init` to allocate each
  query block's scratch buffers (`m`/`l`/`acc`/scores) once per worker thread
  instead of once per block.
- Measured net effect of the above (Apple M4, aarch64, NEON kernel): roughly
  1.9-2x single-threaded speedup over the pre-optimization baseline across
  `flash_attention_v1/_v2/_v3`, causal and non-causal, at every benchmarked
  `seq_len`; `naive_attention` is unaffected (it doesn't use the `Kernel`
  trait). See the README's Benchmarks section for full before/after tables.

## [0.1.0]

Initial release.

### Added

- Three explicit FlashAttention algorithm variants, matching the real
  algorithmic deltas between the published papers rather than one blended
  implementation:
  - `flash_attention_v1` (Dao et al., 2022): output accumulator normalized
    on every KV-block step, no causal tile-skip.
  - `flash_attention_v2` (Dao, 2023): deferred single normalization, causal
    tile-skip. `flash_attention`/`flash_attention_multihead` alias this
    version for convenience.
  - `flash_attention_v3`: this crate's same-core, instruction-level-
    parallelism analog of FlashAttention-3's compute/softmax overlap
    (software-pipelined score computation) — explicitly not a port of FA3's
    actual Hopper-specific warp-specialization/async-tensor-core mechanism,
    which has no same-core CPU equivalent.
  - Batched multi-head entry points (`flash_attention_multihead_v1/_v2/_v3`)
    for all three.
- Hand-vectorized SIMD kernels behind a shared `Kernel` trait, covering
  every major CPU target: AVX-512F and AVX2+FMA on x86_64 (runtime-selected
  via `is_x86_feature_detected!`), NEON on aarch64 (unconditional — it's
  AArch64 baseline), and WASM SIMD128 on wasm32 (compile-time opt-in via
  `target_feature = "simd128"`, since WASM has no runtime feature
  detection), with a portable scalar fallback everywhere else.
- `FlashAttentionConfig` for tuning block sizes and enabling causal masking.
- `naive::naive_attention`, an O(n²)-memory reference implementation used
  as the correctness oracle throughout the test suite.
