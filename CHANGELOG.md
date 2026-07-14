# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Implemented all three tiers of `UNSAFE.md`'s unsafe-minimization plan:
  - **Tier 0**: enabled `clippy::missing_safety_doc`,
    `clippy::undocumented_unsafe_blocks`, and `#![warn(unsafe_op_in_unsafe_fn)]`
    (verified to work on stable Rust, edition 2021, no bump needed), then
    wrote real `// SAFETY:` justifications for every `unsafe fn`/`unsafe {}`
    in the crate (160+ locations) — not placeholders, actual per-function
    and per-block reasoning (CPU-feature preconditions, loop-guard bounds
    arguments).
  - **Tier 1**: redesigned the `Kernel` trait around a "CPU feature token"
    pattern. Every method is now a safe `&self` fn; `Avx2Kernel`/
    `Avx512Kernel`/`Sse41Kernel::new() -> Option<Self>` perform
    `is_x86_feature_detected!` exactly once, and simply possessing the
    resulting value is the proof every other method needs;
    `NeonKernel`/`Simd128Kernel`/`ScalarKernel::new() -> Self` are
    infallible (mandatory baseline / compile-time `#[cfg(...)]` gate / no
    precondition at all, respectively). Measured result: `unsafe fn` count
    133 → 62 (−53%); `v1.rs`/`v2.rs`/`v3.rs` went from 34 `unsafe` blocks to
    0; `scalar.rs` (which never did anything actually unsafe internally,
    only forced into `unsafe fn` by the old trait signature) went from 10
    `unsafe fn`/5 blocks to 0. No behavior or performance change — the ZST
    tokens monomorphize away identically to the old static dispatch.
  - **Tier 2**: prototyped and benchmarked adopting the `pulp` crate for
    the register-blocked arithmetic primitives, not completed. Real,
    2-way-unrolled-vs-2-way-unrolled benchmarks on this sandbox's aarch64
    host show pulp performs on par with (sometimes faster than) this
    crate's hand-tuned `dot_neon` once matched to the same register-blocking
    shape — a naive single-accumulator pulp version had measured
    1.2–1.9x slower first, which is why the shape match mattered. But
    `exp()`'s IEEE-754 bit-manipulation technique (used by all five
    existing kernels) needs integer arithmetic pulp's `Simd` trait doesn't
    expose at all — confirmed by exhaustively grepping its source, not a
    missed API. The floor/round half of that gap *is* solved (the "magic
    number" rounding trick was checked numerically against 1.78M points in
    this crate's actual clamp range and measured zero difference in final
    `exp()` accuracy), but without integer add or a float-to-int
    conversion there's no way to finish the exponent-bit reconstruction.
    Completing a "pulp port" today would mean either overstating the
    result (still shipping hand-written unsafe for `exp()`) or inventing
    and separately validating a new float-only `exp()` algorithm — neither
    done here. See `UNSAFE.md`'s Tier 2 section for the full writeup and
    what a future attempt would need.

- `UNSAFE.md`: a from-scratch audit of this crate's unsafe code (133
  `unsafe fn`, ~140 `unsafe` blocks, measured directly, not estimated),
  researching how to minimize it without touching hot-path codegen.
  Distinguishes what's actually irreducible (calling `std::arch`
  intrinsics) from what's pure ceremony — `scalar.rs`'s entire kernel (10
  `unsafe fn`) does nothing unsafe internally, forced only by `Kernel`'s
  shared trait signature, and `v1.rs`/`v2.rs`/`v3.rs`'s ~35 call-site
  `unsafe` blocks are thin wrappers already downstream of a completed
  `is_x86_feature_detected!` check. Lays out a phased plan: Tier 0
  (`clippy::missing_safety_doc`/`undocumented_unsafe_blocks`,
  `unsafe_op_in_unsafe_fn` — all verified to work on stable Rust, edition
  2021, no bump needed), Tier 1 (a "CPU feature token" pattern eliminating
  unsafe at the orchestration layer, no new dependency), and Tier 2 (a
  scoped evaluation of adopting the `pulp` crate for 4 of 5 SIMD backends
  — verified directly, not from docs alone, including that its AVX-512
  support now compiles on stable Rust as of v0.22.3, that it has no
  packed-floor primitive this crate's `exp` implementations depend on,
  and that it has no sub-AVX2 x86 tier so the just-shipped `sse41.rs`
  would need to stay hand-written regardless). Also directly verified
  (installed nightly + Miri, ran it for real) that this crate's AVX2,
  AVX-512F, and SSE4.1 kernels all pass completely under Miri via
  cross-target interpretation, while the NEON kernel currently fails on
  an unimplemented horizontal-add intrinsic — concrete grounds for adding
  a scoped Miri CI job rather than assuming Miri either works or doesn't.
  No code changed this round — research and plan only.

## [0.2.0] - 2026-07-14

First published release on crates.io.

### Added

- New `src/sse41.rs` kernel: a fourth x86_64 SIMD tier, slotted into
  `v1`/`v2`/`v3`'s dispatch between AVX2+FMA and the scalar fallback via
  `is_x86_feature_detected!("sse4.1")`. Closes the gap ROADMAP.md item 9
  identified — x86_64 CPUs without AVX2 (EVC/Hyper-V-masked cloud VMs,
  budget/embedded Atom-class chips, and the x86-64-v2 floor RHEL/Anaconda
  are standardizing on for 2026) previously fell straight through to
  scalar. Same 4-lane shape as `neon.rs`/`simd128.rs` (no native FMA at
  this tier — separate mul+add, composed inline since there's no
  conditional branch to consolidate here), including the same vectorized
  `exp` algorithm (`_mm_floor_ps`-based range reduction — an SSE4.1
  instruction, not available in baseline SSE2). Type-checks and passes
  clippy cross-compiled to `x86_64-apple-darwin` (this sandbox can't
  execute x86_64 binaries, same caveat AVX-512F already carries), but
  correctness is validated by real execution once in CI: unlike AVX-512F,
  SSE4.1 is present on every real x86_64 CI runner, so the new
  `sse41`-specific unit tests (mirroring `avx2.rs`'s test shape, plus
  `scalar_and_sse41_agree_with_each_other` mutual-agreement checks in each
  of `v1.rs`/`v2.rs`/`v3.rs`) actually execute for real there. No isolated
  throughput number published this round — the crate's public API (which
  `bench_quick.rs`/`bench.rs` are limited to) dispatches to whatever the
  real CI runner's best tier is (AVX2), so getting a direct SSE4.1-vs-scalar
  number would need exposing crate-private dispatch functions publicly
  just for benchmarking purposes, not judged worth the API-surface cost.
- `ROADMAP.md` items 9-10: a CPU-hardware-coverage research pass (per
  request, focused on expanding the *number* of supported hardware types
  rather than precision variants). Found a real, immediately-actionable
  gap in an architecture already supported: x86_64 CPUs without AVX2
  (EVC/Hyper-V-masked cloud VMs, budget/embedded Atom-class chips, and the
  x86-64-v2 floor RHEL/Anaconda are standardizing on for 2026) currently
  fall straight through to the scalar fallback with no SIMD kernel at all.
  Verified an SSE4.1 tier — including the packed-floor instruction
  (`_mm_floor_ps`) the `exp` implementation needs — compiles cleanly on
  stable Rust with no toolchain blocker, unlike every other architecture
  surveyed this round: PowerPC64 VSX, s390x vector facility, LoongArch
  LSX/LASX, and 32-bit Arm (`armv7`) NEON are all confirmed nightly-only
  by direct compilation (cross-compiled where this sandbox's own aarch64
  host can't run them natively), not secondhand research. 32-bit Arm NEON
  in particular is a real, previously-unnoticed gap (this crate's existing
  NEON kernel is `aarch64`-only), just currently blocked at the Rust
  toolchain level rather than by this crate. No implementation this
  round — documentation only, per current priorities.
- `ROADMAP.md` item 2 (bf16 dot-product acceleration) re-verified against
  stable Rust directly rather than left on the strength of the earlier
  research pass: AVX512_BF16 (`__m256bh`/`__m512bh`, `_mm256_dpbf16_ps`,
  `_mm512_dpbf16_ps`, `_mm256_cvtneps_pbh`) compiles cleanly on stable
  rustc 1.93 (unverified by real hardware in this sandbox, same caveat the
  AVX-512F path already carries) — but Arm's `bfdot` equivalent
  (`bfloat16x8_t`, `vbfdotq_f32`, `vcvtq_low_bf16_f32`) does **not** exist
  on stable Rust, confirmed by compiling directly and natively on this
  sandbox's own aarch64 host (not cross-compiled) and getting
  "cannot find type/function in this scope." The item's original framing
  assumed both were straightforwardly available; corrected, with scoping
  options for a future round now split out (x86_64-only `vdpbf16ps` path
  vs. a portable widen-in-kernel path that doesn't need either dot-product
  ISA extension vs. both, sequenced). No code changed this round — this
  is a documentation-only correction, implementation deferred pending a
  scoping decision.
- WASM `relaxed-simd` real FMA: `simd128.rs` gained a single `fma128_ps`
  helper (`a*b+c`) used at every accumulation site (`dot`, `dot4`,
  `dot4x4`, `axpy`, `pv4`, `exp128_ps`'s polynomial) that selects the real,
  single-rounding `f32x4_relaxed_madd` under a further opt-in
  `target_feature = "relaxed-simd"` (stable since Rust 1.82, standardized
  as WebAssembly's Phase-4 relaxed-simd proposal since 2024 — this crate's
  own docs previously, incorrectly, called it not-yet-standard) and falls
  back to the previous separate multiply-then-add otherwise. Validated
  with real execution both ways via `wasm-pack test --node`; CI gained a
  dedicated `wasm-relaxed-simd` job (guarded by a
  `relaxed_simd_target_feature_is_actually_enabled` test so it can't
  silently degrade to testing the unfused path). See
  [ROADMAP.md](ROADMAP.md#1-wasm-relaxed-simd-real-fma-doc-correction--open-opportunity).
  No throughput number is published for this — this repo has no wasm32
  timing harness (`Instant` panics without a JS shim there, and
  Criterion/`bench_quick.rs` are both native-only); the guaranteed benefit
  is numerical (one rounding instead of two), consistent with this
  project's practice of only publishing measured numbers, not assumed ones.
- Fixed a latent, pre-existing gap surfaced while validating the above:
  `simd128.rs`'s own inline `#[cfg(test)] mod tests` (all ten of them,
  including the new `fma_matches_mul_add`) were plain `#[test]` functions,
  which `wasm-bindgen-test`'s harness silently never runs on
  `wasm32-unknown-unknown` (`wasm-pack test --node` reported "no tests to
  run!" for that binary) — converted to `#[wasm_bindgen_test]` so they
  actually execute now.
- Causal masking, in `v1.rs`/`v2.rs`/`v3.rs`, replaces a scalar
  branch-per-element loop (`if k_start + j > global_i { *s = -inf }`) with
  a computed cutoff column + `[f32]::fill()` — the mask condition is
  monotonic in `j` for a fixed row, so this is an exact reimplementation,
  not an approximation. `v1.rs` additionally gained the tile-skip guard
  `v2.rs`/`v3.rs` already had, so fully-past KV blocks no longer enter the
  masking loop at all. Isolated microbenchmark on the masking pass itself:
  3.9-4.5x faster (NEON, correctness-checked against the branchy
  reference). End-to-end this is Amdahl's-law-limited (masking only ever
  touches one tile per query block), measured via drift-controlled
  Criterion pairs at 4.1-6.3% faster at `seq_len=128`, 2.7-4.1% at
  `seq_len=512`, within noise at `seq_len=2048` — see the README's
  Benchmarks section for the full numbers and the methodological note on
  controlling for this machine's thermal drift.
- Persistent, cross-commit benchmark history: `examples/bench_quick.rs`
  gained a `--csv` mode (tagged with commit/os/arch/thread-count, no new
  dependency — Unix-epoch timestamp instead of pulling in `chrono`/`time`);
  `benches/history.csv` (git-tracked) accumulates rows over time; a new
  `examples/bench_compare.rs` diffs two commits' worth of it, joined on
  everything except the timing so it never compares across different
  targets/thread-counts by accident. CI appends automatically after every
  push to `main` (one job per OS leg uploads its CSV as an artifact, a
  single downstream job concatenates and commits, idempotent against
  job re-runs via a per-commit dedup check); the same `--csv` flag works
  identically for a deliberate local snapshot.
- Project scaffolding: dual MIT/Apache-2.0 license, CI (GitHub Actions:
  native test matrix, x86_64 cross-check, WASM via `wasm-pack test --node`,
  MSRV check, `cargo-deny`), `cargo-fuzz` differential-fuzzing harness,
  `CONTRIBUTING.md`, this changelog.
- MSRV set to 1.89 (verified empirically — the AVX-512F intrinsics are the
  binding constraint; NEON/SIMD128/rayon all work much earlier).
- Register-blocked micro-kernel, in two layers:
  - `Kernel::dot4`: 4 query rows blocked against 1 key row at a time,
    sharing the key row's vector loads across four independent FMA
    accumulator chains.
  - `Kernel::dot4x4`: 4 query rows *and* 4 key rows blocked together
    (BLIS/OpenBLAS-style two-sided packed register tiling), sharing both
    operands' loads across 16 independent FMA accumulator chains — the
    QK^T bulk path, falling back to `dot4` for the `block_size_kv % 4`
    column remainder and to `dot` for the `block_size_q % 4` row remainder.
  - `Kernel::pv4`: PV accumulation for a 4-query-row group against a whole
    KV tile, keeping accumulator registers resident across the entire
    key/value sweep and writing back to memory once per chunk instead of
    once per key/value row (replaces the earlier `axpy4`, which read/wrote
    the accumulator through memory on every row). Processes 2 native-width
    chunks (8 independent accumulator chains) per outer step rather than 1
    (4 chains): 4 chains isn't enough concurrent work to hide the FMA
    latency of each row's long sequential `bc`-sweep dependency.
  - Implemented for all five kernels (scalar, AVX2, AVX-512F, NEON,
    SIMD128) and wired into `v1`/`v2`/`v3`. Each layer was validated first
    as an isolated NEON micro-benchmark before being wired in: `dot4`/`pv4`'s
    one-sided predecessor measured 1.74x/1.29x (`d_head=64`/`128`); `dot4x4`
    measured a further ~1.5-1.6x over that; `pv4`'s first (4-chain) version
    measured ~1.2x over a naive one-`axpy`-per-row PV loop, and doubling to
    8 chains measured a further ~1.8-2.3x on top of that.
- `Kernel::sub_exp_sum_inplace` fuses the online-softmax subtract-max,
  `exp`, and sum-reduction into a single SIMD pass per kernel (previously
  three separate traversals of the score row: max, subtract+exp, sum).
  `Kernel::max_reduce4`/`sub_exp_sum_inplace4` additionally row-block this
  bookkeeping by 4 (the same ILP problem as `pv4`'s: one row's reduction is
  a single dependency chain, and rows are mutually independent within a
  tile), measured ~1.8-2.5x over one-row-at-a-time.
- `tests/correctness.rs::block_size_q_not_a_multiple_of_four` and
  `::block_size_kv_not_a_multiple_of_four`, covering the register-blocked
  micro-kernel's row and column remainder paths.
- CI now runs `cargo run --release --example bench_quick` (plus a CPU-flags
  diagnostic) on every `test` job leg, giving real — if noisier,
  shared-runner — x86_64 (AVX2/AVX-512F) and second-aarch64 timing data for
  the first time in this project's history.
- `tests/correctness.rs`: `shape_mismatches_panic` and
  `multihead_shape_mismatch_panics` actually exercise the `# Panics`
  contract every public entry point documents (previously asserted only in
  doc comments, never in a test); `zero_sized_dimensions_leave_out_untouched`
  covers `seq_len_q`/`seq_len_k`/`d_head == 0` each independently;
  `zero_block_size_is_clamped` confirms `block_size_q`/`block_size_kv: 0`
  degrade to a correct (if slow) `block_size` of 1 rather than a
  division-by-zero or silently wrong output.
- `#![warn(missing_docs)]` at the crate root — coverage was already
  complete (confirmed via `RUSTFLAGS="-W missing_docs" cargo build --lib`,
  natively and cross-compiled to `x86_64-apple-darwin`, both clean), this
  just guards against future regressions.
- CI: a `cargo doc --no-deps --lib` step (`RUSTDOCFLAGS=-D warnings`) in the
  `test` job, catching a broken intra-doc link or missing-docs regression
  before it would otherwise only surface on docs.rs after a release.
- `SECURITY.md` (private vulnerability reporting via GitHub's advisory
  flow) and `.github/dependabot.yml` (weekly `cargo` — both the root crate
  and `fuzz/`'s separate workspace — and `github-actions` update PRs,
  grouped by minor/patch) — motivated directly by the `crossbeam-epoch`
  advisory (RUSTSEC-2026-0204) that broke `cargo-deny` on `main` earlier
  in this project's history and was only caught because CI happened to run
  again; Dependabot should open that kind of PR proactively instead.
- `ROADMAP.md`: research-backed prioritization of README's Extension
  points, checked against current upstream/ecosystem status rather than
  left as a static wishlist. Corrects a stale claim in both README and
  `simd128.rs`'s module docs — WASM `relaxed-simd`'s real FMA was described
  as "not-yet-standard," but the proposal reached Phase 4 in 2024 and Rust
  stabilized the corresponding intrinsics in 1.82, under this crate's MSRV;
  it's a real, currently-unused near-term opportunity, not a future one.
  Also sequences bf16 ahead of int8/VNNI (bf16 needs no calibration
  infrastructure, int8 does — now backed by real INT8-attention accuracy
  data rather than a guess), and adds Arm SVE and SME2/KleidiAI as newly
  identified extension points (both nightly-only or lacking any stable
  Rust intrinsics path today, so tracked rather than pursued). A same-core
  algorithmic softmax reformulation (FLASH-D, ISLPED 2025) originally
  flagged here as the one item needing no new ISA at all was subsequently
  worked through in full and **not adopted** — see the `ROADMAP.md` update
  further up this changelog and item 3 there for the derivation showing it
  trades v2's already-deferred single division for a division per tile,
  a net regression rather than a win for this crate's tile-blocked design.

### Fixed

- `.claude/` (local Claude Code assistant state — scheduled-task locks,
  not source) was being swept into the published crate package by cargo's
  default packaging (anything not git-ignored gets included); added to
  `.gitignore`, confirmed via `cargo package --list` that it no longer
  appears in the package contents.

### Changed

- `run_v1`/`run_v2`/`run_v3` now use Rayon's `for_each_init` to allocate each
  query block's scratch buffers (`m`/`l`/`acc`/scores) once per worker thread
  instead of once per block.
- Measured net effect on this machine (Apple M4, aarch64, NEON kernel),
  across all three rounds of the above: roughly 4x single-threaded and
  ~20.7x default-multithreaded speedup over naive for `flash_attention_v2`
  at `seq_len=2048` (up from ~1.5x/~8-9x before any of these rounds);
  `naive_attention` is unaffected throughout (it doesn't use the `Kernel`
  trait). See the README's Benchmarks section for full before/after tables
  and the isolated-microbenchmark numbers behind each change.

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
