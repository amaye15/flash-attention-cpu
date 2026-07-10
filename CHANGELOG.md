# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
