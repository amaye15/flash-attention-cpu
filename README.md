# flash-attention-cpu

[![CI](https://github.com/amaye15/flash-attention-cpu/actions/workflows/ci.yml/badge.svg)](https://github.com/amaye15/flash-attention-cpu/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/flash-attention-cpu.svg)](https://crates.io/crates/flash-attention-cpu)
[![docs.rs](https://docs.rs/flash-attention-cpu/badge.svg)](https://docs.rs/flash-attention-cpu)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Pure-Rust, CPU-optimized Flash Attention: three explicit algorithm variants
(v1, v2, v3) built on tiled online-softmax, hand-vectorized SIMD (AVX-512F
and AVX2+FMA on x86_64, NEON on aarch64, SIMD128 on wasm32) with a portable
scalar fallback, and Rayon parallelism. No BLAS, no C bindings, no
`nightly`.

## Design

All three variants share the same tiling skeleton, adapted for CPU cache
hierarchy rather than GPU SRAM:

- Split `Q` into row blocks (`Br`) and `K`/`V` into row blocks (`Bc`).
- For each `Q` block, sweep over `K`/`V` blocks maintaining a running max
  `m`, running sum `l`, and an output accumulator — the usual online-softmax
  rescaling trick — so the full `[seq_len_q, seq_len_k]` score matrix never
  exists. Peak extra memory is `O(block_size_q * (d_head + block_size_kv))`,
  independent of sequence length.

Where they differ is exactly where the published FlashAttention papers
differ, translated into CPU terms:

- **v1** ([`flash_attention_v1`], Dao et al., 2022 — Algorithm 1): the output
  accumulator is fully normalized (divided by the running sum) and then
  un-normalized again on **every** KV-block step, mirroring the original
  paper's need to write a consistent `O` back to HBM after each step. On CPU
  there's no HBM round-trip forcing this, so it's strictly extra
  non-matmul FLOPs, kept here as a faithful, honestly-slower comparison
  point. Causal masking also has no early-exit: fully-future KV tiles are
  computed and then masked rather than skipped, matching the original
  paper (which didn't have that optimization).
- **v2** ([`flash_attention_v2`], Dao, 2023): normalization is deferred to a
  single division after the whole KV sweep, and causal masking skips
  fully-future `K`/`V` tiles outright (the loop `break`s once a tile is
  entirely beyond the diagonal, since tiles are visited in increasing key
  order) rather than computing and masking them. This was the crate's
  original (unversioned) algorithm.
- **v3** ([`flash_attention_v3`], loosely inspired by FlashAttention-3, Shah
  et al., 2024): v2's algorithm, plus the next KV tile's score computation
  (`QK^T`, independent FMA-heavy work) is software-pipelined one step ahead
  of the current tile's online-softmax-update + PV finish (which has a
  longer-latency `exp` in its dependency chain). **This is not a port of
  FA3's actual mechanism** — FA3's real headline feature is Hopper-specific
  warp specialization with asynchronous tensor-core (WGMMA/TMA) pipelines,
  plus FP8 low-precision numerics, both GPU-hardware-specific concepts with
  no same-core CPU equivalent. What's implemented is a program-order
  restructuring that gives the compiler's instruction scheduler and the
  CPU's out-of-order execution window more independent work to interleave —
  the closest same-core analog of "overlap softmax with the next matmul."
  It's a hint, not a hardware guarantee: see Benchmarks below for what it
  actually measures as, which on this sandbox's hardware is a wash, not a
  win.

`flash_attention`/`flash_attention_multihead` remain as aliases to the v2
functions for backward compatibility with callers from before this crate had
explicit versions.

**SIMD.** Four accelerated kernels plus a portable fallback, all implementing
the same small [`Kernel`](src/kernel.rs) trait (dot product, axpy, in-place
scale, a max reduction, a fused subtract+`exp`+sum) so each variant's tiling
algorithm (`src/v1.rs`, `src/v2.rs`, `src/v3.rs`) is written once, generic
over whichever kernel gets selected:

**Register-blocked micro-kernel.** The naive way to call `dot`/`axpy` is once
per `(query_row, key_row)` pair — for a fixed query row, the whole `K`/`V`
tile gets re-scanned from scratch, with zero register reuse across query
rows. Two rounds of register blocking address this from both directions:

- **QK^T** (`Kernel::dot4x4`): 4 query rows *and* 4 key rows blocked
  together (BLIS/OpenBLAS-style two-sided register tiling) — both operands'
  vector loads are shared across the resulting 16 independent FMA
  accumulator chains. An earlier one-sided version (`Kernel::dot4`, still
  used for the `this_bc % 4` column remainder) only blocked the query side,
  leaving the four query-row vectors reloaded from memory on every key row;
  going two-sided measured a further ~1.5-1.6x on top of that (see
  Benchmarks).
- **PV** (`Kernel::pv4`): the mirror-image inefficiency was on the
  accumulator side, not the operand side — a naive "one `axpy` per V-row"
  loop reads/writes the whole output accumulator through memory once per
  V-row. `pv4` keeps each `d_head`-chunk's 4 accumulator registers resident
  across the *entire* key/value sweep for a 4-query-row group, writing back
  to memory only once per chunk — measured ~1.2x over the one-`axpy`-per-row
  pattern.

Both fall back to scalar-row/column handling (`dot4`, `dot`, plain `axpy`)
for the `block_size_q % 4` / `block_size_kv % 4` remainders.

| Kernel | File | Target | Lanes | Selection |
|---|---|---|---|---|
| AVX-512F | `avx512.rs` | x86_64 | 16 | runtime (`is_x86_feature_detected!("avx512f")`), checked first |
| AVX2+FMA | `avx2.rs` | x86_64 | 8 | runtime, checked if AVX-512F isn't available |
| NEON | `neon.rs` | aarch64 | 4 | unconditional — part of the mandatory AArch64 baseline |
| SIMD128 | `simd128.rs` | wasm32 | 4 | compile-time (`target_feature = "simd128"`, opt-in — see WASM below) |
| Scalar | `scalar.rs` | everywhere else | 1 | fallback when none of the above apply |

Runtime-detected kernels (AVX-512F, AVX2) mean the compiled binary is
portable and still fast wherever the feature actually exists, rather than
requiring `-C target-cpu=native` (which SIGILLs on older hardware). NEON has
no runtime check because AArch64 makes it mandatory. WASM has no runtime
check *available* — see below.

The vectorized `exp` in every SIMD kernel is a range-reduction +
degree-5 minimax-polynomial approximation (the same family of technique
used in most SIMD math libraries) rather than a scalar `libm` call per lane
— softmax is exp-dominated, so this matters. Measured max relative error vs
`f32::exp` across `x ∈ [-80, 80]`: **~1.2e-7** on every kernel (see each
module's `tests::exp_matches_std`). One numerical wrinkle: AVX2/AVX-512/NEON
all have a genuine fused multiply-add (`vfmadd*`/`vfmaq_f32`, single
rounding); WASM SIMD128's baseline instruction set doesn't (`dot`/`axpy`
there are a separate multiply then add — see `simd128.rs`'s module docs).

### WASM

Unlike the three native targets, WASM has no runtime CPU-feature-detection
mechanism the module can query — `simd128` support is a property of the
*build*, decided by whoever compiles the `.wasm`, not discovered at
execution time. So `src/simd128.rs` is gated on `target_feature = "simd128"`
at compile time rather than a runtime check: build with
`RUSTFLAGS="-C target-feature=+simd128"` (or a `.cargo/config.toml`
`[target.wasm32-unknown-unknown] rustflags` entry — this repo's own
`.cargo/config.toml` does exactly that, so `cargo test`/`cargo check
--target wasm32-unknown-unknown` from this checkout use it automatically)
to opt in; without it, `simd128.rs` doesn't even get compiled and every
variant falls back to `scalar.rs`. This only affects building this crate
*from this checkout* — a downstream `Cargo.toml` dependency on
`flash-attention-cpu` from crates.io doesn't inherit this repo's
`.cargo/config.toml`, so consumers targeting wasm32 need to set the flag
themselves to get the SIMD128 kernel instead of scalar.

Validated with real execution, not just compilation: `wasm-pack test --node`
runs `tests/wasm_simd.rs` (a `wasm-bindgen-test`-based suite, since plain
`#[test]` functions are silently not picked up by
`wasm-bindgen-test-runner`) against Node.js, checking `flash_attention_v1`/
`_v2`/`_v3` against the naive oracle across several shapes, plus a guard
test (`simd128_target_feature_is_actually_enabled`) asserting the feature
really is on — otherwise the other tests would quietly pass against the
scalar fallback instead of `Simd128Kernel`, defeating the point.

**Parallelism.** Rayon splits work across query blocks within a single
call, and across `batch * heads` in the `_multihead` entry points. The
nesting is intentional and cheap — Rayon's work-stealing scheduler handles
it — so the same code scales down to one long sequence and up to many
short ones without a separate code path.

## Correctness

`src/naive.rs` is a textbook implementation that materializes the full
score matrix — the O(n²) memory flash attention exists to avoid — and
serves as the correctness oracle. `tests/correctness.rs` checks all three
variants' public API against it across exact-block-multiple and
deliberately ragged sizes (so remainder-handling in both the tiling loop
and the SIMD tail loops gets exercised), causal and non-causal, and the
batched multi-head API, plus a `v1_v2_v3_mutually_agree` test — since all
three implement the same mathematical operation, they must agree with each
other, not just with the oracle independently (loose tolerance for v1,
which accumulates slightly more floating-point rounding from its extra
per-step normalize/denormalize round-trip; a tight tolerance between v2 and
v3, whose arithmetic is byte-for-byte identical per row), plus
`block_size_q_not_a_multiple_of_four` and `block_size_kv_not_a_multiple_of_four`
tests specifically exercising the register-blocked micro-kernel's
row/column remainder paths (1, 2, and 3 leftover rows or columns) with
`block_size_q`/`block_size_kv` values that aren't multiples of 4.
`src/v1.rs`, `src/v2.rs`, and `src/v3.rs` each additionally have internal
tests that call the scalar, AVX2, AVX-512F, NEON, and SIMD128 kernels
directly (bypassing dispatch), so every path is checked regardless of which
one a given host would auto-select; `v3` also has boundary-case tests for
its pipeline running zero, one, and multiple steady-state iterations.
`src/avx2.rs`, `src/avx512.rs`, `src/neon.rs`, and `src/simd128.rs` each
have their own kernel-level tests too (`exp_matches_std`,
`dot_matches_scalar`, `reductions_match_scalar`,
`axpy_and_scale_match_scalar`, `dot4_matches_four_dots`,
`dot4x4_matches_naive`, `pv4_matches_naive`). 38 tests total (26 unit + 11
integration + 1 doctest), all passing on this machine (aarch64 — the
AVX2/AVX-512-specific tests are `#[cfg(target_arch = "x86_64")]`-gated and
don't run here).

Cross-target validation, since a single dev machine can't natively execute
every architecture: `cargo check`/`cargo clippy --target x86_64-apple-darwin`
type-check the AVX-512/AVX2 path cleanly, but this sandbox has no working
Rosetta 2, so that code has **not** been executed here — only its
near-mechanical mirroring of the already-executed AVX2 algorithm (same
math, same coefficients, wider vectors) backs its correctness, pending
real x86_64 hardware. WASM SIMD128 *is* executed for real: `wasm-pack test
--node` builds for `wasm32-unknown-unknown` with `simd128` enabled and runs
`tests/wasm_simd.rs` under Node.js — 5 passing tests, including a guard
that asserts the feature is actually on (see the SIMD/WASM section above).

## Benchmarks

Measured in this sandbox: **Apple M4, 10 cores, aarch64** — the NEON kernel
(`src/neon.rs`) is what actually runs here, not the scalar fallback (see
Correctness above for how that's verified). Reproduce with
`cargo run --release --example bench_quick` or `cargo bench`; the AVX2
kernel will engage instead on x86_64 hardware, with its own numbers.
`d_head=64` unless noted; times in ms.

These numbers reflect a second round of optimization: a packed two-sided
`Kernel::dot4x4` micro-kernel for QK^T and a register-resident `Kernel::pv4`
for PV (see Design above), plus a fused subtract+`exp`+sum softmax pass —
on top of round 1's `dot4`/`axpy4` blocking and `for_each_init` buffer reuse.
Run-to-run variance on this shared dev laptop is real (repeated runs of the
same binary vary by up to ~20-30%, likely thermal/background-load
dependent) — the tables below are a representative run, and the
percentages in "Honest reading" are rounded accordingly rather than
over-precise.

**Single-threaded** (`RAYON_NUM_THREADS=1`, isolating algorithmic/SIMD
differences from Rayon's parallelism):

| seq_len | naive | v1 | v2 | v3 | v1 causal | v2 causal | v3 causal |
|--------:|------:|---:|---:|---:|----------:|----------:|----------:|
|  256 |  1.51 |  0.44 |  0.40 |  0.39 |  0.41 |  0.29 |  0.28 |
|  512 |  4.09 |  1.25 |  1.19 |  1.22 |  1.34 |  0.81 |  0.83 |
| 1024 | 16.50 |  4.99 |  4.93 |  4.92 |  5.35 |  2.87 |  2.79 |
| 2048 | 65.69 | 20.53 | 20.18 | 20.38 | 21.79 | 11.15 | 11.02 |
| 1024 (d=128) | 38.17 |  9.44 |  9.36 |  9.33 |  9.69 |  5.25 |  5.31 |

**Default** (Rayon parallelism active across query blocks, all 10 cores):

| seq_len | naive | v1 | v2 | v3 | v1 causal | v2 causal | v3 causal |
|--------:|------:|---:|---:|---:|----------:|----------:|----------:|
|  256 |  1.62 | 0.17 | 0.15 | 0.15 | 0.15 | 0.14 | 0.15 |
|  512 |  4.13 | 0.35 | 0.34 | 0.35 | 0.36 | 0.31 | 0.28 |
| 1024 | 16.92 | 1.08 | 1.12 | 1.06 | 1.12 | 0.73 | 0.62 |
| 2048 | 66.42 | 3.53 | 3.69 | 3.56 | 3.88 | 2.00 | 2.04 |
| 1024 (d=128) | 39.74 | 1.95 | 1.93 | 1.95 | 1.99 | 1.18 | 1.20 |

Cross-checked against `cargo bench` (Criterion, default threading) as a
direct before/after on the same benchmark IDs: every `flash`/`v1`/`v3`
case (causal and non-causal, `seq_len` 512-2048) came back **-15% to -25%
time** ("Performance has improved", `p < 0.05`) versus the round-1
baseline, while `naive` stayed flat (within ±3%, noise) — the pattern
expected, since `naive_attention` doesn't use the `Kernel` trait and so
isn't touched by any of this work.

**Honest reading of these numbers, on this hardware:**

- **This round adds a further ~1.1-1.4x on top of round 1's already-large
  win**, both single- and multi-threaded — smaller than the ~1.5-1.6x
  (QK^T) and ~1.2x (PV) improvements measured in isolated micro-benchmarks
  for the same change (see Design above), which is expected: QK^T and PV
  are each a *fraction* of total tile time (softmax bookkeeping, causal
  masking, and tiling overhead are unaffected by either change), so
  Amdahl's law dilutes the isolated per-phase win once it's wired into the
  full algorithm. Non-causal `v2` at `seq_len=2048` goes from ~2.8x over
  naive (round 1) to ~3.3x single-threaded, and from ~15.5x to ~18x at the
  default thread count.
- **Causal, v2/v3 still add a further ~1.8-2x on top of v1/non-causal** —
  from skipping whole future KV tiles instead of computing and masking
  them. This remains the one asymptotic, unambiguous improvement in this
  whole comparison, unaffected in kind by either round's micro-kernel work
  (though it compounds with it).
- **v3 vs v2 is still a wash on this hardware**, unchanged by this round —
  sometimes marginally faster, sometimes marginally slower, never by more
  than a few percent. Same caveat as before: v3's software pipelining is a
  program-order hint for the compiler/out-of-order engine, not a hardware
  guarantee of overlap. Don't take v3 on faith — measure it on your own
  target hardware.
- Rayon parallelism (default vs. single-threaded columns) gives roughly a
  5-5.5x multiplicative speedup at `seq_len=2048` on this 10-core machine,
  independent of and on top of the algorithmic/SIMD/micro-kernel
  differences above.
- **x86_64 is now measured for the first time in this project's history**
  (previously only cross-compiled/correctness-tested, never benchmarked):
  GitHub Actions' `ubuntu-latest`/`windows-latest` runners confirm
  `avx512f` is *not* present (only `avx2`+`fma`), so the AVX2 kernel is
  what actually runs there in CI. See
  [the CI logs](https://github.com/amaye15/flash-attention-cpu/actions/workflows/ci.yml)
  for real (shared-runner, noisier than this dedicated dev machine) AVX2
  timing — `dot4x4`'s 16 accumulators are comfortable on NEON/AVX-512 (32
  registers each) but leave no spare AVX2 YMM registers (only 16 total),
  so whether the two-sided packing helps or regresses on AVX2 specifically
  is a real open question this data starts to answer, not an assumption.

**Note on the causal speedup**: it comes from skipping whole `K`/`V` tiles
(v2/v3 only), so it scales with how many tiles are actually skippable — at
small `seq_len` relative to `block_size_kv`, the entire sequence is one
tile and there's nothing to skip, so causal ≈ non-causal there. For
short-sequence causal workloads (e.g. small-context autoregressive
decoding), a smaller `block_size_kv` trades a little non-causal throughput
for more skip granularity — tune `FlashAttentionConfig` for your workload.

Peak extra memory at `seq_len=4096`: naive's score matrix is **64.0 MB**;
v1/v2's tile scratch is **32.0 KB**; v3's is **64.0 KB** (double-buffered
for the pipeline) — all independent of `seq_len`.

## Usage

```rust
use flash_attention_cpu::{flash_attention, FlashAttentionConfig};

let (seq_len, d_head) = (1024, 64);
let q = vec![0.0f32; seq_len * d_head];
let k = vec![0.0f32; seq_len * d_head];
let v = vec![0.0f32; seq_len * d_head];
let mut out = vec![0.0f32; seq_len * d_head];

// Non-causal (flash_attention aliases flash_attention_v2)
flash_attention(&q, &k, &v, seq_len, seq_len, d_head, &FlashAttentionConfig::default(), &mut out);

// Causal (autoregressive)
let config = FlashAttentionConfig { causal: true, ..Default::default() };
flash_attention(&q, &k, &v, seq_len, seq_len, d_head, &config, &mut out);
```

Prefer explicit versions to be clear about which algorithmic tradeoffs
you're getting:

```rust
use flash_attention_cpu::{flash_attention_v1, flash_attention_v2, flash_attention_v3};
# use flash_attention_cpu::FlashAttentionConfig;
# let (seq_len, d_head) = (1024, 64);
# let q = vec![0.0f32; seq_len*d_head]; let k = q.clone(); let v = q.clone();
# let mut out = q.clone();
# let config = FlashAttentionConfig::default();
flash_attention_v1(&q, &k, &v, seq_len, seq_len, d_head, &config, &mut out);
flash_attention_v2(&q, &k, &v, seq_len, seq_len, d_head, &config, &mut out);
flash_attention_v3(&q, &k, &v, seq_len, seq_len, d_head, &config, &mut out);
```

Batched multi-head — layout `[batch, heads, seq, d_head]`, contiguous
(each version has a `flash_attention_multihead_vN` counterpart;
`flash_attention_multihead` aliases v2):

```rust
use flash_attention_cpu::flash_attention_multihead;
# use flash_attention_cpu::FlashAttentionConfig;
# let (batch, heads, seq_len, d_head) = (2, 8, 1024, 64);
# let q = vec![0.0f32; batch*heads*seq_len*d_head];
# let k = q.clone(); let v = q.clone();
# let mut out = q.clone();
flash_attention_multihead(&q, &k, &v, batch, heads, seq_len, seq_len, d_head, &FlashAttentionConfig::default(), &mut out);
```

`cargo run --release --example basic` for a runnable version of all of the
above.

## Extension points

Deliberately out of scope for this pass, but the architecture leaves room:

- ~~**AVX-512**~~ — implemented (`src/avx512.rs`), checked ahead of AVX2 in
  the x86_64 feature-detection chain. Type-checks and passes clippy
  cross-compiled to `x86_64-apple-darwin`, but hasn't been executed in this
  sandbox (no working Rosetta 2 here to run x86_64 binaries) — real
  hardware validation is still pending.
- ~~**NEON** (ARM/Apple Silicon)~~ — implemented (`src/neon.rs`); see SIMD
  above and the Benchmarks section for real numbers.
- ~~**WASM SIMD128**~~ — implemented (`src/simd128.rs`), opt-in via
  `target_feature = "simd128"` since WASM has no runtime feature detection;
  see the WASM subsection above. Real relaxed-simd FMA (WASM's proposal
  adding a fused multiply-add) is a further, not-yet-standard step beyond
  baseline `simd128` and isn't used here.
- **AVX-512 sub-extensions** (`avx512dq`/`avx512bw`/`avx512vl`/`avx512vnni`,
  etc.): `avx512.rs` only uses the baseline `avx512f`. VNNI in particular
  targets int8 dot-product acceleration, which would only matter alongside
  the lower-precision work below.
- ~~**Register-blocked micro-kernel**~~ — implemented in two rounds:
  one-sided blocking (`Kernel::dot4`/former `axpy4`), then a packed
  two-sided QK^T micro-kernel (`Kernel::dot4x4`) and a register-resident
  PV accumulator (`Kernel::pv4`), wired into `src/v1.rs`/`src/v2.rs`/
  `src/v3.rs`; see Design and Benchmarks above for what each measures as.
- **AVX2-specific packing factor**: `dot4x4`'s 16 accumulators fit
  comfortably in NEON/AVX-512's 32 registers but leave no spare AVX2 YMM
  registers (only 16 total) — implemented the same way on AVX2 for API
  consistency, but a narrower factor (e.g. 4x2) tuned specifically for
  AVX2's smaller register file is a real, not-yet-explored possibility if
  real x86_64 CI numbers (see Benchmarks) show it regressing there.
- **Lower precision** (`bf16`/`f16`/FP8): halves (or quarters) memory
  bandwidth, which is often the actual bottleneck at long sequence lengths,
  and is part of FlashAttention-3's actual numerics story (incoherent
  processing + FP8). Considered and explicitly deferred for `v3` here —
  it's a separate, substantial piece of work (quantization/calibration, a
  new low-precision kernel, accuracy validation) independent of the
  scheduling change `v3` focuses on.
- **Thread-level producer/consumer pipelining**: a more literal CPU analog
  of GPU warp specialization (one thread computing score tiles into a
  bounded queue, another consuming them for softmax+PV) was considered for
  `v3` and deferred in favor of the same-core software-pipelining approach
  actually implemented — real risk that thread hand-off/sync overhead
  exceeds any gain at CPU tile granularity.
- **RISC-V Vector (RVV)**: growing hardware relevance, but Rust's RVV
  intrinsics are still nightly-only, and RVV's variable-length-vector model
  (`vsetvli`-style, no fixed lane count known at compile time) doesn't map
  onto this crate's fixed-width `Kernel` trait as directly as SSE/NEON/wasm
  did — a real design question, not just a new file, so left until Rust's
  intrinsics stabilize.
- **Backward pass**: this is forward/inference only. Training needs the
  recomputation-based backward pass from the flash attention paper.

## Layout

```
src/
  lib.rs      public API + crate docs
  kernel.rs   Kernel trait (the primitives that differ per-backend)
  scalar.rs   portable fallback implementation
  avx2.rs     AVX2+FMA implementation (x86_64), incl. vectorized exp + its tests
  avx512.rs   AVX-512F implementation (x86_64), incl. vectorized exp + its tests
  neon.rs     NEON implementation (aarch64), incl. vectorized exp + its tests
  simd128.rs  WASM SIMD128 implementation (wasm32, opt-in), incl. vectorized exp + its tests
  common.rs   shared FlashAttentionConfig + shape-assert/multihead-dispatch helpers
  v1.rs       FlashAttention-1-style: per-step normalize, no causal skip
  v2.rs       FlashAttention-2-style: deferred normalize, causal tile-skip
  v3.rs       v2 + same-core software-pipelined score/softmax overlap
  naive.rs    O(n²) reference implementation (testing/benchmark baseline)
tests/correctness.rs   v1/v2/v3 vs. naive across shapes, causal + multihead,
                       plus a v1/v2/v3 mutual-agreement check (native targets)
tests/wasm_simd.rs     wasm-bindgen-test suite, run via `wasm-pack test --node`
benches/bench.rs       Criterion benchmarks (naive, v1, v2/flash, v3) — not wasm32
examples/basic.rs       usage demo
examples/bench_quick.rs manual-timing sanity check (no Criterion wait)
fuzz/fuzz_targets/flash_attention.rs   cargo-fuzz differential fuzzing, v1/v2/v3 vs. each other
.cargo/config.toml      enables wasm32 SIMD128 for this repo's own builds/tests
```

## Building/testing for each target

```bash
cargo test                                             # native, whatever SIMD tier the host has
cargo test --target x86_64-apple-darwin                # AVX-512/AVX2 cross-target type-check
                                                        # (won't execute without real x86_64 hardware/Rosetta)
wasm-pack test --node                                  # WASM SIMD128, real execution via Node.js
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for more (adding a new SIMD kernel,
fuzzing, MSRV policy).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Unless you explicitly state
otherwise, any contribution intentionally submitted for inclusion in this
crate, as defined in the Apache-2.0 license, shall be dual-licensed as
above, without any additional terms or conditions.
