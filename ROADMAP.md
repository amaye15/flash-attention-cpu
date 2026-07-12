# Roadmap & research notes

This expands on README's [Extension points](README.md#extension-points) list
with the "why" behind each item's priority: what's actually blocking it,
what external evidence (papers, other projects' production experience)
says about whether it's worth doing, and roughly how big a piece of work it
is relative to what's already landed. Re-checked periodically against
upstream Rust/WASM/Arm status, since several of these are gated on
toolchain support that moves independently of this crate.

Same rule as everywhere else in this repo: a line item only moves from
"considered" to "implemented" after an isolated microbenchmark and real
before/after numbers, not on the strength of a paper's claims alone.

## Tier 1 — near-term, stable-Rust-accessible today

### 1. WASM `relaxed-simd` real FMA (doc correction + open opportunity)

**Status: implemented.**

`src/simd128.rs`'s module docs and README both used to say relaxed-simd's
FMA "isn't part of stable `simd128`" — that was stale. The proposal reached
Phase 4 (full standardization) in 2024, and Rust stabilized
`core::arch::wasm32` relaxed-simd intrinsics (incl. `f32x4_relaxed_madd`) in
**1.82** — this crate's MSRV is already 1.89, so no toolchain bump was
needed. Chrome, Firefox, and Node/V8 (what `wasm-pack test --node` runs
against) support it; Safari still has it behind a flag as of early 2026,
same shape of portability tradeoff the crate already accepts for baseline
`simd128` (opt-in, not universal) — so it's a further, separate opt-in
(`-C target-feature=+relaxed-simd`) layered on top, not folded into
`.cargo/config.toml`'s default.

Landed as a single `fma128_ps` helper (`a*b+c`, one call site) used
everywhere `simd128.rs` previously did a separate multiply-then-add: `dot`,
`dot4`, `dot4x4`, `axpy`, `pv4`, and `exp128_ps`'s Horner-form polynomial —
selecting `f32x4_relaxed_madd` under `#[cfg(target_feature = "relaxed-simd")]`
and the original `f32x4_add(c, f32x4_mul(a, b))` otherwise, so the
fused/unfused choice is made in exactly one place. Validated with real
execution both ways (`wasm-pack test --node`, with and without
`RUSTFLAGS="-C target-feature=+relaxed-simd"`) — CI's `wasm-relaxed-simd`
job runs the flagged build permanently, guarded by a
`relaxed_simd_target_feature_is_actually_enabled` test so that job can't
silently degrade to testing the un-fused path.

Side finding while validating this: `wasm-pack test --node` reported
"no tests to run!" for `simd128.rs`'s own inline `#[cfg(test)] mod tests` —
`wasm-bindgen-test`'s harness only discovers `#[wasm_bindgen_test]`-marked
functions, never plain `#[test]`, on `wasm32-unknown-unknown` (no OS
process/argv support for the default libtest harness there). All ten of
that module's unit tests (including the new `fma_matches_mul_add`) were
silently dead code prior to this change; converted to
`#[wasm_bindgen_test]` so they actually execute now.

**No throughput number is published here.** This repo has no wasm32 timing
harness — `std::time::Instant` panics without a JS shim on bare
`wasm32-unknown-unknown`, and Criterion/`bench_quick.rs` are both
native-only (see `Cargo.toml`'s dev-dependency comment). The guaranteed win
is numerical (single rounding instead of two, per this crate's own
"real numbers, not paper claims" standard — not fabricating a speed number
that wasn't measured); actual throughput is JIT/engine-dependent and would
need a `js_sys::Date::now()`-based harness to measure honestly, which
didn't seem proportionate to build for one fused-multiply-add change.

- [Stabilize Wasm relaxed SIMD (rust-lang/rust#117468, merged)](https://github.com/rust-lang/rust/pull/117468)
- [WebAssembly relaxed-simd proposal (Phase 4)](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)

### 2. bf16 dot-product acceleration (AVX512_BF16 `vdpbf16ps`, Arm `bfdot`)

**Status: partially re-verified, not implemented.** Actually tried
compiling both hardware paths against stable Rust before committing to an
implementation plan (same discipline as everywhere else in this list —
check the toolchain claim directly rather than trust it secondhand), and
found a real asymmetry this item didn't originally account for:

- **AVX512_BF16 (x86_64): confirmed available on stable Rust.**
  `core::arch::x86_64`'s `__m256bh`/`__m512bh` types and
  `_mm256_dpbf16_ps`/`_mm512_dpbf16_ps`/`_mm256_cvtneps_pbh` all compile
  cleanly under `#[target_feature(enable = "avx512bf16,avx512f,...")]` on
  rustc 1.93. Same caveat this crate already documents for AVX-512F
  itself, though: compiles, but unverified by real execution in this
  sandbox (no working Rosetta 2 / no AVX512_BF16-capable x86_64 hardware
  here) — same "type-checks and passes clippy, hardware validation still
  pending" status the existing AVX-512F path carries.
- **Arm `bfdot` (NEON): not available on stable Rust.** Tried the direct
  equivalent — `bfloat16x8_t`, `vbfdotq_f32`, `vcvtq_low_bf16_f32` — on
  this sandbox's own aarch64 (Apple Silicon) host, natively, not
  cross-compiled. All three fail to resolve on stable rustc 1.93
  (`cannot find type/function in this scope`); stdarch's NEON bf16
  intrinsics are still gated behind an unstable feature. So the original
  framing above ("Arm's equivalent is available on ARMv8.6+ and Apple
  Silicon") was wrong about *Rust's* support specifically — the
  *hardware* extension exists on that silicon, but nothing in stable
  `core::arch::aarch64` exposes it yet.

This matters because it changes what "bf16 support" can mean right now:
a real `vdpbf16ps`-style two-sided bf16×bf16→f32 dot product is only
buildable (and only real-execution-testable in this project's own CI,
which has no x86_64 AVX512_BF16 runner) on x86_64, with no equivalent
native-instruction path on Arm today. A *portable* bf16 story (storage as
bf16, widened to f32 with a plain bit-shift — no dot-product instruction
needed at all, since `bf16` is exactly the truncated top 16 bits of `f32`
— then fed through the existing per-arch f32 kernels) is possible on every
target including Arm, but only captures the caller-side storage-footprint
win, not the full during-tile-sweep re-read bandwidth reduction that
motivated this item, unless the widen happens inside each kernel's inner
loop (i.e., still means writing a new kernel, just one that doesn't need a
dedicated bf16 ISA extension to do it).

**Recommendation:** revisit as a scoped decision, not a default "yes" —
options are (a) x86_64-only `vdpbf16ps` path, real hardware validation
blocked until this project has AVX512_BF16-capable CI/test hardware,
(b) a portable widen-in-kernel path (scalar + NEON accelerated, both
real-execution-testable in this sandbox today) that doesn't depend on
either dot-product ISA extension, or (c) both, sequenced with (b) first
since it's testable now and (a) can follow once hardware access exists.
Int8/VNNI (item 4) should still come after whichever of these is chosen,
per the original sequencing rationale below.

External signal this is still worth pursuing eventually, just not
speculative busywork: oneDNN's reference scaled-dot-product-attention
primitive already treats f32/bf16/f16 as first-tier supported dtypes
(int8 is a separate, later tier — see item 4 below), and a dedicated
`vllm-cpu-avx512bf16` package shipped April 2026 pairing AVX512-BF16+VNNI
specifically with FlashAttention/FlashInfer integration. bf16 halves
memory bandwidth, which FlashAttention-3's own numerics story identifies
as the actual bottleneck at long sequence lengths — same motivation,
CPU-side.

**Effort:** real, not a drop-in — needs the storage/kernel-scope decision
above, a new kernel per arch actually pursued, and accuracy validation vs.
the existing `naive.rs` f32 reference.

- [AVX-512 BF16 instructions (VDPBF16PS)](https://en.wikichip.org/wiki/x86/avx512_bf16)
- [oneDNN Scaled Dot-Product Attention (SDPA) — dtype support](https://www.intel.com/content/www/us/en/docs/onednn/developer-guide-reference/2025-2/scaled-dot-product-attention-sdpa.html)
- [vllm-cpu-avx512bf16 (PyPI, Apr 2026)](https://pypi.org/project/vllm-cpu-avx512bf16/)

### 3. FLASH-D-style hidden-softmax-division reformulation

**Status: investigated, not adopted.**

This looked, from the abstract alone, like the one item needing no new
ISA/hardware at all. Working through FLASH-D's actual math (ISLPED 2025 /
arXiv:2505.14201) against this crate's tile-blocked design changed that
conclusion — writing the derivation down here so it isn't re-litigated
without new information later.

**What FLASH-D actually does.** Per key `i`, instead of tracking a running
max `M_i` and sum `L_i` (FA2's `m`/`l`), it tracks one scalar
`z_i = M_i + ln(L_i)` — exactly `logsumexp(s_1..s_i)` — and a **normalized**
running output `O_i`. The update is `w_i = sigmoid(s_i - z_{i-1})`,
`O_i = O_{i-1} + (v_i - O_{i-1}) · w_i`, with `z_i = z_{i-1} + softplus(s_i - z_{i-1})`.
No explicit max/compare unit, no separate divider — a sigmoid (already
common ASIC-side for gating/activations) does both jobs. This is a genuine,
exact reformulation, and it's a good trade *for streaming, one-key-per-cycle
hardware*: normalizing every step is free there (it's just another pipeline
stage), so trading it for eliminated max-tracking hardware is a clean win —
hence the paper's 20–28%/16–27% area/power numbers.

**Why it doesn't transfer to a tile-blocked CPU kernel.** Generalizing the
same identity from one key to one KV-tile (`s_i` → tile-local logsumexp
`z_local = m_local + ln(l_local)`, `v_i` → tile-local normalized output
`V_local_avg = v_local / l_local`) gives the same `O_i = O_{i-1} +
(V_local_avg - O_{i-1}) · w_i` form — but `V_local_avg` requires dividing
the tile's whole `d_head`-wide accumulator by `l_local`, **every tile**, to
maintain the always-normalized invariant that's the whole reason the max
tracking becomes unnecessary. Compare that to what `v2`/`v3` already do:
track `(m, l, o)` unnormalized and divide **once**, at the very end of each
query row's whole KV sweep (`o_N / l_N`) — this is precisely FA2's
"deferred normalization" that `v2`'s docs already describe. In other words,
v2 already made the trade FLASH-D is offering, and made it in the cheaper
direction: FLASH-D swaps *frequent* division for *no max-tracking*; v2
already has *no frequent division* while keeping max-tracking (which is
just a cheap SIMD compare, not a divide). Adopting FLASH-D's tile-level
form here would mean trading v2's 1 division-pass per row for
`num_tiles` division-passes per row — a strict regression in exactly the
operation (division) both techniques are trying to minimize.

Two more reasons it doesn't help even setting that aside:
- The **within-tile** local max (`max_reduce4`, needed just to safely
  exponentiate raw `QK^T` scores before any softmax normalization at all)
  isn't the running max FLASH-D removes — that one's still needed
  regardless, so `max_reduce4`/`sub_exp_sum_inplace4` wouldn't shrink.
- FLASH-D's recurrence is a strict sequential dependency chain (`w_i`
  depends on `z_{i-1}` depends on `w_{i-1}`...). This crate's speed on
  exactly this bookkeeping comes from the opposite property — `max_reduce4`/
  `sub_exp_sum_inplace4`/`pv4` are fast *because* rows/tiles are mutually
  independent and SIMD-parallelizable (see Design above). A literal
  per-key application would also serialize what's currently a wide,
  vectorized reduction over the whole `Bc` tile.

**Conclusion:** not implemented. This is an ASIC area/power optimization
whose mechanism (avoid a max-tracking unit by paying a divide every step)
is specifically valuable when "every step" is one hardware cycle and
divides are otherwise-idle pipeline capacity — neither holds for a
tile-batched software kernel that already defers to one division per row.
No microbenchmark was run because the operation-count argument above is
structural, not close enough to call empirically (this crate still ran the
numbers-first policy — the numbers here are already decisive without
needing to write the code to know it'd be slower).

- [FLASH-D: FlashAttention with Hidden Softmax Division (arXiv:2505.14201)](https://arxiv.org/abs/2505.14201)

## Tier 2 — real, but bigger investment or currently hardware/toolchain-gated

### 4. int8 quantized QK^T/PV (VNNI path)

README already flagged `avx512vnni` as only mattering "alongside the
lower-precision work" — that's now backed by real accuracy data instead of
a guess. INT-FlashAttention (arXiv:2409.16997) reports token-level
post-training quantization getting attention activations to full INT8 with
1.69–9% relative error depending on input distribution (vs. ~9% for FP8
FlashAttention under the same uniform-distribution test), and ~72% faster
than FP16 FlashAttention on hardware without native FP8. The useful
takeaway isn't the exact numbers (GPU paper, different hardware) — it's
the *calibration scheme* (token-level, not naive per-tensor quantization)
as a starting point if this is ever pursued.

**Sequencing:** after bf16 (item 2), not before — bf16 needs no
calibration/accuracy-validation infrastructure and int8 does; building that
infrastructure once and reusing it for int8 is cheaper than building it
twice.

- [INT-FlashAttention: Enabling Flash Attention for INT8 Quantization (arXiv:2409.16997)](https://arxiv.org/pdf/2409.16997)

### 5. Arm SVE (new item — not previously in README)

Targets server Arm (Graviton3/4, Neoverse V-series) where NEON's fixed
128-bit width leaves throughput on the table vs. SVE's scalable vectors.
**Not actionable on stable Rust today**: intrinsics are nightly-only, and
stabilization is blocked upstream on Rust's "Sized Hierarchy" language
prerequisite. A stdarch PR with initial SVE types/intrinsics is open but
unmerged as of mid-2026.

**Recommendation:** track, don't implement — same posture as RVV (item 8),
for the same reason (nightly-only intrinsics, no committed stabilization
date).

- [SVE and SME on AArch64 — Rust Project Goals](https://rust-lang.github.io/rust-project-goals/2025h1/arm-sve-sme.html)
- [Tracking issue: Sized Hierarchy and Scalable Vectors (rust-lang/rust-project-goals#270)](https://github.com/rust-lang/rust-project-goals/issues/270)
- [initial SVE intrinsics (rust-lang/stdarch#2071, open)](https://github.com/rust-lang/stdarch/pull/2071)

### 6. Arm SME2 / KleidiAI-style matrix acceleration (new item)

Apple M4+ and the newest Android flagships; Arm reports up to 6x LLM
inference speedup when XNNPACK routes matrix-heavy ops through KleidiAI's
SME2 kernels, no application changes needed *inside XNNPACK*. But that's
precisely the catch for this crate: SME is a streaming-mode matrix-tile
ISA extension (special calling convention, tile registers configured via
dedicated instructions), not "a wider vector register" like every SIMD
tier this crate currently supports. It's a different category of
engineering than the shared `Kernel` trait (dot/axpy/scale/reduce) this
codebase is built around — closer to Intel AMX (item 7) in shape than to
adding another NEON-like backend. No stable Rust intrinsics path exists.

**Recommendation:** track as a long-horizon, high-ceiling item; not a
near-term "add a SIMD tier" task like AVX-512/NEON/SIMD128 were.

- [Boost AI inference 6x on Arm CPUs with SME2 and KleidiAI](https://www.arm.com/technologies/sme2/accelerate-on-device-ai)
- [ARM-software/kleidiai](https://github.com/ARM-software/kleidiai)

### 7. Intel AMX

Sapphire Rapids/Granite Rapids Xeon AMX gives large inference gains for
matrix-heavy workloads. Interestingly, the Apple-side equivalent research
reaches a structural conclusion worth carrying over: "Above the Inner
Loop" (arXiv:2606.25426) shows a hand-written Apple AMX kernel beating
Accelerate by ~1.17–1.23x not via a faster inner loop, but via better
thread-panel scheduling and weight pre-packing — i.e. AMX-class wins come
from tile/thread scheduling around the coprocessor, not micro-op-level
tuning. That's a materially different kind of engineering effort than
this crate's per-kernel SIMD trait architecture.

**Recommendation:** same posture as SME2 — track, don't build, until
there's a stable, portable Rust story for programming these
coprocessors (currently unsafe inline-asm territory, OS-level enablement
on Linux, no `std::arch` intrinsics).

- [Exploiting Intel AMX for LLM Inference (IEEE)](https://ieeexplore.ieee.org/document/10538369)
- [Above the Inner Loop: Exceeding Accelerate at LLM Prefill GEMM on the M1 AMX (arXiv:2606.25426)](https://arxiv.org/pdf/2606.25426)

## Tier 3 — re-checked, still correct as-is

### 8. RISC-V Vector (RVV)

Re-verified, no change to README's existing assessment: intrinsics remain
nightly-only (`riscv_ext_intrinsics`), tracked in
[rust-lang/rust#114544](https://github.com/rust-lang/rust/issues/114544),
with no committed stabilization date. RVV's `vsetvli`-style variable
vector length still doesn't map onto this crate's fixed-width `Kernel`
trait as directly as SSE/NEON/wasm did — a real design question, not just
a new file, so left until intrinsics stabilize.

## External validation (not action items — context for future calls)

- **llama.cpp's own CPU flash-attention experience**: reported as helping
  prompt processing/prefill more than decode/token-generation on CPU — the
  same Amdahl's-law shape this crate already documents for the causal-mask
  optimization (gains concentrated in specific shapes, in the noise
  elsewhere). Corroborates treating "where does this actually help" as a
  per-optimization question rather than assuming a uniform win.
  [(DeepWiki: CPU Backend and Optimization)](https://deepwiki.com/ggml-org/llama.cpp/4.2-cpu-backend-and-optimization)
- **`matrixmultiply`/`gemm` crates** (bluss, sarah quiñones) independently
  use the same BLIS-style two-sided packed macro/microkernel approach this
  crate arrived at for `dot4x4`/`pv4` — external validation that the
  register-blocking direction was sound engineering, not a local optimum
  peculiar to this project.
  [(matrixmultiply)](https://github.com/bluss/matrixmultiply) ·
  [(gemm)](https://crates.io/crates/gemm)
- **Positioning vs. the Rust ML ecosystem**: candle/burn are also investing
  in dedicated SIMD CPU backends (burn-flex, macerator-based elementwise
  ops), but as one backend inside a general tensor framework. This crate
  remains the only from-scratch, attention-specific, tiled-flash-attention
  CPU kernel in pure Rust found during this research pass — worth keeping
  as an explicit differentiator rather than reinventing a general GEMM
  library.

## Not re-litigated this pass

`bf16`/`f16`/FP8 as a category, thread-level producer/consumer pipelining,
and the backward pass remain deferred for the reasons already given in
README's Extension points section — this pass didn't surface new evidence
that changes those calls, beyond sequencing bf16 ahead of int8 (item 2 vs.
4 above).
