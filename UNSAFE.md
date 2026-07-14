# Minimizing unsafe code — research & plan

This crate is fundamentally a hand-vectorized SIMD library — `unsafe` isn't
incidental here, it's central to what the crate does (`std::arch` intrinsics
are `unsafe fn` by definition; there's no way to call `_mm256_fmadd_ps` or
`vaddvq_f32` without an `unsafe` block). The question this document answers
isn't "how do we eliminate unsafe" — that's not fully possible without
giving up hand-written SIMD entirely — it's "how do we shrink it to its
true minimum, and make what remains as auditable and as provably-sound as
possible, without touching codegen for the hot path." Every claim below was
checked directly against this codebase or by compiling/running real code,
not assumed from documentation.

## Starting state (measured, not estimated, before any of this document's plan was applied)

```
unsafe fn declarations:  133  (avx2.rs 23, avx512.rs 21, sse41.rs 23, neon.rs 21, simd128.rs 24, scalar.rs 10, kernel.rs 11 trait decls)
unsafe {} blocks:       ~140  (SIMD kernel files ~105, v1.rs/v2.rs/v3.rs call sites ~35)
// SAFETY comments:        0
clippy safety lints:      none enabled
unsafe_op_in_unsafe_fn:   not enabled (bodies rely on the pre-2024 "whole fn body is an unsafe block" default)
```

(See "What actually happened," below the tier writeups, for the current
state after Tiers 0 and 1 were implemented — `unsafe fn` count is now 62,
not 133. This section is kept as the historical before-picture the rest of
the document's reasoning was built from.)

Two findings from actually reading the code (not just counting it) change
the shape of the plan:

- **`scalar.rs`'s entire kernel — all 10 `unsafe fn` — do nothing unsafe
  internally.** Every method body is plain iterator/slice code
  (`iter().zip().map().sum()`, `chunks_exact`, `fold`). The `unsafe` is
  pure ceremony, forced only by [`Kernel`](src/kernel.rs)'s trait
  signature requiring every implementor to match one shape. This is the
  cheapest possible win in the whole codebase: fixing the trait design
  (Tier 1 below) makes these genuinely safe, not just labeled otherwise.
- **`v1.rs`/`v2.rs`/`v3.rs`'s ~35 `unsafe` blocks are all of the shape
  `unsafe { K::dot(...) }`** — thin call-site wrappers around trait
  dispatch, not intrinsic calls themselves. The actual "is this safe to
  call" question (has `is_x86_feature_detected!` been checked?) was
  already answered once, earlier in the same function, by the dispatch
  `if` chain. These blocks exist only because `Kernel`'s methods are
  typed `unsafe fn` — they carry no additional risk beyond "trust the
  dispatch above," so they're also a strong candidate for elimination via
  the same trait redesign.

That leaves the **real, irreducible core**: ~85 `unsafe fn` and ~105
`unsafe` blocks inside `avx2.rs`/`avx512.rs`/`sse41.rs`/`neon.rs`/
`simd128.rs` themselves, where raw intrinsics actually get called. No
design pattern makes calling `_mm256_loadu_ps` safe to spell without
`unsafe` in the source — the goal for this tier is minimizing surface
area and maximizing auditability, not elimination.

## Tier 0 — do now: zero risk, zero performance cost

These don't reduce the *count* of unsafe operations, but they're the most
direct, lowest-risk answer to "improve safety as much as possible" when
elimination isn't fully possible — and they cost nothing at runtime.

1. **Enable `clippy::missing_safety_doc` and `clippy::undocumented_unsafe_blocks`.**
   The first requires every `unsafe fn` to carry a `# Safety` doc section;
   the second requires every `unsafe {}` block to carry a `// SAFETY:`
   comment justifying it. Neither exists anywhere in this codebase today
   (verified: `grep -rc "// SAFETY" src/*.rs` → 0 across all 13 files).
   This is *the* standard Rust community convention for exactly this
   situation — [`// SAFETY:`](https://github.com/rust-lang/rust-clippy/blob/master/clippy_lints/src/undocumented_unsafe_blocks.rs)
   outnumbers the alternate casing "// Safety:" 1072-to-67 in rust-lang/rust
   itself, and `missing_safety_doc` checks for that exact markdown section.
2. **Enable `#![warn(unsafe_op_in_unsafe_fn)]`.** Verified directly
   (compiled a minimal repro) that this works on stable Rust, edition
   2021, with *no* edition bump required. It's warn-by-default starting
   in edition 2024, but it's been available as an explicit opt-in lint
   since long before that. Right now every `unsafe fn` in this crate
   relies on the old "the whole function body is an implicit unsafe
   block" behavior — e.g. `dot_avx2`'s body freely calls `_mm256_fmadd_ps`
   with no inner `unsafe {}}` marking exactly which lines need it. Turning
   this on forces every actual unsafe operation to be individually
   bracketed, which is a real audit improvement (you can `grep` for
   exactly the N lines that need scrutiny instead of "the whole
   function") for zero behavior change.
3. **Actually write the safety justifications this surfaces.** For this
   codebase these fall into a handful of repeated, well-understood
   patterns:
   - *SIMD kernel functions* (`avx2.rs` etc.): "caller must ensure the
     target CPU supports `avx2`+`fma` (checked via
     `is_x86_feature_detected!` in `v1.rs`/`v2.rs`/`v3.rs`'s dispatch
     before this type is ever selected)."
   - *Pointer-offset loads inside a bounds-checked loop* (`a.as_ptr().add(i)`
     under a `while i + 4 <= len` guard): "the loop guard above guarantees
     `i + 4 <= len`, so this load reads `a[i..i+4]`, in bounds."
   - *`Kernel` trait methods generally*: link back to the one canonical
     explanation in `kernel.rs` rather than repeating it 133 times.
4. **Fix `scalar.rs`'s 10 ceremonial `unsafe fn`** — see Tier 1's trait
   redesign; this is the concrete mechanism, not a separate step.
5. **Replace manual `.as_ptr().add(i)`/`.as_mut_ptr().add(i)` arithmetic
   with `chunks_exact()`/`chunks_exact_mut()`** wherever the actual SIMD
   load/store still needs a raw pointer. `chunks_exact` is fully stable
   today and produces already-length-validated sub-slices with zero
   runtime cost (LLVM optimizes it identically to hand-written pointer
   arithmetic) — it doesn't remove the `unsafe` block around the load
   intrinsic itself, but it removes the hand-checked "is `i+4` actually
   `<= len`" bookkeeping from the *unsafe* portion of the reasoning,
   shrinking exactly what a reviewer needs to verify by hand. (`slice::as_chunks`,
   which would go one step further and hand back a checked `[f32; 4]`
   array directly, is **still nightly-only** as of 2026 — the
   stabilization PR (`slice_as_chunks`) is in progress but not landed;
   confirmed via the tracking issue, not assumed.)

## Tier 1 — the "CPU feature token" pattern (no new dependency)

This is the modern, idiomatic answer to "wrap unsafe in safe" for exactly
this runtime-dispatch-over-SIMD-tiers situation, and it doesn't require
adopting any external crate — it's a refactor of this crate's own `Kernel`
trait and dispatch functions.

**The pattern** (independently confirmed in two places: a June 2026 piece
on making SIMD "safe on the inside," and `pulp`'s own `Simd`/`Arch` design,
see Tier 2): define a zero-sized type per SIMD tier (`Avx2Token`,
`Sse41Token`, ...) that can *only* be constructed by a safe function which
performs the `is_x86_feature_detected!` check once. Downstream operations
become **safe** functions parameterized by (or generic over) that token —
possessing the token *is* the proof the intrinsics are sound to call, so
the actual `unsafe { _mm256_fmadd_ps(...) }` call moves inside the token
type's own method body, written and audited exactly once, instead of the
caller needing an `unsafe` block at every use site.

**Applied to this crate**: `Kernel`'s methods would take a token
parameter instead of being `unsafe fn`. Concretely:
- `v1.rs`/`v2.rs`/`v3.rs`'s ~35 `unsafe { K::method(...) }` call sites
  become plain safe calls — the dispatch `if is_x86_feature_detected!(...)`
  chain that already exists is exactly where the token gets constructed,
  once, so nothing about *when* the check happens changes, only that the
  compiler now enforces it can't be skipped.
- `ScalarKernel`'s 10 methods, which do nothing unsafe internally, would
  become genuinely safe functions matching a safe trait — no more
  ceremonial `unsafe`.
- The ~85 `unsafe fn`/~105 blocks *inside* `avx2.rs`/`avx512.rs`/`sse41.rs`/
  `neon.rs`/`simd128.rs` are unaffected — the intrinsic calls themselves
  still need `unsafe`, now wrapped one layer deeper inside each token
  type's methods instead of at the `Kernel` trait boundary.

**Net effect**: eliminates unsafe at the *orchestration* layer (~45 of the
~140 blocks, plus turns 10 fake-unsafe `scalar.rs` functions genuinely
safe) with a self-contained refactor, no new dependency, no risk to
codegen (the token is a ZST, monomorphization erases it same as today).
Real effort — touches `kernel.rs` and all three variant files — but
bounded and well-understood.

## Tier 2 — `pulp` for the arithmetic primitives: real, but not a full 4-for-4

**Status: prototyped and benchmarked for real; partial finding, not a full
port.** The original plan called this "the biggest potential win, pending
validation." Two rounds of hands-on prototyping (not just reading docs)
turned up one genuine green light and one hard blocker deeper than
initially scoped — worth recording precisely so this isn't re-litigated
from a rosier starting point later.

**[`pulp`](https://github.com/sarah-quinones/pulp)** (sarah quiñones) —
same author/lineage as `gemm`/`faer`, already cited in
[ROADMAP.md](ROADMAP.md) as validation for this crate's own
register-blocked microkernel design. Confirmed directly: call-site API is
100% safe (`splat_f32s`/`mul_add_f32s`/`reduce_sum_f32s`/etc., zero
`unsafe` needed by callers), covers x86_64 (V3=AVX2, V4=AVX-512 — the
latter now stable-Rust-compatible as of v0.22.3, unlike an earlier
version that needed nightly), aarch64, and wasm32 — 4 of this crate's 5
backends (no sub-AVX2 x86 tier, so `sse41.rs` would stay hand-written
regardless).

### Green light: register-blocked arithmetic (dot, dot4x4, pv4, reductions)

Wrote a real 2-way-unrolled dot product against pulp's `Simd` trait and
benchmarked it on this sandbox's actual aarch64 host against this crate's
own `dot_neon` (copy-pasted verbatim as the baseline), across
`d_head` = 64/128/256/1024, 2M iterations each:

```
d=   64  hand-NEON: 8.27 ns   pulp-2way: 6.98 ns (0.845x)   pulp-2way-prehoisted: 4.57 ns (0.553x)
d=  128  hand-NEON: 7.47 ns   pulp-2way: 7.99 ns (1.070x)   pulp-2way-prehoisted: 5.83 ns (0.781x)
d=  256  hand-NEON: 12.31 ns  pulp-2way: 11.61 ns (0.943x)  pulp-2way-prehoisted: 11.43 ns (0.929x)
d= 1024  hand-NEON: 68.80 ns  pulp-2way: 66.66 ns (0.969x)  pulp-2way-prehoisted: 66.62 ns (0.968x)
```

Matching the hand-written kernel's *register-blocking shape* (two
independent accumulator chains, not one) was essential — a first, naive
single-accumulator pulp version measured 1.2–1.9x *slower*, and widening
with `d_head`. Once shaped correctly, pulp lands within noise of (and at
small sizes, faster than) the hand-tuned NEON code, both with `Arch::new()`
called fresh each time and pre-hoisted outside the hot loop (this crate's
actual usage pattern, since kernel selection already happens once per
`flash_attention_vN` call). **Conclusion: the register-blocked arithmetic
primitives — `dot`/`dot4`/`dot4x4`/`pv4`/`axpy`/`scale_inplace`/
`max_reduce`/`max_reduce4` — port to pulp with no performance regression,
verified, not assumed.**

### Hard blocker: `exp()`'s bit-manipulation technique has no pulp path

Every one of this crate's five `exp` implementations (`avx2.rs`,
`avx512.rs`, `sse41.rs`, `neon.rs`, `simd128.rs`) uses the same standard
SIMD-math-library technique: range-reduce `x = n*ln2 + r`, evaluate a
degree-5 polynomial for `exp(r)`, then reconstruct `2^n` by treating a
small integer `n` as raw IEEE-754 exponent bits — `bits = (n_as_i32 + 127)
<< 23`, bitcast back to `f32`. This needs, in order: a float→int
conversion, integer addition, an integer shift, and an int→float bitcast.

- **The floor/round step is solved.** The classic "magic number" trick
  (`(x + 1.5·2²³) − 1.5·2²³`, exploiting f32 rounding instead of a
  dedicated round instruction) was checked numerically against this
  crate's actual `floor(x·log2(e) + 0.5)` step over 1.78M swept points in
  the domain this crate clamps to (±88.376): only 4 disagreements (off by
  1 in the integer, at rounding boundaries), and — critically — **zero
  difference in the final `exp(x)` accuracy**: both methods measured
  `1.191016e-7` max relative error against `f32::exp` across the same
  sweep this crate's own `exp_matches_std` tests use. This part of the
  plan holds up.
- **The integer arithmetic does not exist in pulp at all.** Exhaustively
  grepped v0.22.3's `Simd` trait for every `i32s`/`u32s`-related method:
  bitcast/transmute, rotate, a *dynamic* shift (`wrapping_dyn_shl_u32s`,
  present but falls back to a scalar loop internally in pulp's own default
  impl), and `select` — but **no integer add, no float→int conversion, no
  `impl Add for` any integer SIMD type**. Without integer addition there
  is no way to compute `n + 127` on the vector, and without a float→int
  conversion there is no way to get `n` as an integer at all — both
  irreplaceable for this technique, and pulp exposes neither today.

This is a materially different (and worse) finding than the roadmap's
original framing, which only flagged the *floor* step as missing.
Working around it means either (a) inventing a different `exp()`
algorithm that stays entirely in float arithmetic (a real, separate
research effort — needs its own accuracy characterization against
`f32::exp`, not a rewrite that can piggyback on this pass), or (b) keeping
hand-written, arch-specific `unsafe` code for `exp()` alone even inside an
otherwise-pulp-based kernel. Since `exp()` (via `sub_exp_sum_inplace`/
`sub_exp_sum_inplace4`, both part of the same `Kernel` trait as the
arithmetic primitives) is where a large share of each kernel file's actual
complexity already lives, option (b) means a "pulp port" would still carry
real hand-written unsafe in the one function most worth removing it from
— a meaningfully smaller win than "port 4 backends to pulp" implied.

**Recommendation:** don't complete a full port on this pass. The
arithmetic-primitive finding is real and worth acting on eventually, but
bundling it with a half-solved `exp()` would either overstate what's been
achieved or require a separate, unvalidated algorithm change to actually
finish the job. Revisit if: pulp adds integer arithmetic to its `Simd`
trait, or a validated float-only `exp()` approximation surfaces (worth a
dedicated research pass of its own, not a rider on this one).

### Alternatives considered and why they're not the first choice

- **[`std::simd`/portable_simd](https://doc.rust-lang.org/std/simd/index.html)**:
  the "obvious" answer, and the most complete safe abstraction if it were
  usable — but confirmed still nightly-only as of a February 2026 survey,
  with named-and-unresolved stabilization blockers (mask type design,
  swizzle ergonomics). Adopting it would mean requiring nightly Rust,
  directly contradicting this crate's stated "no nightly" design
  principle. Not viable today; revisit if/when it stabilizes.
- **[`wide`](https://github.com/Lokathor/wide)** (Lokathor): portable SIMD
  on stable Rust, wraps `safe_arch` internally, actively maintained
  (rust-version tracked deliberately). A legitimate alternative to `pulp`
  — not chosen as the primary recommendation mainly because `pulp`'s
  lineage (same author as `gemm`/`faer`) is a more direct match for this
  crate's specific register-blocked-microkernel style, already
  independently cited in this project's own research. Worth a second look
  if `pulp`'s floor-primitive gap turns out to be a hard blocker.
- **[`safe_arch`](https://github.com/Lokathor/safe_arch)** (Lokathor):
  lower-level than `wide` — safe intrinsics gated by compile-time
  `cfg`/target-feature, explicitly "not intended for everyday users," a
  foundation layer other crates build on rather than a top-level solution.
  Useful as a conceptual validation of "safe wrapper via cfg," not a
  direct fit for this crate's runtime-dispatch design.
- **`fearless_simd`**: introduced in the same June 2026 article that
  described the CPU-feature-token pattern; checked its current published
  version (0.6.0) directly. Promising design, but far less proven than
  `pulp` (v0.22.3, powers a real benchmarked library) — worth re-checking
  in a future pass, not a bet to make yet.

## Verification tooling (orthogonal to which tier gets adopted)

Whatever unsafe remains — even just Tier 0's irreducible core — should be
checked by tools that catch UB directly, not just reviewed by eye.

**Miri**: tested directly against this crate's actual code, not assumed
from documentation. Installed `nightly` + the `miri` component and ran
`cargo +nightly miri test` for real:
- **All three x86_64 kernels — AVX2, AVX-512F, SSE4.1 — pass completely**
  under Miri via cross-target interpretation (`--target
  x86_64-unknown-linux-gnu`, run from this sandbox's aarch64 host): 27/27
  tests across all three kernels, including the vectorized `exp`
  (floor-based range reduction, bit-manipulation exponent reconstruction)
  and horizontal max/sum reductions. Zero unsupported-operation errors.
- **NEON does not work today**: running natively on this aarch64 sandbox,
  `neon::tests::dot_matches_scalar` fails with `unsupported operation:
  can't call foreign function llvm.aarch64.neon.faddv.f32.v4f32` — Miri
  doesn't yet implement that specific horizontal-add intrinsic. This
  matches a known, open gap (Miri's aarch64 NEON intrinsic coverage is
  behind its x86 coverage), not a bug in this crate.
- WASM32 wasn't checked — Miri doesn't interpret wasm32 as a target the
  way it cross-interprets x86_64.

**Recommendation**: add a Miri CI job scoped to `--target
x86_64-unknown-linux-gnu` only (covers `avx2.rs`/`avx512.rs`/`sse41.rs`
completely). Real, immediately-available verification value for 3 of 5
kernels today — don't wait for full multi-arch Miri coverage to get
something out of this.

## What actually happened (this pass)

1. **Tier 0 — done.** `clippy::missing_safety_doc`/`undocumented_unsafe_blocks`
   and `unsafe_op_in_unsafe_fn` enabled; every `unsafe fn`/`unsafe {}` block
   in the crate (160+ locations) now carries a real `// SAFETY:`
   justification, not a placeholder. Verified clean (`-D warnings`) on
   native, x86_64 cross-compile, and both wasm32 configurations.
2. **Tier 1 — done.** `Kernel` trait redesigned around the CPU-feature-token
   pattern: every method is now a safe `&self` fn; `Avx2Kernel`/
   `Avx512Kernel`/`Sse41Kernel::new() -> Option<Self>` do the
   `is_x86_feature_detected!` check exactly once; `NeonKernel`/
   `Simd128Kernel`/`ScalarKernel::new() -> Self` are infallible (mandatory
   baseline / compile-time gate / no precondition at all, respectively).
   Measured result: **`unsafe fn` count 133 → 62 (−53%)**; `v1.rs`/`v2.rs`/
   `v3.rs` went from 34 `unsafe` blocks to **0**; `scalar.rs` went from 10
   `unsafe fn`/5 blocks to **0** — the one fully-safe `Kernel` impl in the
   crate. The five SIMD kernel files' internal unsafe (the genuinely
   irreducible core) is unchanged in nature, just reached through safe
   methods instead of bare `unsafe fn`. Full test suite, clippy (`-D
   warnings`), `wasm-pack test --node` (both `simd128` and
   `simd128`+`relaxed-simd`), MSRV (1.89.0), and `cargo fmt` all verified
   clean after the refactor.
3. **Tier 2 — prototyped, not completed.** Real benchmarks (not
   assumptions) show pulp's register-blocked arithmetic primitives
   (`dot`/`dot4x4`/`pv4`/reductions) perform on par with this crate's
   hand-tuned NEON code once matched to the same 2-way-unrolled shape. But
   `exp()`'s IEEE-754 bit-manipulation technique — used by all five
   existing kernels — needs integer arithmetic pulp's `Simd` trait doesn't
   expose at all (confirmed by exhaustive search, not a missing grep hit).
   Completing a "pulp port" today would mean either overstating the result
   (still shipping hand-written unsafe for `exp()`) or inventing and
   separately validating a new float-only `exp()` algorithm. Neither is
   done on this pass — see the Tier 2 section above for what a future
   attempt would need.
