//! The [`Kernel`] trait isolates the handful of numeric primitives that
//! differ between the portable scalar fallback and the accelerated SIMD
//! paths (AVX-512F/AVX2+FMA/SSE4.1 on x86_64, NEON on aarch64). Everything
//! else — tiling, online-softmax running-stat bookkeeping, causal masking,
//! parallel dispatch — is written once, generically, per variant in
//! `v1.rs`/`v2.rs`/`v3.rs`, and monomorphized over whichever `Kernel` is
//! selected at runtime.

/// Numeric primitives used by the flash-attention inner loop.
///
/// Every method takes `&self` and is **safe** to call — the "CPU feature
/// token" pattern: [`crate::avx2::Avx2Kernel`], [`crate::avx512::Avx512Kernel`],
/// and [`crate::sse41::Sse41Kernel`] can only be *constructed* through their
/// respective (fallible) `new()` associated functions, which perform
/// `is_x86_feature_detected!("avx2")` / `"fma"` / `"avx512f"` / `"sse4.1"`
/// exactly once, up front. Simply *possessing* a value of one of these
/// types is therefore proof the check already passed, so every method can
/// be safe: the one `unsafe` intrinsic call each method needs internally is
/// justified by `Self`'s constructor having run, not by the caller
/// promising anything. [`crate::neon::NeonKernel`]/[`crate::simd128::Simd128Kernel`]/
/// [`crate::scalar::ScalarKernel`]'s `new()` are infallible (NEON is
/// mandatory AArch64 baseline; SIMD128 is a compile-time `#[cfg(...)]` gate;
/// scalar has no precondition at all) — see each type's own docs.
///
/// This eliminates `unsafe` at the call sites in `v1.rs`/`v2.rs`/`v3.rs`
/// entirely: those only ever hold a `&K` obtained from a `new()` that
/// already ran, so there's nothing left for the caller to promise.
pub(crate) trait Kernel {
    /// Dot product of two equal-length slices.
    fn dot(&self, a: &[f32], b: &[f32]) -> f32;

    /// `[dot(a0, b), dot(a1, b), dot(a2, b), dot(a3, b)]` — four dot
    /// products sharing one another's `b` vector loads. Register-blocking:
    /// a fixed `b` row's vector loads are amortized across four FMA
    /// accumulator chains instead of `b` being reloaded once per query row.
    /// Used for the `this_bc % 4` remainder columns [`Kernel::dot4x4`]
    /// leaves over.
    fn dot4(&self, a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4];

    /// `result[r][c] = dot(q[r], k[c])` for `r, c` in `0..4` — 4 query rows
    /// *and* 4 key rows blocked together ("packed" register tiling,
    /// BLIS/OpenBLAS-style), so both operands' vector loads are shared
    /// across the resulting 16 independent FMA accumulator chains. This is
    /// what [`Kernel::dot4`] alone doesn't capture: with only the query side
    /// blocked, the four query-row vectors still get reloaded from memory
    /// on every key row, since nothing keeps them resident across that
    /// loop. Measured ~1.5-1.6x throughput improvement over `dot4` alone on
    /// the QK^T step (see the crate's benchmarks).
    fn dot4x4(&self, q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4];

    /// Fused in-place `x[i] = exp(x[i] - m)` for all `i`, returning
    /// `sum(x)` *after* the exponential — one pass over `x` instead of a
    /// separate subtract-exp pass followed by a separate sum reduction,
    /// since every call site (online-softmax bookkeeping) needs both.
    fn sub_exp_sum_inplace(&self, x: &mut [f32], m: f32) -> f32;

    /// `dst[i] += src[i] * scale` for all `i` (equal-length slices).
    fn axpy(&self, dst: &mut [f32], src: &[f32], scale: f32);

    /// PV accumulation for a 4-query-row group against a whole KV tile:
    /// `acc[r][i] += sum_j p[r][j] * v_block[j][i]` for `r` in `0..4`,
    /// where `v_block` is `bc` contiguous rows of `d_head` elements and
    /// `p[r]` is row `r`'s `bc` already-exp'd probabilities. Unlike a naive
    /// "one `axpy` per V-row" loop, this keeps each `d_head`-chunk's
    /// accumulator registers resident across the *entire* `bc` sweep,
    /// touching `acc` in memory only once per chunk instead of once per
    /// V-row — the PV-side analog of [`Kernel::dot4x4`]'s insight, just
    /// eliminating accumulator round-trips instead of operand reloads.
    /// Each implementation processes 2 native-width chunks (8 independent
    /// accumulator chains) per outer step rather than 1 (4 chains): with
    /// only 4 chains, each carrying a genuine sequential FMA dependency
    /// across the whole (often 128+) `bc` sweep, there isn't enough
    /// concurrent independent work to hide FMA latency — doubling the
    /// chain count measured a further ~1.8-2.3x on top of the original
    /// one-`axpy`-per-V-row pattern's ~1.2x (see the crate's benchmarks).
    fn pv4(&self, acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]);

    /// `dst[i] *= scale` for all `i`.
    fn scale_inplace(&self, dst: &mut [f32], scale: f32);

    /// Max over all elements. Empty slice returns `-inf`.
    fn max_reduce(&self, x: &[f32]) -> f32;

    /// `[max_reduce(x[0]), max_reduce(x[1]), max_reduce(x[2]), max_reduce(x[3])]`
    /// — 4 independent max-reduction chains interleaved instead of 4
    /// separate sequential calls. Rows are mutually independent within one
    /// KV-tile (each row's max only depends on that row's own scores), but
    /// a single row's `max_reduce` is one dependency chain over `bc/lanes`
    /// iterations — often too short on its own to hide reduction latency,
    /// the same class of problem [`Kernel::pv4`] has. Used for the bulk of
    /// the online-softmax bookkeeping loop's row-max step, with the
    /// existing [`Kernel::max_reduce`] as the `this_br % 4` remainder.
    fn max_reduce4(&self, x: [&[f32]; 4]) -> [f32; 4];

    /// [`Kernel::sub_exp_sum_inplace`], 4 rows at once with per-row `m`
    /// values interleaved for the same reason as [`Kernel::max_reduce4`].
    fn sub_exp_sum_inplace4(&self, x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4];
}
