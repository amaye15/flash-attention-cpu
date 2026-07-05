//! The [`Kernel`] trait isolates the handful of numeric primitives that
//! differ between the portable scalar fallback and the accelerated SIMD
//! paths (AVX2+FMA on x86_64, NEON on aarch64). Everything else — tiling,
//! online-softmax running-stat bookkeeping, causal masking, parallel
//! dispatch — is written once, generically, per variant in `v1.rs`/
//! `v2.rs`/`v3.rs`, and monomorphized over whichever `Kernel` is selected at
//! runtime.

/// Numeric primitives used by the flash-attention inner loop.
///
/// All methods are `unsafe fn` so that [`crate::avx2::Avx2Kernel`],
/// [`crate::neon::NeonKernel`], and [`crate::scalar::ScalarKernel`] share
/// one signature: the AVX2 implementation is only sound to call after
/// checking `is_x86_feature_detected!("avx2")` / `"fma"`, which each
/// variant's entry point does once, up front, before selecting a kernel
/// type. NEON needs no such check — it's part of the mandatory AArch64
/// baseline.
pub(crate) trait Kernel {
    /// Dot product of two equal-length slices.
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32;

    /// `[dot(a0, b), dot(a1, b), dot(a2, b), dot(a3, b)]` — four dot
    /// products sharing one another's `b` vector loads. This is the
    /// register-blocking trick that improves the compute-to-load ratio
    /// over calling `dot` four times independently: in the naive
    /// per-`(query_row, key_row)` access pattern the QK^T/PV loops used to
    /// use exclusively, a fixed `b` row gets reloaded from scratch for
    /// every query row that visits it. Blocking four query rows together
    /// means each `b` row's vector loads are amortized across four FMA
    /// accumulator chains instead of one. Measured ~1.3-1.7x throughput
    /// improvement on the QK^T step alone (see the crate's benchmarks).
    unsafe fn dot4(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4];

    /// Fused in-place `x[i] = exp(x[i] - m)` for all `i` — one pass over
    /// `x` instead of a separate subtract-then-exp pass, since every call
    /// site needs exactly this (the online-softmax max-subtraction always
    /// immediately precedes the exponential).
    unsafe fn sub_exp_inplace(x: &mut [f32], m: f32);

    /// `dst[i] += src[i] * scale` for all `i` (equal-length slices).
    unsafe fn axpy(dst: &mut [f32], src: &[f32], scale: f32);

    /// `dst[k][i] += b[i] * scale[k]` for `k` in `0..4` — same
    /// register-blocking idea as [`Kernel::dot4`], applied to the PV
    /// accumulation: one streamed `b` (a V row) updates four destination
    /// rows instead of `b` being reloaded once per destination row.
    unsafe fn axpy4(dst: [&mut [f32]; 4], b: &[f32], scale: [f32; 4]);

    /// `dst[i] *= scale` for all `i`.
    unsafe fn scale_inplace(dst: &mut [f32], scale: f32);

    /// Max over all elements. Empty slice returns `-inf`.
    unsafe fn max_reduce(x: &[f32]) -> f32;

    /// Sum over all elements. Empty slice returns `0`.
    unsafe fn sum_reduce(x: &[f32]) -> f32;
}
