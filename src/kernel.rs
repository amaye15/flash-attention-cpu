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

    /// In-place exponential: `x[i] = exp(x[i])` for all `i`.
    unsafe fn exp_inplace(x: &mut [f32]);

    /// `dst[i] += src[i] * scale` for all `i` (equal-length slices).
    unsafe fn axpy(dst: &mut [f32], src: &[f32], scale: f32);

    /// `dst[i] *= scale` for all `i`.
    unsafe fn scale_inplace(dst: &mut [f32], scale: f32);

    /// Max over all elements. Empty slice returns `-inf`.
    unsafe fn max_reduce(x: &[f32]) -> f32;

    /// Sum over all elements. Empty slice returns `0`.
    unsafe fn sum_reduce(x: &[f32]) -> f32;
}
