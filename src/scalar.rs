//! Portable scalar fallback. Used on x86_64 CPUs that lack AVX2/AVX-512,
//! and on any target that isn't x86_64, aarch64, or wasm32+simd128 (which
//! have dedicated SIMD kernels — see `avx2.rs`/`avx512.rs`/`neon.rs`/
//! `simd128.rs`). Written as plain iterator code so LLVM can still
//! autovectorize it reasonably well (SSE2 on the x86_64 baseline, NEON on
//! aarch64).

use crate::kernel::Kernel;

// On aarch64, and on wasm32 built with `+simd128`, `flash_attention_v1`/
// `_v2`/`_v3` dispatch unconditionally to `NeonKernel`/`Simd128Kernel` (see
// each module's entry point), so in a non-test build this struct is
// genuinely unused there — it's kept only for the
// `scalar_and_neon_agree_with_each_other`/`scalar_and_simd128_agree_with_each_other`
// cross-check tests.
#[cfg_attr(
    any(
        all(target_arch = "aarch64", not(test)),
        all(target_arch = "wasm32", target_feature = "simd128", not(test))
    ),
    allow(dead_code)
)]
pub(crate) struct ScalarKernel;

impl Kernel for ScalarKernel {
    #[inline]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    unsafe fn exp_inplace(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = v.exp();
        }
    }

    #[inline]
    unsafe fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
        debug_assert_eq!(dst.len(), src.len());
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += s * scale;
        }
    }

    #[inline]
    unsafe fn scale_inplace(dst: &mut [f32], scale: f32) {
        for d in dst.iter_mut() {
            *d *= scale;
        }
    }

    #[inline]
    unsafe fn max_reduce(x: &[f32]) -> f32 {
        x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    #[inline]
    unsafe fn sum_reduce(x: &[f32]) -> f32 {
        x.iter().sum()
    }
}
