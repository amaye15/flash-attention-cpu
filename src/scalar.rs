//! Portable scalar fallback. Used on x86_64 CPUs that lack AVX2/AVX-512,
//! and on any target that isn't x86_64, aarch64, or wasm32+simd128 (which
//! have dedicated SIMD kernels — see `avx2.rs`/`avx512.rs`/`neon.rs`/
//! `simd128.rs`). Written as plain iterator code so LLVM can still
//! autovectorize it reasonably well (SSE2 on the x86_64 baseline, NEON on
//! aarch64).
//!
//! Every method is plain safe Rust — no intrinsics, so no precondition to
//! prove and nothing for `Self::new()` to check. This is the one `Kernel`
//! impl in the crate with zero `unsafe` in it, start to finish.

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

impl ScalarKernel {
    /// Always succeeds — this kernel does nothing actually unsafe
    /// internally, so there's no precondition to check.
    #[cfg_attr(
        any(
            all(target_arch = "aarch64", not(test)),
            all(target_arch = "wasm32", target_feature = "simd128", not(test))
        ),
        allow(dead_code)
    )]
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Kernel for ScalarKernel {
    #[inline]
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    fn dot4(&self, a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
        // No register-blocking benefit possible without SIMD lanes to
        // share loads across — this exists purely so call sites in
        // `v1.rs`/`v2.rs`/`v3.rs` don't need an arch-specific branch.
        [
            self.dot(a0, b),
            self.dot(a1, b),
            self.dot(a2, b),
            self.dot(a3, b),
        ]
    }

    #[inline]
    fn dot4x4(&self, q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
        // No register-blocking benefit possible without SIMD lanes to
        // share loads across — this exists purely so call sites in
        // `v1.rs`/`v2.rs`/`v3.rs` don't need an arch-specific branch.
        std::array::from_fn(|r| std::array::from_fn(|c| self.dot(q[r], k[c])))
    }

    #[inline]
    fn sub_exp_sum_inplace(&self, x: &mut [f32], m: f32) -> f32 {
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        sum
    }

    #[inline]
    fn axpy(&self, dst: &mut [f32], src: &[f32], scale: f32) {
        debug_assert_eq!(dst.len(), src.len());
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += s * scale;
        }
    }

    #[inline]
    fn pv4(&self, acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
        let [a0, a1, a2, a3] = acc;
        let d = a0.len();
        for (j, v_row) in v_block.chunks_exact(d).enumerate() {
            self.axpy(a0, v_row, p[0][j]);
            self.axpy(a1, v_row, p[1][j]);
            self.axpy(a2, v_row, p[2][j]);
            self.axpy(a3, v_row, p[3][j]);
        }
    }

    #[inline]
    fn scale_inplace(&self, dst: &mut [f32], scale: f32) {
        for d in dst.iter_mut() {
            *d *= scale;
        }
    }

    #[inline]
    fn max_reduce(&self, x: &[f32]) -> f32 {
        x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    #[inline]
    fn max_reduce4(&self, x: [&[f32]; 4]) -> [f32; 4] {
        // No ILP benefit possible without independent SIMD chains to
        // interleave — this exists purely so call sites don't need an
        // arch-specific branch.
        std::array::from_fn(|r| self.max_reduce(x[r]))
    }

    #[inline]
    fn sub_exp_sum_inplace4(&self, x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
        let [x0, x1, x2, x3] = x;
        [
            self.sub_exp_sum_inplace(x0, m[0]),
            self.sub_exp_sum_inplace(x1, m[1]),
            self.sub_exp_sum_inplace(x2, m[2]),
            self.sub_exp_sum_inplace(x3, m[3]),
        ]
    }
}
