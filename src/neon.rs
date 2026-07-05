//! NEON accelerated kernel (aarch64 only).
//!
//! Unlike AVX2 on x86_64, NEON is part of the mandatory AArch64 baseline —
//! every aarch64 target Rust supports has it — so there's no runtime
//! feature check here (contrast [`crate::avx2`]'s `is_x86_feature_detected!`
//! dance): `flash_attention_v1`/`_v2`/`_v3` select this kernel unconditionally
//! on `#[cfg(target_arch = "aarch64")]`.
//!
//! `exp128_ps` mirrors `avx2::exp256_ps`'s algorithm exactly (range-reduce
//! `x = n*ln(2) + r`, degree-5 minimax polynomial for `exp(r)`, reconstruct
//! `2^n` via direct IEEE-754 exponent-bit manipulation), just over 4 lanes
//! instead of 8 and with NEON's fused-multiply-add argument order
//! (`vfmaq_f32(a, b, c) == a + b*c`, accumulator first — the reverse of
//! x86's `_mm256_fmadd_ps(a, b, c) == a*b + c`).
//!
//! Gated at the `mod neon;` declaration in `lib.rs` (`#[cfg(target_arch =
//! "aarch64")]`), not here — an inner `#![cfg(...)]` matching the outer one
//! is redundant (`clippy::duplicated_attributes`).

use crate::kernel::Kernel;
use std::arch::aarch64::*;

pub(crate) struct NeonKernel;

impl Kernel for NeonKernel {
    #[inline]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        dot_neon(a, b)
    }

    #[inline]
    unsafe fn exp_inplace(x: &mut [f32]) {
        exp_inplace_neon(x)
    }

    #[inline]
    unsafe fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
        axpy_neon(dst, src, scale)
    }

    #[inline]
    unsafe fn scale_inplace(dst: &mut [f32], scale: f32) {
        scale_neon(dst, scale)
    }

    #[inline]
    unsafe fn max_reduce(x: &[f32]) -> f32 {
        max_reduce_neon(x)
    }

    #[inline]
    unsafe fn sum_reduce(x: &[f32]) -> f32 {
        sum_reduce_neon(x)
    }
}

/// Dot product, 2-way accumulator unrolled (8 f32 / iteration) to hide FMA
/// latency behind independent accumulation chains — same idea as
/// `avx2::dot_avx2`, just 4-lane instead of 8-lane vectors.
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= len {
        let a0 = vld1q_f32(a.as_ptr().add(i));
        let b0 = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, a0, b0);
        let a1 = vld1q_f32(a.as_ptr().add(i + 4));
        let b1 = vld1q_f32(b.as_ptr().add(i + 4));
        acc1 = vfmaq_f32(acc1, a1, b1);
        i += 8;
    }
    while i + 4 <= len {
        let av = vld1q_f32(a.as_ptr().add(i));
        let bv = vld1q_f32(b.as_ptr().add(i));
        acc0 = vfmaq_f32(acc0, av, bv);
        i += 4;
    }
    let mut sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Vectorized exp over 4 lanes. See module docs for the algorithm.
///
/// The polynomial coefficients are the same published Cephes-derived
/// minimax fit `avx2::exp256_ps` uses, kept at full precision as documented
/// there rather than trimmed to placate `clippy::excessive_precision`.
#[target_feature(enable = "neon")]
#[allow(clippy::excessive_precision)]
unsafe fn exp128_ps(x: float32x4_t) -> float32x4_t {
    let exp_hi = vdupq_n_f32(88.376_26_f32);
    let exp_lo = vdupq_n_f32(-88.376_26_f32);
    let log2ef = vdupq_n_f32(std::f32::consts::LOG2_E);
    let half = vdupq_n_f32(0.5_f32);
    let c1 = vdupq_n_f32(0.693_359_375_f32);
    let c2 = vdupq_n_f32(-2.121_944_4e-4_f32);
    let p0 = vdupq_n_f32(1.987_569_15e-4_f32);
    let p1 = vdupq_n_f32(1.398_199_950_7e-3_f32);
    let p2 = vdupq_n_f32(8.333_451_907_3e-3_f32);
    let p3 = vdupq_n_f32(4.166_579_589_4e-2_f32);
    let p4 = vdupq_n_f32(1.666_666_545_9e-1_f32);
    let p5 = vdupq_n_f32(5.000_000_120_1e-1_f32);
    let one = vdupq_n_f32(1.0_f32);

    let x = vminq_f32(x, exp_hi);
    let x = vmaxq_f32(x, exp_lo);

    // n = floor(x / ln(2) + 0.5)
    let fx = vfmaq_f32(half, x, log2ef);
    let fx = vrndmq_f32(fx);

    // r = x - n*ln(2), split hi/lo for precision
    let tmp = vmulq_f32(fx, c1);
    let z = vmulq_f32(fx, c2);
    let x = vsubq_f32(x, tmp);
    let x = vsubq_f32(x, z);

    let z = vmulq_f32(x, x);

    // degree-5 minimax polynomial for exp(r)
    let mut y = p0;
    y = vfmaq_f32(p1, y, x);
    y = vfmaq_f32(p2, y, x);
    y = vfmaq_f32(p3, y, x);
    y = vfmaq_f32(p4, y, x);
    y = vfmaq_f32(p5, y, x);
    y = vfmaq_f32(x, y, z);
    y = vaddq_f32(y, one);

    // 2^n via direct exponent-bit construction
    let imm0 = vcvtq_s32_f32(fx);
    let imm0 = vaddq_s32(imm0, vdupq_n_s32(0x7f));
    let imm0 = vshlq_n_s32::<23>(imm0);
    let pow2n = vreinterpretq_f32_s32(imm0);

    vmulq_f32(y, pow2n)
}

#[target_feature(enable = "neon")]
unsafe fn exp_inplace_neon(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0usize;
    while i + 4 <= len {
        let v = vld1q_f32(x.as_ptr().add(i));
        let r = exp128_ps(v);
        vst1q_f32(x.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < len {
        x[i] = x[i].exp();
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn axpy_neon(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();
    let vscale = vdupq_n_f32(scale);
    let mut i = 0usize;
    while i + 4 <= len {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let s = vld1q_f32(src.as_ptr().add(i));
        let r = vfmaq_f32(d, s, vscale);
        vst1q_f32(dst.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < len {
        dst[i] += src[i] * scale;
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn scale_neon(dst: &mut [f32], scale: f32) {
    let len = dst.len();
    let vscale = vdupq_n_f32(scale);
    let mut i = 0usize;
    while i + 4 <= len {
        let d = vld1q_f32(dst.as_ptr().add(i));
        let r = vmulq_f32(d, vscale);
        vst1q_f32(dst.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < len {
        dst[i] *= scale;
        i += 1;
    }
}

#[target_feature(enable = "neon")]
unsafe fn max_reduce_neon(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }
    let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = vld1q_f32(x.as_ptr().add(i));
        acc = vmaxq_f32(acc, v);
        i += 4;
    }
    let mut m = vmaxvq_f32(acc);
    while i < len {
        m = m.max(x[i]);
        i += 1;
    }
    m
}

#[target_feature(enable = "neon")]
unsafe fn sum_reduce_neon(x: &[f32]) -> f32 {
    let len = x.len();
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = vld1q_f32(x.as_ptr().add(i));
        acc = vaddq_f32(acc, v);
        i += 4;
    }
    let mut s = vaddvq_f32(acc);
    while i < len {
        s += x[i];
        i += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std() {
        let xs: Vec<f32> = (-800..800).map(|i| i as f32 * 0.1).collect();
        let mut got = xs.clone();
        unsafe { exp_inplace_neon(&mut got) };
        let mut max_rel_err = 0.0f32;
        for (x, g) in xs.iter().zip(got.iter()) {
            let want = x.exp();
            if want.is_finite() && want > 1e-30 {
                let rel_err = ((g - want) / want).abs();
                max_rel_err = max_rel_err.max(rel_err);
                assert!(
                    rel_err < 1e-5,
                    "exp({x}) got {g}, want {want}, rel_err {rel_err}"
                );
            }
        }
        eprintln!("max relative error vs f32::exp: {max_rel_err:e}");
    }

    #[test]
    fn dot_matches_scalar() {
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.71).cos()).collect();
            let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let simd = unsafe { dot_neon(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3 * (scalar.abs() + 1.0),
                "len={len} scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn reductions_match_scalar() {
        for len in [0usize, 1, 5, 8, 13, 16, 33] {
            let x: Vec<f32> = (0..len).map(|i| ((i as f32) * 1.3).sin() * 5.0).collect();
            let want_sum: f32 = x.iter().sum();
            let want_max: f32 = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let got_sum = unsafe { sum_reduce_neon(&x) };
            let got_max = unsafe { max_reduce_neon(&x) };
            assert!((want_sum - got_sum).abs() < 1e-3, "len={len} sum");
            assert_eq!(want_max, got_max, "len={len} max");
        }
    }

    #[test]
    fn axpy_and_scale_match_scalar() {
        for len in [0usize, 1, 7, 8, 9, 31, 32, 33] {
            let mut dst: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let src: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();
            let scale = 1.37f32;
            let mut want = dst.clone();
            for (d, s) in want.iter_mut().zip(src.iter()) {
                *d += s * scale;
            }
            unsafe { axpy_neon(&mut dst, &src, scale) };
            for (w, g) in want.iter().zip(dst.iter()) {
                assert!((w - g).abs() < 1e-4, "axpy len={len}");
            }

            let mut d2 = want.clone();
            let want2: Vec<f32> = d2.iter().map(|v| v * 2.5).collect();
            unsafe { scale_neon(&mut d2, 2.5) };
            for (w, g) in want2.iter().zip(d2.iter()) {
                assert!((w - g).abs() < 1e-4, "scale len={len}");
            }
        }
    }
}
