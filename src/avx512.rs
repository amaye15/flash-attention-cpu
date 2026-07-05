//! AVX-512F accelerated kernel (x86_64 only).
//!
//! Every intrinsics-using function is `#[target_feature(enable = "avx512f")]`
//! and `unsafe`. Callers (via [`crate::kernel::Kernel`]) must only reach
//! these after `is_x86_feature_detected!("avx512f")` returns true, which
//! each variant's entry point checks once, up front, ahead of the AVX2
//! check — same runtime-dispatch pattern as [`crate::avx2`], just one tier
//! higher: 16 lanes/instruction instead of 8, on hosts that have it.
//!
//! `exp512_ps` is the same algorithm as `avx2::exp256_ps` (range-reduce,
//! degree-5 minimax polynomial, direct IEEE-754 exponent-bit
//! reconstruction of `2^n`) over 16 lanes instead of 8. Two things are
//! genuinely simpler at this width: AVX-512F provides direct horizontal
//! `_mm512_reduce_add_ps`/`_mm512_reduce_max_ps` intrinsics (no manual
//! shuffle-and-add tree like `avx2::hsum256_ps`/`hmax256_ps`), and floor is
//! `_mm512_roundscale_ps` with a rounding-mode immediate rather than a
//! dedicated floor instruction (AVX-512 generalized rounding into one
//! instruction family instead of separate floor/ceil/round ops).
//!
//! Gated at the `mod avx512;` declaration in `lib.rs`
//! (`#[cfg(target_arch = "x86_64")]`), not here — see `avx2.rs`'s module
//! docs for why an inner `#![cfg(...)]` would be redundant.

use crate::kernel::Kernel;
use std::arch::x86_64::*;

/// Round toward negative infinity, no-exception: the immediate for
/// `_mm512_roundscale_ps` that implements `floor`.
const ROUND_FLOOR: i32 = _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC;

pub(crate) struct Avx512Kernel;

impl Kernel for Avx512Kernel {
    #[inline]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        dot_avx512(a, b)
    }

    #[inline]
    unsafe fn exp_inplace(x: &mut [f32]) {
        exp_inplace_avx512(x)
    }

    #[inline]
    unsafe fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
        axpy_avx512(dst, src, scale)
    }

    #[inline]
    unsafe fn scale_inplace(dst: &mut [f32], scale: f32) {
        scale_avx512(dst, scale)
    }

    #[inline]
    unsafe fn max_reduce(x: &[f32]) -> f32 {
        max_reduce_avx512(x)
    }

    #[inline]
    unsafe fn sum_reduce(x: &[f32]) -> f32 {
        sum_reduce_avx512(x)
    }
}

/// Dot product, 2-way accumulator unrolled (32 f32 / iteration) to hide
/// FMA latency behind independent accumulation chains — same idea as
/// `avx2::dot_avx2`, just 16-lane instead of 8-lane vectors.
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= len {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        i += 32;
    }
    while i + 16 <= len {
        let av = _mm512_loadu_ps(a.as_ptr().add(i));
        let bv = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(av, bv, acc0);
        i += 16;
    }
    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Vectorized exp over 16 lanes. See module docs for the algorithm.
///
/// The polynomial coefficients below are the same published Cephes-derived
/// minimax fit as `avx2::exp256_ps` uses, kept at full precision as
/// documented there rather than trimmed to placate
/// `clippy::excessive_precision`.
#[target_feature(enable = "avx512f")]
#[allow(clippy::excessive_precision)]
unsafe fn exp512_ps(x: __m512) -> __m512 {
    let exp_hi = _mm512_set1_ps(88.376_26_f32);
    let exp_lo = _mm512_set1_ps(-88.376_26_f32);
    let log2ef = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let half = _mm512_set1_ps(0.5_f32);
    let c1 = _mm512_set1_ps(0.693_359_375_f32);
    let c2 = _mm512_set1_ps(-2.121_944_4e-4_f32);
    let p0 = _mm512_set1_ps(1.987_569_15e-4_f32);
    let p1 = _mm512_set1_ps(1.398_199_950_7e-3_f32);
    let p2 = _mm512_set1_ps(8.333_451_907_3e-3_f32);
    let p3 = _mm512_set1_ps(4.166_579_589_4e-2_f32);
    let p4 = _mm512_set1_ps(1.666_666_545_9e-1_f32);
    let p5 = _mm512_set1_ps(5.000_000_120_1e-1_f32);
    let one = _mm512_set1_ps(1.0_f32);

    let x = _mm512_min_ps(x, exp_hi);
    let x = _mm512_max_ps(x, exp_lo);

    // n = floor(x / ln(2) + 0.5)
    let fx = _mm512_fmadd_ps(x, log2ef, half);
    let fx = _mm512_roundscale_ps::<ROUND_FLOOR>(fx);

    // r = x - n*ln(2), split hi/lo for precision
    let tmp = _mm512_mul_ps(fx, c1);
    let z = _mm512_mul_ps(fx, c2);
    let x = _mm512_sub_ps(x, tmp);
    let x = _mm512_sub_ps(x, z);

    let z = _mm512_mul_ps(x, x);

    // degree-5 minimax polynomial for exp(r)
    let mut y = p0;
    y = _mm512_fmadd_ps(y, x, p1);
    y = _mm512_fmadd_ps(y, x, p2);
    y = _mm512_fmadd_ps(y, x, p3);
    y = _mm512_fmadd_ps(y, x, p4);
    y = _mm512_fmadd_ps(y, x, p5);
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, one);

    // 2^n via direct exponent-bit construction
    let imm0 = _mm512_cvttps_epi32(fx);
    let imm0 = _mm512_add_epi32(imm0, _mm512_set1_epi32(0x7f));
    let imm0 = _mm512_slli_epi32::<23>(imm0);
    let pow2n = _mm512_castsi512_ps(imm0);

    _mm512_mul_ps(y, pow2n)
}

#[target_feature(enable = "avx512f")]
unsafe fn exp_inplace_avx512(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0usize;
    while i + 16 <= len {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        let r = exp512_ps(v);
        _mm512_storeu_ps(x.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < len {
        x[i] = x[i].exp();
        i += 1;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn axpy_avx512(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();
    let vscale = _mm512_set1_ps(scale);
    let mut i = 0usize;
    while i + 16 <= len {
        let d = _mm512_loadu_ps(dst.as_ptr().add(i));
        let s = _mm512_loadu_ps(src.as_ptr().add(i));
        let r = _mm512_fmadd_ps(s, vscale, d);
        _mm512_storeu_ps(dst.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < len {
        dst[i] += src[i] * scale;
        i += 1;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn scale_avx512(dst: &mut [f32], scale: f32) {
    let len = dst.len();
    let vscale = _mm512_set1_ps(scale);
    let mut i = 0usize;
    while i + 16 <= len {
        let d = _mm512_loadu_ps(dst.as_ptr().add(i));
        let r = _mm512_mul_ps(d, vscale);
        _mm512_storeu_ps(dst.as_mut_ptr().add(i), r);
        i += 16;
    }
    while i < len {
        dst[i] *= scale;
        i += 1;
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn max_reduce_avx512(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }
    let mut acc = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 16 <= len {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        acc = _mm512_max_ps(acc, v);
        i += 16;
    }
    let mut m = _mm512_reduce_max_ps(acc);
    while i < len {
        m = m.max(x[i]);
        i += 1;
    }
    m
}

#[target_feature(enable = "avx512f")]
unsafe fn sum_reduce_avx512(x: &[f32]) -> f32 {
    let len = x.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= len {
        let v = _mm512_loadu_ps(x.as_ptr().add(i));
        acc = _mm512_add_ps(acc, v);
        i += 16;
    }
    let mut s = _mm512_reduce_add_ps(acc);
    while i < len {
        s += x[i];
        i += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avx512_available() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    #[test]
    fn exp_matches_std() {
        if !avx512_available() {
            return;
        }
        let xs: Vec<f32> = (-800..800).map(|i| i as f32 * 0.1).collect();
        let mut got = xs.clone();
        unsafe { exp_inplace_avx512(&mut got) };
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
        if !avx512_available() {
            return;
        }
        for len in [0usize, 1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.71).cos()).collect();
            let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let simd = unsafe { dot_avx512(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3 * (scalar.abs() + 1.0),
                "len={len} scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn reductions_match_scalar() {
        if !avx512_available() {
            return;
        }
        for len in [0usize, 1, 5, 8, 16, 17, 33] {
            let x: Vec<f32> = (0..len).map(|i| ((i as f32) * 1.3).sin() * 5.0).collect();
            let want_sum: f32 = x.iter().sum();
            let want_max: f32 = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let got_sum = unsafe { sum_reduce_avx512(&x) };
            let got_max = unsafe { max_reduce_avx512(&x) };
            assert!((want_sum - got_sum).abs() < 1e-3, "len={len} sum");
            assert_eq!(want_max, got_max, "len={len} max");
        }
    }

    #[test]
    fn axpy_and_scale_match_scalar() {
        if !avx512_available() {
            return;
        }
        for len in [0usize, 1, 7, 8, 16, 17, 31, 32, 33] {
            let mut dst: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let src: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();
            let scale = 1.37f32;
            let mut want = dst.clone();
            for (d, s) in want.iter_mut().zip(src.iter()) {
                *d += s * scale;
            }
            unsafe { axpy_avx512(&mut dst, &src, scale) };
            for (w, g) in want.iter().zip(dst.iter()) {
                assert!((w - g).abs() < 1e-4, "axpy len={len}");
            }

            let mut d2 = want.clone();
            let want2: Vec<f32> = d2.iter().map(|v| v * 2.5).collect();
            unsafe { scale_avx512(&mut d2, 2.5) };
            for (w, g) in want2.iter().zip(d2.iter()) {
                assert!((w - g).abs() < 1e-4, "scale len={len}");
            }
        }
    }
}
