//! WASM SIMD128 accelerated kernel (wasm32 only, and only when compiled
//! with the `simd128` target feature enabled — it is not part of the
//! WebAssembly MVP, so unlike NEON on aarch64 it isn't always available;
//! unlike AVX2/AVX-512 on x86_64 there's also no runtime feature-detection
//! mechanism to fall back from, since WASM doesn't expose CPU-feature
//! queries to the module the way native targets do. Build with
//! `RUSTFLAGS="-C target-feature=+simd128"` (or a `.cargo/config.toml`
//! `[target.wasm32-unknown-unknown] rustflags` entry) to opt in; without
//! it, this module doesn't exist and every variant falls back to
//! [`crate::scalar`].
//!
//! **No native fused-multiply-add.** WASM SIMD128's baseline instruction
//! set (unlike AVX2/AVX-512's `vfmadd*`/NEON's `vfmaq_f32`) has no fused
//! multiply-add — `dot`/`axpy` here are a separate multiply then add, two
//! roundings instead of one. (WASM's "relaxed-simd" proposal adds a
//! relaxed FMA, but it isn't part of stable `simd128` and isn't assumed
//! available here.)
//!
//! `exp128_ps` is the same algorithm as `avx2::exp256_ps`/
//! `neon::exp128_ps` (range-reduce, degree-5 minimax polynomial, direct
//! IEEE-754 exponent-bit reconstruction of `2^n`), over 4 lanes. One
//! simplification versus AVX2/NEON: `core::arch::wasm32` has a single
//! `v128` type for all lane interpretations (no distinct `__m256`/`__m256i`
//! or separate NEON vector types), so there's no bitcast needed to move
//! from an integer exponent back to a float — and `f32x4_floor` exists
//! directly, unlike AVX2 (needs `_mm256_floor_ps`, fine) or the
//! integer-conversion trick some ISAs require.

use crate::kernel::Kernel;
use std::arch::wasm32::*;

pub(crate) struct Simd128Kernel;

impl Kernel for Simd128Kernel {
    #[inline]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        dot_simd128(a, b)
    }

    #[inline]
    unsafe fn exp_inplace(x: &mut [f32]) {
        exp_inplace_simd128(x)
    }

    #[inline]
    unsafe fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
        axpy_simd128(dst, src, scale)
    }

    #[inline]
    unsafe fn scale_inplace(dst: &mut [f32], scale: f32) {
        scale_simd128(dst, scale)
    }

    #[inline]
    unsafe fn max_reduce(x: &[f32]) -> f32 {
        max_reduce_simd128(x)
    }

    #[inline]
    unsafe fn sum_reduce(x: &[f32]) -> f32 {
        sum_reduce_simd128(x)
    }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn hsum128_ps(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v)
        + f32x4_extract_lane::<1>(v)
        + f32x4_extract_lane::<2>(v)
        + f32x4_extract_lane::<3>(v)
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn hmax128_ps(v: v128) -> f32 {
    f32x4_extract_lane::<0>(v)
        .max(f32x4_extract_lane::<1>(v))
        .max(f32x4_extract_lane::<2>(v))
        .max(f32x4_extract_lane::<3>(v))
}

/// Dot product, 2-way accumulator unrolled (8 f32 / iteration) — same idea
/// as `avx2::dot_avx2`/`neon::dot_neon`, just without a fused multiply-add
/// (see module docs), so each step is a separate `mul` + `add`.
#[target_feature(enable = "simd128")]
unsafe fn dot_simd128(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0usize;
    while i + 8 <= len {
        let a0 = v128_load(a.as_ptr().add(i) as *const v128);
        let b0 = v128_load(b.as_ptr().add(i) as *const v128);
        acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));
        let a1 = v128_load(a.as_ptr().add(i + 4) as *const v128);
        let b1 = v128_load(b.as_ptr().add(i + 4) as *const v128);
        acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
        i += 8;
    }
    while i + 4 <= len {
        let av = v128_load(a.as_ptr().add(i) as *const v128);
        let bv = v128_load(b.as_ptr().add(i) as *const v128);
        acc0 = f32x4_add(acc0, f32x4_mul(av, bv));
        i += 4;
    }
    let mut sum = hsum128_ps(f32x4_add(acc0, acc1));
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Vectorized exp over 4 lanes. See module docs for the algorithm.
///
/// The polynomial coefficients below are the same published Cephes-derived
/// minimax fit `avx2::exp256_ps`/`neon::exp128_ps` use, kept at full
/// precision as documented there rather than trimmed to placate
/// `clippy::excessive_precision`.
#[target_feature(enable = "simd128")]
#[allow(clippy::excessive_precision)]
unsafe fn exp128_ps(x: v128) -> v128 {
    let exp_hi = f32x4_splat(88.376_26_f32);
    let exp_lo = f32x4_splat(-88.376_26_f32);
    let log2ef = f32x4_splat(std::f32::consts::LOG2_E);
    let half = f32x4_splat(0.5_f32);
    let c1 = f32x4_splat(0.693_359_375_f32);
    let c2 = f32x4_splat(-2.121_944_4e-4_f32);
    let p0 = f32x4_splat(1.987_569_15e-4_f32);
    let p1 = f32x4_splat(1.398_199_950_7e-3_f32);
    let p2 = f32x4_splat(8.333_451_907_3e-3_f32);
    let p3 = f32x4_splat(4.166_579_589_4e-2_f32);
    let p4 = f32x4_splat(1.666_666_545_9e-1_f32);
    let p5 = f32x4_splat(5.000_000_120_1e-1_f32);
    let one = f32x4_splat(1.0_f32);

    let x = f32x4_min(x, exp_hi);
    let x = f32x4_max(x, exp_lo);

    // n = floor(x / ln(2) + 0.5)
    let fx = f32x4_add(f32x4_mul(x, log2ef), half);
    let fx = f32x4_floor(fx);

    // r = x - n*ln(2), split hi/lo for precision
    let tmp = f32x4_mul(fx, c1);
    let z = f32x4_mul(fx, c2);
    let x = f32x4_sub(x, tmp);
    let x = f32x4_sub(x, z);

    let z = f32x4_mul(x, x);

    // degree-5 minimax polynomial for exp(r)
    let mut y = p0;
    y = f32x4_add(f32x4_mul(y, x), p1);
    y = f32x4_add(f32x4_mul(y, x), p2);
    y = f32x4_add(f32x4_mul(y, x), p3);
    y = f32x4_add(f32x4_mul(y, x), p4);
    y = f32x4_add(f32x4_mul(y, x), p5);
    y = f32x4_add(f32x4_mul(y, z), x);
    y = f32x4_add(y, one);

    // 2^n via direct exponent-bit construction. No bitcast needed: `v128`
    // is a single type for every lane interpretation in this API.
    let imm0 = i32x4_trunc_sat_f32x4(fx);
    let imm0 = i32x4_add(imm0, i32x4_splat(0x7f));
    let pow2n = i32x4_shl(imm0, 23);

    f32x4_mul(y, pow2n)
}

#[target_feature(enable = "simd128")]
unsafe fn exp_inplace_simd128(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0usize;
    while i + 4 <= len {
        let v = v128_load(x.as_ptr().add(i) as *const v128);
        let r = exp128_ps(v);
        v128_store(x.as_mut_ptr().add(i) as *mut v128, r);
        i += 4;
    }
    while i < len {
        x[i] = x[i].exp();
        i += 1;
    }
}

#[target_feature(enable = "simd128")]
unsafe fn axpy_simd128(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();
    let vscale = f32x4_splat(scale);
    let mut i = 0usize;
    while i + 4 <= len {
        let d = v128_load(dst.as_ptr().add(i) as *const v128);
        let s = v128_load(src.as_ptr().add(i) as *const v128);
        let r = f32x4_add(d, f32x4_mul(s, vscale));
        v128_store(dst.as_mut_ptr().add(i) as *mut v128, r);
        i += 4;
    }
    while i < len {
        dst[i] += src[i] * scale;
        i += 1;
    }
}

#[target_feature(enable = "simd128")]
unsafe fn scale_simd128(dst: &mut [f32], scale: f32) {
    let len = dst.len();
    let vscale = f32x4_splat(scale);
    let mut i = 0usize;
    while i + 4 <= len {
        let d = v128_load(dst.as_ptr().add(i) as *const v128);
        let r = f32x4_mul(d, vscale);
        v128_store(dst.as_mut_ptr().add(i) as *mut v128, r);
        i += 4;
    }
    while i < len {
        dst[i] *= scale;
        i += 1;
    }
}

#[target_feature(enable = "simd128")]
unsafe fn max_reduce_simd128(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }
    let mut acc = f32x4_splat(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = v128_load(x.as_ptr().add(i) as *const v128);
        acc = f32x4_max(acc, v);
        i += 4;
    }
    let mut m = hmax128_ps(acc);
    while i < len {
        m = m.max(x[i]);
        i += 1;
    }
    m
}

#[target_feature(enable = "simd128")]
unsafe fn sum_reduce_simd128(x: &[f32]) -> f32 {
    let len = x.len();
    let mut acc = f32x4_splat(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = v128_load(x.as_ptr().add(i) as *const v128);
        acc = f32x4_add(acc, v);
        i += 4;
    }
    let mut s = hsum128_ps(acc);
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
        unsafe { exp_inplace_simd128(&mut got) };
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
            let simd = unsafe { dot_simd128(&a, &b) };
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
            let got_sum = unsafe { sum_reduce_simd128(&x) };
            let got_max = unsafe { max_reduce_simd128(&x) };
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
            unsafe { axpy_simd128(&mut dst, &src, scale) };
            for (w, g) in want.iter().zip(dst.iter()) {
                assert!((w - g).abs() < 1e-4, "axpy len={len}");
            }

            let mut d2 = want.clone();
            let want2: Vec<f32> = d2.iter().map(|v| v * 2.5).collect();
            unsafe { scale_simd128(&mut d2, 2.5) };
            for (w, g) in want2.iter().zip(d2.iter()) {
                assert!((w - g).abs() < 1e-4, "scale len={len}");
            }
        }
    }
}
