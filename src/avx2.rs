//! AVX2+FMA accelerated kernel (x86_64 only).
//!
//! Every intrinsics-using function is `#[target_feature(enable = "avx2,fma")]`
//! and `unsafe`. Callers (via [`crate::kernel::Kernel`]) must only reach
//! these after `is_x86_feature_detected!("avx2")` and `"fma"` both return
//! true, which `flash_attention` checks exactly once, up front. This is the
//! standard "runtime dispatch" pattern: the crate compiles to a portable
//! binary that still uses AVX2 wherever the host CPU actually supports it,
//! rather than requiring `-C target-cpu=native` (which would crash with
//! SIGILL on older hardware).
//!
//! `exp256_ps` is a vectorized single-precision exponential: range-reduce
//! `x = n*ln(2) + r`, compute `exp(r)` via a degree-5 minimax polynomial,
//! and reconstruct `2^n` via direct IEEE-754 exponent-bit manipulation. This
//! is the same family of technique used in most SIMD math libraries
//! (Cephes-derived). It trades a couple ULP of accuracy against `f32::exp`
//! for the ability to process 8 lanes per instruction with no libm call
//! overhead — see `tests::exp_matches_std` for the accuracy check.
//!
//! Gated at the `mod avx2;` declaration in `lib.rs` (`#[cfg(target_arch =
//! "x86_64")]`), not here — an inner `#![cfg(...)]` matching the outer one
//! is redundant (`clippy::duplicated_attributes`) and, unlike the outer
//! gate, invisible when this file simply isn't compiled for other targets.

use crate::kernel::Kernel;
use std::arch::x86_64::*;

pub(crate) struct Avx2Kernel;

impl Kernel for Avx2Kernel {
    #[inline]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        dot_avx2(a, b)
    }

    #[inline]
    unsafe fn dot4(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
        dot4_avx2(a0, a1, a2, a3, b)
    }

    #[inline]
    unsafe fn dot4x4(q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
        dot4x4_avx2(q, k)
    }

    #[inline]
    unsafe fn sub_exp_sum_inplace(x: &mut [f32], m: f32) -> f32 {
        sub_exp_sum_inplace_avx2(x, m)
    }

    #[inline]
    unsafe fn sub_exp_sum_inplace4(x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
        sub_exp_sum_inplace4_avx2(x, m)
    }

    #[inline]
    unsafe fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
        axpy_avx2(dst, src, scale)
    }

    #[inline]
    unsafe fn pv4(acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
        pv4_avx2(acc, v_block, p)
    }

    #[inline]
    unsafe fn scale_inplace(dst: &mut [f32], scale: f32) {
        scale_avx2(dst, scale)
    }

    #[inline]
    unsafe fn max_reduce(x: &[f32]) -> f32 {
        max_reduce_avx2(x)
    }

    #[inline]
    unsafe fn max_reduce4(x: [&[f32]; 4]) -> [f32; 4] {
        max_reduce4_avx2(x)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hmax256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(hi, lo);
    let shuf = _mm_movehdup_ps(max128);
    let maxs = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, maxs);
    let result = _mm_max_ss(maxs, shuf2);
    _mm_cvtss_f32(result)
}

/// Dot product, 2-way accumulator unrolled (16 f32 / iteration) to hide
/// FMA latency behind independent accumulation chains.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }
    while i + 8 <= len {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(av, bv, acc0);
        i += 8;
    }
    let mut sum = hsum256_ps(_mm256_add_ps(acc0, acc1));
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Four dot products sharing `b`'s vector loads across four independent FMA
/// accumulator chains — see [`crate::kernel::Kernel::dot4`] for why this is
/// faster than four separate [`dot_avx2`] calls.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    debug_assert_eq!(a0.len(), b.len());
    debug_assert_eq!(a1.len(), b.len());
    debug_assert_eq!(a2.len(), b.len());
    debug_assert_eq!(a3.len(), b.len());
    let len = b.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let bv = _mm256_loadu_ps(b.as_ptr().add(i)); // loaded once, shared 4 ways
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0.as_ptr().add(i)), bv, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1.as_ptr().add(i)), bv, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2.as_ptr().add(i)), bv, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3.as_ptr().add(i)), bv, acc3);
        i += 8;
    }
    let mut sums = [
        hsum256_ps(acc0),
        hsum256_ps(acc1),
        hsum256_ps(acc2),
        hsum256_ps(acc3),
    ];
    while i < len {
        sums[0] += a0[i] * b[i];
        sums[1] += a1[i] * b[i];
        sums[2] += a2[i] * b[i];
        sums[3] += a3[i] * b[i];
        i += 1;
    }
    sums
}

/// 4 query rows x 4 key rows blocked together — see
/// [`crate::kernel::Kernel::dot4x4`]. 16 accumulators is comfortable on
/// NEON/AVX-512 (32 registers each) but leaves no spare AVX2 YMM registers
/// (only 16 total) for the per-iteration loads — plausible spill risk on
/// paper. Real CI numbers on `ubuntu-latest`/`windows-latest` (see the
/// README's Benchmarks section) show no such regression: `v2`/`v3` came
/// back ~1.4-1.6x faster with this than the prior one-sided `dot4`-only
/// path, so the theoretical register-pressure concern isn't costing
/// anything in practice on the x86_64 hardware actually measured.
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4x4_avx2(q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
    let d = q[0].len();
    let mut acc = [[_mm256_setzero_ps(); 4]; 4];
    let mut p = 0usize;
    while p + 8 <= d {
        let qv = [
            _mm256_loadu_ps(q[0].as_ptr().add(p)),
            _mm256_loadu_ps(q[1].as_ptr().add(p)),
            _mm256_loadu_ps(q[2].as_ptr().add(p)),
            _mm256_loadu_ps(q[3].as_ptr().add(p)),
        ];
        for c in 0..4 {
            let kv = _mm256_loadu_ps(k[c].as_ptr().add(p)); // loaded once, shared 4 ways
            for r in 0..4 {
                acc[r][c] = _mm256_fmadd_ps(qv[r], kv, acc[r][c]);
            }
        }
        p += 8;
    }
    let mut sums = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            sums[r][c] = hsum256_ps(acc[r][c]);
        }
    }
    while p < d {
        for r in 0..4 {
            for c in 0..4 {
                sums[r][c] += q[r][p] * k[c][p];
            }
        }
        p += 1;
    }
    sums
}

/// Vectorized exp over 8 lanes. See module docs for the algorithm.
///
/// The polynomial coefficients below are a published Cephes-derived
/// minimax fit and are intentionally kept at full precision as documented
/// rather than trimmed to placate `clippy::excessive_precision` — they're
/// empirically checked against `f32::exp` in `tests::exp_matches_std`, and
/// trimming them isn't something to do without re-validating.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::excessive_precision)]
unsafe fn exp256_ps(x: __m256) -> __m256 {
    let exp_hi = _mm256_set1_ps(88.376_26_f32);
    let exp_lo = _mm256_set1_ps(-88.376_26_f32);
    let log2ef = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let half = _mm256_set1_ps(0.5_f32);
    let c1 = _mm256_set1_ps(0.693_359_375_f32);
    let c2 = _mm256_set1_ps(-2.121_944_4e-4_f32);
    let p0 = _mm256_set1_ps(1.987_569_15e-4_f32);
    let p1 = _mm256_set1_ps(1.398_199_950_7e-3_f32);
    let p2 = _mm256_set1_ps(8.333_451_907_3e-3_f32);
    let p3 = _mm256_set1_ps(4.166_579_589_4e-2_f32);
    let p4 = _mm256_set1_ps(1.666_666_545_9e-1_f32);
    let p5 = _mm256_set1_ps(5.000_000_120_1e-1_f32);
    let one = _mm256_set1_ps(1.0_f32);

    let x = _mm256_min_ps(x, exp_hi);
    let x = _mm256_max_ps(x, exp_lo);

    // n = floor(x / ln(2) + 0.5)
    let fx = _mm256_fmadd_ps(x, log2ef, half);
    let fx = _mm256_floor_ps(fx);

    // r = x - n*ln(2), split hi/lo for precision
    let tmp = _mm256_mul_ps(fx, c1);
    let z = _mm256_mul_ps(fx, c2);
    let x = _mm256_sub_ps(x, tmp);
    let x = _mm256_sub_ps(x, z);

    let z = _mm256_mul_ps(x, x);

    // degree-5 minimax polynomial for exp(r)
    let mut y = p0;
    y = _mm256_fmadd_ps(y, x, p1);
    y = _mm256_fmadd_ps(y, x, p2);
    y = _mm256_fmadd_ps(y, x, p3);
    y = _mm256_fmadd_ps(y, x, p4);
    y = _mm256_fmadd_ps(y, x, p5);
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, one);

    // 2^n via direct exponent-bit construction
    let imm0 = _mm256_cvttps_epi32(fx);
    let imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    let imm0 = _mm256_slli_epi32(imm0, 23);
    let pow2n = _mm256_castsi256_ps(imm0);

    _mm256_mul_ps(y, pow2n)
}

/// Fused `x[i] = exp(x[i] - m)`, returning `sum(x)` after the exponential:
/// subtract, exponential, and sum accumulation all in the same pass over
/// `x` (one load/store per lane, plus the sum, instead of two separate
/// passes).
#[target_feature(enable = "avx2,fma")]
unsafe fn sub_exp_sum_inplace_avx2(x: &mut [f32], m: f32) -> f32 {
    let len = x.len();
    let vm = _mm256_set1_ps(m);
    let mut sum_acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let v = _mm256_sub_ps(v, vm);
        let r = exp256_ps(v);
        _mm256_storeu_ps(x.as_mut_ptr().add(i), r);
        sum_acc = _mm256_add_ps(sum_acc, r);
        i += 8;
    }
    let mut sum = hsum256_ps(sum_acc);
    while i < len {
        let e = (x[i] - m).exp();
        x[i] = e;
        sum += e;
        i += 1;
    }
    sum
}

/// [`sub_exp_sum_inplace_avx2`], 4 rows at once with per-row `m` values,
/// interleaved into 4 independent chains — see
/// [`crate::kernel::Kernel::sub_exp_sum_inplace4`] for why.
#[target_feature(enable = "avx2,fma")]
unsafe fn sub_exp_sum_inplace4_avx2(x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
    let [x0, x1, x2, x3] = x;
    let len = x0.len();
    debug_assert_eq!(x1.len(), len);
    debug_assert_eq!(x2.len(), len);
    debug_assert_eq!(x3.len(), len);
    let vm = [
        _mm256_set1_ps(m[0]),
        _mm256_set1_ps(m[1]),
        _mm256_set1_ps(m[2]),
        _mm256_set1_ps(m[3]),
    ];
    let mut sum_acc = [_mm256_setzero_ps(); 4];
    let mut i = 0usize;
    while i + 8 <= len {
        let r0 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x0.as_ptr().add(i)), vm[0]));
        _mm256_storeu_ps(x0.as_mut_ptr().add(i), r0);
        sum_acc[0] = _mm256_add_ps(sum_acc[0], r0);
        let r1 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x1.as_ptr().add(i)), vm[1]));
        _mm256_storeu_ps(x1.as_mut_ptr().add(i), r1);
        sum_acc[1] = _mm256_add_ps(sum_acc[1], r1);
        let r2 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x2.as_ptr().add(i)), vm[2]));
        _mm256_storeu_ps(x2.as_mut_ptr().add(i), r2);
        sum_acc[2] = _mm256_add_ps(sum_acc[2], r2);
        let r3 = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(x3.as_ptr().add(i)), vm[3]));
        _mm256_storeu_ps(x3.as_mut_ptr().add(i), r3);
        sum_acc[3] = _mm256_add_ps(sum_acc[3], r3);
        i += 8;
    }
    let mut sum: [f32; 4] = std::array::from_fn(|r| hsum256_ps(sum_acc[r]));
    let rows: [&mut [f32]; 4] = [x0, x1, x2, x3];
    while i < len {
        for r in 0..4 {
            let e = (rows[r][i] - m[r]).exp();
            rows[r][i] = e;
            sum[r] += e;
        }
        i += 1;
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0usize;
    while i + 8 <= len {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i));
        let s = _mm256_loadu_ps(src.as_ptr().add(i));
        let r = _mm256_fmadd_ps(s, vscale, d);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < len {
        dst[i] += src[i] * scale;
        i += 1;
    }
}

/// PV accumulation for a 4-row group against a whole KV tile, `d_head`-chunk
/// outer / V-row inner: each chunk's accumulator registers stay resident
/// across the entire `bc` sweep and are written back to `acc` only once per
/// chunk, instead of once per V-row — see [`crate::kernel::Kernel::pv4`].
///
/// Processes 2 lanes-of-8 (16 lanes, 8 independent accumulator chains) per
/// outer step rather than 1 (4 chains): see `neon::pv4_neon`'s docs for why
/// 4 chains isn't enough concurrent independent work to hide the FMA
/// latency of each row's `bc`-long sequential accumulation. A single-chunk
/// (4-chain) fallback handles one leftover lane-of-8, then a scalar tail.
#[target_feature(enable = "avx2,fma")]
unsafe fn pv4_avx2(acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
    let [a0, a1, a2, a3] = acc;
    let d = a0.len();
    debug_assert_eq!(a1.len(), d);
    debug_assert_eq!(a2.len(), d);
    debug_assert_eq!(a3.len(), d);
    let bc = p[0].len();
    debug_assert_eq!(v_block.len(), bc * d);
    debug_assert_eq!(p[1].len(), bc);
    debug_assert_eq!(p[2].len(), bc);
    debug_assert_eq!(p[3].len(), bc);

    let mut chunk = 0usize;
    while chunk + 16 <= d {
        let mut acc0a = _mm256_loadu_ps(a0.as_ptr().add(chunk));
        let mut acc0b = _mm256_loadu_ps(a0.as_ptr().add(chunk + 8));
        let mut acc1a = _mm256_loadu_ps(a1.as_ptr().add(chunk));
        let mut acc1b = _mm256_loadu_ps(a1.as_ptr().add(chunk + 8));
        let mut acc2a = _mm256_loadu_ps(a2.as_ptr().add(chunk));
        let mut acc2b = _mm256_loadu_ps(a2.as_ptr().add(chunk + 8));
        let mut acc3a = _mm256_loadu_ps(a3.as_ptr().add(chunk));
        let mut acc3b = _mm256_loadu_ps(a3.as_ptr().add(chunk + 8));
        let mut j = 0usize;
        while j < bc {
            let vva = _mm256_loadu_ps(v_block.as_ptr().add(j * d + chunk));
            let vvb = _mm256_loadu_ps(v_block.as_ptr().add(j * d + chunk + 8));
            let s0 = _mm256_set1_ps(p[0][j]);
            let s1 = _mm256_set1_ps(p[1][j]);
            let s2 = _mm256_set1_ps(p[2][j]);
            let s3 = _mm256_set1_ps(p[3][j]);
            acc0a = _mm256_fmadd_ps(vva, s0, acc0a);
            acc0b = _mm256_fmadd_ps(vvb, s0, acc0b);
            acc1a = _mm256_fmadd_ps(vva, s1, acc1a);
            acc1b = _mm256_fmadd_ps(vvb, s1, acc1b);
            acc2a = _mm256_fmadd_ps(vva, s2, acc2a);
            acc2b = _mm256_fmadd_ps(vvb, s2, acc2b);
            acc3a = _mm256_fmadd_ps(vva, s3, acc3a);
            acc3b = _mm256_fmadd_ps(vvb, s3, acc3b);
            j += 1;
        }
        _mm256_storeu_ps(a0.as_mut_ptr().add(chunk), acc0a);
        _mm256_storeu_ps(a0.as_mut_ptr().add(chunk + 8), acc0b);
        _mm256_storeu_ps(a1.as_mut_ptr().add(chunk), acc1a);
        _mm256_storeu_ps(a1.as_mut_ptr().add(chunk + 8), acc1b);
        _mm256_storeu_ps(a2.as_mut_ptr().add(chunk), acc2a);
        _mm256_storeu_ps(a2.as_mut_ptr().add(chunk + 8), acc2b);
        _mm256_storeu_ps(a3.as_mut_ptr().add(chunk), acc3a);
        _mm256_storeu_ps(a3.as_mut_ptr().add(chunk + 8), acc3b);
        chunk += 16;
    }
    if chunk + 8 <= d {
        let mut acc0 = _mm256_loadu_ps(a0.as_ptr().add(chunk));
        let mut acc1 = _mm256_loadu_ps(a1.as_ptr().add(chunk));
        let mut acc2 = _mm256_loadu_ps(a2.as_ptr().add(chunk));
        let mut acc3 = _mm256_loadu_ps(a3.as_ptr().add(chunk));
        let mut j = 0usize;
        while j < bc {
            let vv = _mm256_loadu_ps(v_block.as_ptr().add(j * d + chunk));
            acc0 = _mm256_fmadd_ps(vv, _mm256_set1_ps(p[0][j]), acc0);
            acc1 = _mm256_fmadd_ps(vv, _mm256_set1_ps(p[1][j]), acc1);
            acc2 = _mm256_fmadd_ps(vv, _mm256_set1_ps(p[2][j]), acc2);
            acc3 = _mm256_fmadd_ps(vv, _mm256_set1_ps(p[3][j]), acc3);
            j += 1;
        }
        _mm256_storeu_ps(a0.as_mut_ptr().add(chunk), acc0);
        _mm256_storeu_ps(a1.as_mut_ptr().add(chunk), acc1);
        _mm256_storeu_ps(a2.as_mut_ptr().add(chunk), acc2);
        _mm256_storeu_ps(a3.as_mut_ptr().add(chunk), acc3);
        chunk += 8;
    }
    while chunk < d {
        let (mut s0, mut s1, mut s2, mut s3) = (a0[chunk], a1[chunk], a2[chunk], a3[chunk]);
        let mut j = 0usize;
        while j < bc {
            let vv = v_block[j * d + chunk];
            s0 += vv * p[0][j];
            s1 += vv * p[1][j];
            s2 += vv * p[2][j];
            s3 += vv * p[3][j];
            j += 1;
        }
        a0[chunk] = s0;
        a1[chunk] = s1;
        a2[chunk] = s2;
        a3[chunk] = s3;
        chunk += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn scale_avx2(dst: &mut [f32], scale: f32) {
    let len = dst.len();
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0usize;
    while i + 8 <= len {
        let d = _mm256_loadu_ps(dst.as_ptr().add(i));
        let r = _mm256_mul_ps(d, vscale);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < len {
        dst[i] *= scale;
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn max_reduce_avx2(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }
    let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        acc = _mm256_max_ps(acc, v);
        i += 8;
    }
    let mut m = hmax256_ps(acc);
    while i < len {
        m = m.max(x[i]);
        i += 1;
    }
    m
}

/// [`max_reduce_avx2`], 4 rows at once, interleaved into 4 independent
/// chains — see [`crate::kernel::Kernel::max_reduce4`] for why.
#[target_feature(enable = "avx2,fma")]
unsafe fn max_reduce4_avx2(x: [&[f32]; 4]) -> [f32; 4] {
    let len = x[0].len();
    debug_assert_eq!(x[1].len(), len);
    debug_assert_eq!(x[2].len(), len);
    debug_assert_eq!(x[3].len(), len);
    if len == 0 {
        return [f32::NEG_INFINITY; 4];
    }
    let mut acc = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
    let mut i = 0usize;
    while i + 8 <= len {
        for r in 0..4 {
            acc[r] = _mm256_max_ps(acc[r], _mm256_loadu_ps(x[r].as_ptr().add(i)));
        }
        i += 8;
    }
    let mut m: [f32; 4] = std::array::from_fn(|r| hmax256_ps(acc[r]));
    while i < len {
        for r in 0..4 {
            m[r] = m[r].max(x[r][i]);
        }
        i += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avx2_available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    #[test]
    fn exp_matches_std() {
        if !avx2_available() {
            return;
        }
        let xs: Vec<f32> = (-800..800).map(|i| i as f32 * 0.1).collect();
        let mut got = xs.clone();
        let sum = unsafe { sub_exp_sum_inplace_avx2(&mut got, 0.0) };
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
        let want_sum: f32 = got.iter().sum();
        assert!(
            (sum - want_sum).abs() < 1e-3 * want_sum.abs().max(1.0),
            "returned sum {sum} vs actual {want_sum}"
        );
        eprintln!("max relative error vs f32::exp: {max_rel_err:e}");
    }

    #[test]
    fn dot_matches_scalar() {
        if !avx2_available() {
            return;
        }
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.71).cos()).collect();
            let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let simd = unsafe { dot_avx2(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3 * (scalar.abs() + 1.0),
                "len={len} scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn dot4_matches_four_dots() {
        if !avx2_available() {
            return;
        }
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk =
                |seed: f32| -> Vec<f32> { (0..len).map(|i| (i as f32 * seed).sin()).collect() };
            let a0 = mk(0.11);
            let a1 = mk(0.23);
            let a2 = mk(0.37);
            let a3 = mk(0.51);
            let b = mk(0.71);

            let want = [
                unsafe { dot_avx2(&a0, &b) },
                unsafe { dot_avx2(&a1, &b) },
                unsafe { dot_avx2(&a2, &b) },
                unsafe { dot_avx2(&a3, &b) },
            ];
            let got = unsafe { dot4_avx2(&a0, &a1, &a2, &a3, &b) };
            for k in 0..4 {
                assert!(
                    (want[k] - got[k]).abs() < 1e-3 * (want[k].abs() + 1.0),
                    "len={len} k={k} want={} got={}",
                    want[k],
                    got[k]
                );
            }
        }
    }

    #[test]
    fn dot4x4_matches_naive() {
        if !avx2_available() {
            return;
        }
        for d in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk = |seed: f32| -> Vec<f32> { (0..d).map(|i| (i as f32 * seed).sin()).collect() };
            let q = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let k = [mk(0.61), mk(0.67), mk(0.73), mk(0.79)];

            let want: [[f32; 4]; 4] =
                std::array::from_fn(|r| std::array::from_fn(|c| unsafe { dot_avx2(&q[r], &k[c]) }));
            let got =
                unsafe { dot4x4_avx2([&q[0], &q[1], &q[2], &q[3]], [&k[0], &k[1], &k[2], &k[3]]) };
            for r in 0..4 {
                for c in 0..4 {
                    assert!(
                        (want[r][c] - got[r][c]).abs() < 1e-3 * (want[r][c].abs() + 1.0),
                        "d={d} r={r} c={c} want={} got={}",
                        want[r][c],
                        got[r][c]
                    );
                }
            }
        }
    }

    #[test]
    fn pv4_matches_naive() {
        if !avx2_available() {
            return;
        }
        for (bc, d) in [
            (0usize, 4usize),
            (1, 4),
            (3, 5),
            (4, 4),
            (5, 7),
            (17, 33),
            (9, 27), // exercises all 3 internal tiers: 16-chunk + 8-chunk + scalar tail
        ] {
            let v_block: Vec<f32> = (0..bc * d).map(|i| (i as f32 * 0.03).cos()).collect();
            let p: [Vec<f32>; 4] = std::array::from_fn(|r| {
                (0..bc)
                    .map(|j| ((j + r) as f32 * 0.07).sin())
                    .collect::<Vec<f32>>()
            });
            let init: [Vec<f32>; 4] =
                std::array::from_fn(|r| (0..d).map(|i| (i as f32 + r as f32) * 0.5).collect());

            let mut want = init.clone();
            for (row, pr) in want.iter_mut().zip(p.iter()) {
                for (j, v_row) in v_block.chunks_exact(d).enumerate() {
                    unsafe { axpy_avx2(row, v_row, pr[j]) };
                }
            }

            let mut got = init;
            let [g0, g1, g2, g3] = &mut got;
            unsafe { pv4_avx2([g0, g1, g2, g3], &v_block, [&p[0], &p[1], &p[2], &p[3]]) };

            for r in 0..4 {
                for i in 0..d {
                    assert!(
                        (want[r][i] - got[r][i]).abs() < 1e-3 * (want[r][i].abs() + 1.0),
                        "bc={bc} d={d} r={r} i={i} want={} got={}",
                        want[r][i],
                        got[r][i]
                    );
                }
            }
        }
    }

    #[test]
    fn reductions_match_scalar() {
        if !avx2_available() {
            return;
        }
        for len in [0usize, 1, 5, 8, 13, 16, 33] {
            let x: Vec<f32> = (0..len).map(|i| ((i as f32) * 1.3).sin() * 5.0).collect();
            let want_max: f32 = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let got_max = unsafe { max_reduce_avx2(&x) };
            assert_eq!(want_max, got_max, "len={len} max");
        }
    }

    #[test]
    fn max_reduce4_matches_four_max_reduces() {
        if !avx2_available() {
            return;
        }
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk =
                |seed: f32| -> Vec<f32> { (0..len).map(|i| (i as f32 * seed).sin()).collect() };
            let rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let want = [
                unsafe { max_reduce_avx2(&rows[0]) },
                unsafe { max_reduce_avx2(&rows[1]) },
                unsafe { max_reduce_avx2(&rows[2]) },
                unsafe { max_reduce_avx2(&rows[3]) },
            ];
            let got = unsafe { max_reduce4_avx2([&rows[0], &rows[1], &rows[2], &rows[3]]) };
            assert_eq!(want, got, "len={len}");
        }
    }

    #[test]
    fn sub_exp_sum_inplace4_matches_four_calls() {
        if !avx2_available() {
            return;
        }
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk = |seed: f32| -> Vec<f32> {
                (0..len).map(|i| (i as f32 * seed).sin() * 4.0).collect()
            };
            let m = [0.1f32, -0.3, 0.5, 0.0];

            let mut want_rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let want_sums: [f32; 4] = std::array::from_fn(|r| unsafe {
                sub_exp_sum_inplace_avx2(&mut want_rows[r], m[r])
            });

            let mut got_rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let [g0, g1, g2, g3] = &mut got_rows;
            let got_sums = unsafe { sub_exp_sum_inplace4_avx2([g0, g1, g2, g3], m) };

            for r in 0..4 {
                assert!(
                    (want_sums[r] - got_sums[r]).abs() < 1e-3 * (want_sums[r].abs() + 1.0),
                    "len={len} r={r} sum want={} got={}",
                    want_sums[r],
                    got_sums[r]
                );
                for i in 0..len {
                    assert!(
                        (want_rows[r][i] - got_rows[r][i]).abs() < 1e-4,
                        "len={len} r={r} i={i} want={} got={}",
                        want_rows[r][i],
                        got_rows[r][i]
                    );
                }
            }
        }
    }

    #[test]
    fn axpy_and_scale_match_scalar() {
        if !avx2_available() {
            return;
        }
        for len in [0usize, 1, 7, 8, 9, 31, 32, 33] {
            let mut dst: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let src: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();
            let scale = 1.37f32;
            let mut want = dst.clone();
            for (d, s) in want.iter_mut().zip(src.iter()) {
                *d += s * scale;
            }
            unsafe { axpy_avx2(&mut dst, &src, scale) };
            for (w, g) in want.iter().zip(dst.iter()) {
                assert!((w - g).abs() < 1e-4, "axpy len={len}");
            }

            let mut d2 = want.clone();
            let want2: Vec<f32> = d2.iter().map(|v| v * 2.5).collect();
            unsafe { scale_avx2(&mut d2, 2.5) };
            for (w, g) in want2.iter().zip(d2.iter()) {
                assert!((w - g).abs() < 1e-4, "scale len={len}");
            }
        }
    }
}
