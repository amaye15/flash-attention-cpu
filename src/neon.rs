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
    unsafe fn dot4(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
        dot4_neon(a0, a1, a2, a3, b)
    }

    #[inline]
    unsafe fn dot4x4(q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
        dot4x4_neon(q, k)
    }

    #[inline]
    unsafe fn sub_exp_sum_inplace(x: &mut [f32], m: f32) -> f32 {
        sub_exp_sum_inplace_neon(x, m)
    }

    #[inline]
    unsafe fn sub_exp_sum_inplace4(x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
        sub_exp_sum_inplace4_neon(x, m)
    }

    #[inline]
    unsafe fn axpy(dst: &mut [f32], src: &[f32], scale: f32) {
        axpy_neon(dst, src, scale)
    }

    #[inline]
    unsafe fn pv4(acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
        pv4_neon(acc, v_block, p)
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
    unsafe fn max_reduce4(x: [&[f32]; 4]) -> [f32; 4] {
        max_reduce4_neon(x)
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

/// Four dot products sharing `b`'s vector loads across four independent FMA
/// accumulator chains — see [`crate::kernel::Kernel::dot4`] for why this is
/// faster than four separate [`dot_neon`] calls.
#[target_feature(enable = "neon")]
unsafe fn dot4_neon(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    debug_assert_eq!(a0.len(), b.len());
    debug_assert_eq!(a1.len(), b.len());
    debug_assert_eq!(a2.len(), b.len());
    debug_assert_eq!(a3.len(), b.len());
    let len = b.len();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        let bv = vld1q_f32(b.as_ptr().add(i)); // loaded once, shared 4 ways
        acc0 = vfmaq_f32(acc0, vld1q_f32(a0.as_ptr().add(i)), bv);
        acc1 = vfmaq_f32(acc1, vld1q_f32(a1.as_ptr().add(i)), bv);
        acc2 = vfmaq_f32(acc2, vld1q_f32(a2.as_ptr().add(i)), bv);
        acc3 = vfmaq_f32(acc3, vld1q_f32(a3.as_ptr().add(i)), bv);
        i += 4;
    }
    let mut sums = [
        vaddvq_f32(acc0),
        vaddvq_f32(acc1),
        vaddvq_f32(acc2),
        vaddvq_f32(acc3),
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

/// 4 query rows x 4 key rows blocked together: both operands' vector loads
/// are shared across all 16 independent FMA accumulator chains — see
/// [`crate::kernel::Kernel::dot4x4`].
#[target_feature(enable = "neon")]
unsafe fn dot4x4_neon(q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
    let d = q[0].len();
    let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
    let mut p = 0usize;
    while p + 4 <= d {
        let qv = [
            vld1q_f32(q[0].as_ptr().add(p)),
            vld1q_f32(q[1].as_ptr().add(p)),
            vld1q_f32(q[2].as_ptr().add(p)),
            vld1q_f32(q[3].as_ptr().add(p)),
        ];
        for c in 0..4 {
            let kv = vld1q_f32(k[c].as_ptr().add(p)); // loaded once, shared 4 ways
            for r in 0..4 {
                acc[r][c] = vfmaq_f32(acc[r][c], qv[r], kv);
            }
        }
        p += 4;
    }
    let mut sums = [[0.0f32; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            sums[r][c] = vaddvq_f32(acc[r][c]);
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

/// Fused `x[i] = exp(x[i] - m)`, returning `sum(x)` after the exponential:
/// subtract, exponential, and sum accumulation all in the same pass over
/// `x` (one load/store per lane, plus the sum, instead of two separate
/// passes).
#[target_feature(enable = "neon")]
unsafe fn sub_exp_sum_inplace_neon(x: &mut [f32], m: f32) -> f32 {
    let len = x.len();
    let vm = vdupq_n_f32(m);
    let mut sum_acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = vld1q_f32(x.as_ptr().add(i));
        let v = vsubq_f32(v, vm);
        let r = exp128_ps(v);
        vst1q_f32(x.as_mut_ptr().add(i), r);
        sum_acc = vaddq_f32(sum_acc, r);
        i += 4;
    }
    let mut sum = vaddvq_f32(sum_acc);
    while i < len {
        let e = (x[i] - m).exp();
        x[i] = e;
        sum += e;
        i += 1;
    }
    sum
}

/// [`sub_exp_sum_inplace_neon`], 4 rows at once with per-row `m` values,
/// interleaved into 4 independent chains — see
/// [`crate::kernel::Kernel::sub_exp_sum_inplace4`] for why.
#[target_feature(enable = "neon")]
unsafe fn sub_exp_sum_inplace4_neon(x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
    let [x0, x1, x2, x3] = x;
    let len = x0.len();
    debug_assert_eq!(x1.len(), len);
    debug_assert_eq!(x2.len(), len);
    debug_assert_eq!(x3.len(), len);
    let vm = [
        vdupq_n_f32(m[0]),
        vdupq_n_f32(m[1]),
        vdupq_n_f32(m[2]),
        vdupq_n_f32(m[3]),
    ];
    let mut sum_acc = [vdupq_n_f32(0.0); 4];
    let mut i = 0usize;
    while i + 4 <= len {
        let r0 = exp128_ps(vsubq_f32(vld1q_f32(x0.as_ptr().add(i)), vm[0]));
        vst1q_f32(x0.as_mut_ptr().add(i), r0);
        sum_acc[0] = vaddq_f32(sum_acc[0], r0);
        let r1 = exp128_ps(vsubq_f32(vld1q_f32(x1.as_ptr().add(i)), vm[1]));
        vst1q_f32(x1.as_mut_ptr().add(i), r1);
        sum_acc[1] = vaddq_f32(sum_acc[1], r1);
        let r2 = exp128_ps(vsubq_f32(vld1q_f32(x2.as_ptr().add(i)), vm[2]));
        vst1q_f32(x2.as_mut_ptr().add(i), r2);
        sum_acc[2] = vaddq_f32(sum_acc[2], r2);
        let r3 = exp128_ps(vsubq_f32(vld1q_f32(x3.as_ptr().add(i)), vm[3]));
        vst1q_f32(x3.as_mut_ptr().add(i), r3);
        sum_acc[3] = vaddq_f32(sum_acc[3], r3);
        i += 4;
    }
    let mut sum: [f32; 4] = std::array::from_fn(|r| vaddvq_f32(sum_acc[r]));
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

/// PV accumulation for a 4-row group against a whole KV tile, `d_head`-chunk
/// outer / V-row inner: each chunk's accumulator registers stay resident
/// across the entire `bc` sweep and are written back to `acc` only once per
/// chunk, instead of once per V-row — see [`crate::kernel::Kernel::pv4`].
///
/// Processes 2 lanes-of-4 (8 lanes, 8 independent accumulator chains) per
/// outer step rather than 1 (4 chains): each row's accumulator chain has a
/// genuine sequential FMA dependency across the whole `bc` sweep, and 4
/// chains isn't enough concurrent independent work to hide FMA latency —
/// doubling to 8 chains (same total FLOPs) measured ~1.8-2.3x over the
/// 4-chain version in isolation. A single-chunk (4-chain) fallback handles
/// one leftover lane-of-4, then a scalar tail for the final `d % 4`.
#[target_feature(enable = "neon")]
unsafe fn pv4_neon(acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
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
    while chunk + 8 <= d {
        let mut acc0a = vld1q_f32(a0.as_ptr().add(chunk));
        let mut acc0b = vld1q_f32(a0.as_ptr().add(chunk + 4));
        let mut acc1a = vld1q_f32(a1.as_ptr().add(chunk));
        let mut acc1b = vld1q_f32(a1.as_ptr().add(chunk + 4));
        let mut acc2a = vld1q_f32(a2.as_ptr().add(chunk));
        let mut acc2b = vld1q_f32(a2.as_ptr().add(chunk + 4));
        let mut acc3a = vld1q_f32(a3.as_ptr().add(chunk));
        let mut acc3b = vld1q_f32(a3.as_ptr().add(chunk + 4));
        let mut j = 0usize;
        while j < bc {
            let vva = vld1q_f32(v_block.as_ptr().add(j * d + chunk));
            let vvb = vld1q_f32(v_block.as_ptr().add(j * d + chunk + 4));
            let s0 = vdupq_n_f32(p[0][j]);
            let s1 = vdupq_n_f32(p[1][j]);
            let s2 = vdupq_n_f32(p[2][j]);
            let s3 = vdupq_n_f32(p[3][j]);
            acc0a = vfmaq_f32(acc0a, vva, s0);
            acc0b = vfmaq_f32(acc0b, vvb, s0);
            acc1a = vfmaq_f32(acc1a, vva, s1);
            acc1b = vfmaq_f32(acc1b, vvb, s1);
            acc2a = vfmaq_f32(acc2a, vva, s2);
            acc2b = vfmaq_f32(acc2b, vvb, s2);
            acc3a = vfmaq_f32(acc3a, vva, s3);
            acc3b = vfmaq_f32(acc3b, vvb, s3);
            j += 1;
        }
        vst1q_f32(a0.as_mut_ptr().add(chunk), acc0a);
        vst1q_f32(a0.as_mut_ptr().add(chunk + 4), acc0b);
        vst1q_f32(a1.as_mut_ptr().add(chunk), acc1a);
        vst1q_f32(a1.as_mut_ptr().add(chunk + 4), acc1b);
        vst1q_f32(a2.as_mut_ptr().add(chunk), acc2a);
        vst1q_f32(a2.as_mut_ptr().add(chunk + 4), acc2b);
        vst1q_f32(a3.as_mut_ptr().add(chunk), acc3a);
        vst1q_f32(a3.as_mut_ptr().add(chunk + 4), acc3b);
        chunk += 8;
    }
    if chunk + 4 <= d {
        let mut acc0 = vld1q_f32(a0.as_ptr().add(chunk));
        let mut acc1 = vld1q_f32(a1.as_ptr().add(chunk));
        let mut acc2 = vld1q_f32(a2.as_ptr().add(chunk));
        let mut acc3 = vld1q_f32(a3.as_ptr().add(chunk));
        let mut j = 0usize;
        while j < bc {
            let vv = vld1q_f32(v_block.as_ptr().add(j * d + chunk));
            acc0 = vfmaq_f32(acc0, vv, vdupq_n_f32(p[0][j]));
            acc1 = vfmaq_f32(acc1, vv, vdupq_n_f32(p[1][j]));
            acc2 = vfmaq_f32(acc2, vv, vdupq_n_f32(p[2][j]));
            acc3 = vfmaq_f32(acc3, vv, vdupq_n_f32(p[3][j]));
            j += 1;
        }
        vst1q_f32(a0.as_mut_ptr().add(chunk), acc0);
        vst1q_f32(a1.as_mut_ptr().add(chunk), acc1);
        vst1q_f32(a2.as_mut_ptr().add(chunk), acc2);
        vst1q_f32(a3.as_mut_ptr().add(chunk), acc3);
        chunk += 4;
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

/// [`max_reduce_neon`], 4 rows at once, interleaved into 4 independent
/// chains — see [`crate::kernel::Kernel::max_reduce4`] for why.
#[target_feature(enable = "neon")]
unsafe fn max_reduce4_neon(x: [&[f32]; 4]) -> [f32; 4] {
    let len = x[0].len();
    debug_assert_eq!(x[1].len(), len);
    debug_assert_eq!(x[2].len(), len);
    debug_assert_eq!(x[3].len(), len);
    if len == 0 {
        return [f32::NEG_INFINITY; 4];
    }
    let mut acc = [vdupq_n_f32(f32::NEG_INFINITY); 4];
    let mut i = 0usize;
    while i + 4 <= len {
        for r in 0..4 {
            acc[r] = vmaxq_f32(acc[r], vld1q_f32(x[r].as_ptr().add(i)));
        }
        i += 4;
    }
    let mut m: [f32; 4] = std::array::from_fn(|r| vmaxvq_f32(acc[r]));
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

    #[test]
    fn exp_matches_std() {
        let xs: Vec<f32> = (-800..800).map(|i| i as f32 * 0.1).collect();
        let mut got = xs.clone();
        let sum = unsafe { sub_exp_sum_inplace_neon(&mut got, 0.0) };
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
    fn dot4_matches_four_dots() {
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk =
                |seed: f32| -> Vec<f32> { (0..len).map(|i| (i as f32 * seed).sin()).collect() };
            let a0 = mk(0.11);
            let a1 = mk(0.23);
            let a2 = mk(0.37);
            let a3 = mk(0.51);
            let b = mk(0.71);

            let want = [
                unsafe { dot_neon(&a0, &b) },
                unsafe { dot_neon(&a1, &b) },
                unsafe { dot_neon(&a2, &b) },
                unsafe { dot_neon(&a3, &b) },
            ];
            let got = unsafe { dot4_neon(&a0, &a1, &a2, &a3, &b) };
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
        for d in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk = |seed: f32| -> Vec<f32> { (0..d).map(|i| (i as f32 * seed).sin()).collect() };
            let q = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let k = [mk(0.61), mk(0.67), mk(0.73), mk(0.79)];

            let want: [[f32; 4]; 4] =
                std::array::from_fn(|r| std::array::from_fn(|c| unsafe { dot_neon(&q[r], &k[c]) }));
            let got =
                unsafe { dot4x4_neon([&q[0], &q[1], &q[2], &q[3]], [&k[0], &k[1], &k[2], &k[3]]) };
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
        for (bc, d) in [
            (0usize, 4usize),
            (1, 4),
            (3, 5),
            (4, 4),
            (5, 7),
            (17, 33),
            (9, 13), // exercises all 3 internal tiers: 8-chunk + 4-chunk + scalar tail
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
                    unsafe { axpy_neon(row, v_row, pr[j]) };
                }
            }

            let mut got = init;
            let [g0, g1, g2, g3] = &mut got;
            unsafe { pv4_neon([g0, g1, g2, g3], &v_block, [&p[0], &p[1], &p[2], &p[3]]) };

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
        for len in [0usize, 1, 5, 8, 13, 16, 33] {
            let x: Vec<f32> = (0..len).map(|i| ((i as f32) * 1.3).sin() * 5.0).collect();
            let want_max: f32 = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let got_max = unsafe { max_reduce_neon(&x) };
            assert_eq!(want_max, got_max, "len={len} max");
        }
    }

    #[test]
    fn max_reduce4_matches_four_max_reduces() {
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk =
                |seed: f32| -> Vec<f32> { (0..len).map(|i| (i as f32 * seed).sin()).collect() };
            let rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let want = [
                unsafe { max_reduce_neon(&rows[0]) },
                unsafe { max_reduce_neon(&rows[1]) },
                unsafe { max_reduce_neon(&rows[2]) },
                unsafe { max_reduce_neon(&rows[3]) },
            ];
            let got = unsafe { max_reduce4_neon([&rows[0], &rows[1], &rows[2], &rows[3]]) };
            assert_eq!(want, got, "len={len}");
        }
    }

    #[test]
    fn sub_exp_sum_inplace4_matches_four_calls() {
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk = |seed: f32| -> Vec<f32> {
                (0..len).map(|i| (i as f32 * seed).sin() * 4.0).collect()
            };
            let m = [0.1f32, -0.3, 0.5, 0.0];

            let mut want_rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let want_sums: [f32; 4] = std::array::from_fn(|r| unsafe {
                sub_exp_sum_inplace_neon(&mut want_rows[r], m[r])
            });

            let mut got_rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let [g0, g1, g2, g3] = &mut got_rows;
            let got_sums = unsafe { sub_exp_sum_inplace4_neon([g0, g1, g2, g3], m) };

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
