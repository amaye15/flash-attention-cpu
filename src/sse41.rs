//! SSE4.1 accelerated kernel (x86_64 only) — the fallback tier between
//! AVX2+FMA and the portable scalar kernel, for x86_64 CPUs that lack AVX2.
//!
//! That's a real, current population, not just old hardware: VMware EVC and
//! Hyper-V processor compatibility mode both deliberately mask AVX2 (and
//! AVX-512) on every VM in a cluster to allow live migration across mixed
//! CPU generations — a common, deliberately-chosen enterprise configuration,
//! independent of what the physical host actually supports. Budget/embedded
//! x86_64 (e.g. Intel Gemini Lake-era Atom chips in fanless mini-PCs and
//! gateways) has SSE4.2 but no AVX2. And "SSE4.1, no AVX2" is exactly the
//! `x86-64-v2` microarchitecture floor RHEL and Anaconda/conda-forge are
//! standardizing distro/platform baselines on for 2026 — see
//! [ROADMAP.md](https://github.com/amaye15/flash-attention-cpu/blob/main/ROADMAP.md#9-x86_64-sse41-baseline-tier-new-item--hardware-coverage-gap-not-a-new-architecture)
//! for the research behind adding this tier. Before this, any x86_64 CPU
//! without AVX2 fell straight through to the scalar fallback.
//!
//! Checked via `is_x86_feature_detected!("sse4.1")` in each variant's entry
//! point, the same runtime-dispatch pattern as AVX-512F/AVX2 — this compiles
//! to a portable binary that still uses SSE4.1 wherever the host actually
//! supports it (which is effectively every x86_64 CPU built since ~2008;
//! SSE2 itself is unconditionally mandatory on x86_64, but SSE4.1 adds the
//! packed `floor` instruction the `exp` implementation below needs).
//!
//! **No native fused-multiply-add.** FMA3 is a separate x86 extension that
//! arrived alongside AVX2 (Haswell on Intel; Piledriver/Steamroller on AMD
//! predate AVX2 but did have FMA3 — a narrow historical combination this
//! kernel doesn't specifically chase). So, like `simd128.rs`'s baseline
//! (pre-`relaxed-simd`) path, every accumulation here is a separate multiply
//! then add: two roundings instead of one, composed inline rather than
//! through a helper, since (unlike `simd128.rs`) there's no conditional
//! fused path to consolidate here.
//!
//! `exp128_ps` is the same algorithm as `avx2::exp256_ps`/`neon::exp128_ps`/
//! `simd128::exp128_ps` (range-reduce, degree-5 minimax polynomial, direct
//! IEEE-754 exponent-bit reconstruction of `2^n`), over 4 lanes — the same
//! lane width as NEON/SIMD128, so this file mirrors their tiling/unroll
//! shape (2-lane-of-4 = 8-element chunks in `pv4`) rather than AVX2's
//! 2-lane-of-8 = 16-element chunks.

use crate::kernel::Kernel;
use std::arch::x86_64::*;

pub(crate) struct Sse41Kernel;

impl Sse41Kernel {
    /// Returns `Some` only if the running CPU actually supports SSE4.1 —
    /// checked once, here. This is the *only* way to obtain an
    /// `Sse41Kernel`, which is what makes every `Kernel` method on it safe:
    /// simply possessing one is proof this check already passed.
    pub(crate) fn new() -> Option<Self> {
        if is_x86_feature_detected!("sse4.1") {
            Some(Self)
        } else {
            None
        }
    }
}

impl Kernel for Sse41Kernel {
    #[inline]
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { dot_sse41(a, b) }
    }

    #[inline]
    fn dot4(&self, a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { dot4_sse41(a0, a1, a2, a3, b) }
    }

    #[inline]
    fn dot4x4(&self, q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { dot4x4_sse41(q, k) }
    }

    #[inline]
    fn sub_exp_sum_inplace(&self, x: &mut [f32], m: f32) -> f32 {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { sub_exp_sum_inplace_sse41(x, m) }
    }

    #[inline]
    fn sub_exp_sum_inplace4(&self, x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { sub_exp_sum_inplace4_sse41(x, m) }
    }

    #[inline]
    fn axpy(&self, dst: &mut [f32], src: &[f32], scale: f32) {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { axpy_sse41(dst, src, scale) }
    }

    #[inline]
    fn pv4(&self, acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { pv4_sse41(acc, v_block, p) }
    }

    #[inline]
    fn scale_inplace(&self, dst: &mut [f32], scale: f32) {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { scale_sse41(dst, scale) }
    }

    #[inline]
    fn max_reduce(&self, x: &[f32]) -> f32 {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { max_reduce_sse41(x) }
    }

    #[inline]
    fn max_reduce4(&self, x: [&[f32]; 4]) -> [f32; 4] {
        // SAFETY: `Self` is only constructible via `Sse41Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { max_reduce4_sse41(x) }
    }
}

#[inline]
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn hsum128_ps(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let sums = _mm_add_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[inline]
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn hmax128_ps(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v);
    let maxs = _mm_max_ps(v, shuf);
    let shuf2 = _mm_movehl_ps(shuf, maxs);
    let result = _mm_max_ss(maxs, shuf2);
    _mm_cvtss_f32(result)
}

/// Dot product, 2-way accumulator unrolled (8 f32 / iteration) — same idea
/// as `avx2::dot_avx2`/`neon::dot_neon`/`simd128::dot_simd128`, just without
/// a fused multiply-add (see module docs), so each step is a separate
/// `mul` + `add`.
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn dot_sse41(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: each load reads a fixed-width window starting at `i`; the `i + 8 <= len` guards above ensure every load stays within `a`/`b` (asserted equal length above).
    unsafe {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= len {
            let a0 = _mm_loadu_ps(a.as_ptr().add(i));
            let b0 = _mm_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(a0, b0));
            let a1 = _mm_loadu_ps(a.as_ptr().add(i + 4));
            let b1 = _mm_loadu_ps(b.as_ptr().add(i + 4));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(a1, b1));
            i += 8;
        }
        while i + 4 <= len {
            let av = _mm_loadu_ps(a.as_ptr().add(i));
            let bv = _mm_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(av, bv));
            i += 4;
        }
        let mut sum = hsum128_ps(_mm_add_ps(acc0, acc1));
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// Four dot products sharing `b`'s vector loads across four independent
/// mul+add accumulator chains — see [`crate::kernel::Kernel::dot4`] for why
/// this is faster than four separate [`dot_sse41`] calls.
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn dot4_sse41(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the shared `b` load and all four `a0..a3` loads each iteration (all four asserted equal length to `b` above).
    unsafe {
        debug_assert_eq!(a0.len(), b.len());
        debug_assert_eq!(a1.len(), b.len());
        debug_assert_eq!(a2.len(), b.len());
        debug_assert_eq!(a3.len(), b.len());
        let len = b.len();
        let mut acc0 = _mm_setzero_ps();
        let mut acc1 = _mm_setzero_ps();
        let mut acc2 = _mm_setzero_ps();
        let mut acc3 = _mm_setzero_ps();
        let mut i = 0usize;
        while i + 4 <= len {
            let bv = _mm_loadu_ps(b.as_ptr().add(i)); // loaded once, shared 4 ways
            acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(a0.as_ptr().add(i)), bv));
            acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(a1.as_ptr().add(i)), bv));
            acc2 = _mm_add_ps(acc2, _mm_mul_ps(_mm_loadu_ps(a2.as_ptr().add(i)), bv));
            acc3 = _mm_add_ps(acc3, _mm_mul_ps(_mm_loadu_ps(a3.as_ptr().add(i)), bv));
            i += 4;
        }
        let mut sums = [
            hsum128_ps(acc0),
            hsum128_ps(acc1),
            hsum128_ps(acc2),
            hsum128_ps(acc3),
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
}

/// 4 query rows x 4 key rows blocked together — see
/// [`crate::kernel::Kernel::dot4x4`]. No native FMA (see module docs), so
/// each accumulation is a separate `mul` + `add`.
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn dot4x4_sse41(q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `p + 4 <= d` guard covers all four `q` and four `k` row loads each iteration.
    unsafe {
        let d = q[0].len();
        let mut acc = [[_mm_setzero_ps(); 4]; 4];
        let mut p = 0usize;
        while p + 4 <= d {
            let qv = [
                _mm_loadu_ps(q[0].as_ptr().add(p)),
                _mm_loadu_ps(q[1].as_ptr().add(p)),
                _mm_loadu_ps(q[2].as_ptr().add(p)),
                _mm_loadu_ps(q[3].as_ptr().add(p)),
            ];
            for c in 0..4 {
                let kv = _mm_loadu_ps(k[c].as_ptr().add(p)); // loaded once, shared 4 ways
                for r in 0..4 {
                    acc[r][c] = _mm_add_ps(acc[r][c], _mm_mul_ps(qv[r], kv));
                }
            }
            p += 4;
        }
        let mut sums = [[0.0f32; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                sums[r][c] = hsum128_ps(acc[r][c]);
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
}

/// Vectorized exp over 4 lanes. See module docs for the algorithm.
///
/// The polynomial coefficients below are the same published Cephes-derived
/// minimax fit `avx2::exp256_ps`/`neon::exp128_ps`/`simd128::exp128_ps` use,
/// kept at full precision as documented there rather than trimmed to
/// placate `clippy::excessive_precision`.
#[target_feature(enable = "sse2,sse3,sse4.1")]
#[allow(clippy::excessive_precision)]
unsafe fn exp128_ps(x: __m128) -> __m128 {
    let exp_hi = _mm_set1_ps(88.376_26_f32);
    let exp_lo = _mm_set1_ps(-88.376_26_f32);
    let log2ef = _mm_set1_ps(std::f32::consts::LOG2_E);
    let half = _mm_set1_ps(0.5_f32);
    let c1 = _mm_set1_ps(0.693_359_375_f32);
    let c2 = _mm_set1_ps(-2.121_944_4e-4_f32);
    let p0 = _mm_set1_ps(1.987_569_15e-4_f32);
    let p1 = _mm_set1_ps(1.398_199_950_7e-3_f32);
    let p2 = _mm_set1_ps(8.333_451_907_3e-3_f32);
    let p3 = _mm_set1_ps(4.166_579_589_4e-2_f32);
    let p4 = _mm_set1_ps(1.666_666_545_9e-1_f32);
    let p5 = _mm_set1_ps(5.000_000_120_1e-1_f32);
    let one = _mm_set1_ps(1.0_f32);

    let x = _mm_min_ps(x, exp_hi);
    let x = _mm_max_ps(x, exp_lo);

    // n = floor(x / ln(2) + 0.5) - _mm_floor_ps needs SSE4.1
    let fx = _mm_add_ps(_mm_mul_ps(x, log2ef), half);
    let fx = _mm_floor_ps(fx);

    // r = x - n*ln(2), split hi/lo for precision
    let tmp = _mm_mul_ps(fx, c1);
    let z = _mm_mul_ps(fx, c2);
    let x = _mm_sub_ps(x, tmp);
    let x = _mm_sub_ps(x, z);

    let z = _mm_mul_ps(x, x);

    // degree-5 minimax polynomial for exp(r)
    let mut y = p0;
    y = _mm_add_ps(_mm_mul_ps(y, x), p1);
    y = _mm_add_ps(_mm_mul_ps(y, x), p2);
    y = _mm_add_ps(_mm_mul_ps(y, x), p3);
    y = _mm_add_ps(_mm_mul_ps(y, x), p4);
    y = _mm_add_ps(_mm_mul_ps(y, x), p5);
    y = _mm_add_ps(_mm_mul_ps(y, z), x);
    y = _mm_add_ps(y, one);

    // 2^n via direct exponent-bit construction
    let imm0 = _mm_cvttps_epi32(fx);
    let imm0 = _mm_add_epi32(imm0, _mm_set1_epi32(0x7f));
    let imm0 = _mm_slli_epi32(imm0, 23);
    let pow2n = _mm_castsi128_ps(imm0);

    _mm_mul_ps(y, pow2n)
}

/// Fused `x[i] = exp(x[i] - m)`, returning `sum(x)` after the exponential:
/// subtract, exponential, and sum accumulation all in the same pass over
/// `x` (one load/store per lane, plus the sum, instead of two separate
/// passes).
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn sub_exp_sum_inplace_sse41(x: &mut [f32], m: f32) -> f32 {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers both the load and the store back to the same index range.
    unsafe {
        let len = x.len();
        let vm = _mm_set1_ps(m);
        let mut sum_acc = _mm_setzero_ps();
        let mut i = 0usize;
        while i + 4 <= len {
            let v = _mm_loadu_ps(x.as_ptr().add(i));
            let v = _mm_sub_ps(v, vm);
            let r = exp128_ps(v);
            _mm_storeu_ps(x.as_mut_ptr().add(i), r);
            sum_acc = _mm_add_ps(sum_acc, r);
            i += 4;
        }
        let mut sum = hsum128_ps(sum_acc);
        while i < len {
            let e = (x[i] - m).exp();
            x[i] = e;
            sum += e;
            i += 1;
        }
        sum
    }
}

/// [`sub_exp_sum_inplace_sse41`], 4 rows at once with per-row `m` values,
/// interleaved into 4 independent chains — see
/// [`crate::kernel::Kernel::sub_exp_sum_inplace4`] for why.
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn sub_exp_sum_inplace4_sse41(x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the load/store for all four rows (asserted equal length above).
    unsafe {
        let [x0, x1, x2, x3] = x;
        let len = x0.len();
        debug_assert_eq!(x1.len(), len);
        debug_assert_eq!(x2.len(), len);
        debug_assert_eq!(x3.len(), len);
        let vm = [
            _mm_set1_ps(m[0]),
            _mm_set1_ps(m[1]),
            _mm_set1_ps(m[2]),
            _mm_set1_ps(m[3]),
        ];
        let mut sum_acc = [_mm_setzero_ps(); 4];
        let mut i = 0usize;
        while i + 4 <= len {
            let r0 = exp128_ps(_mm_sub_ps(_mm_loadu_ps(x0.as_ptr().add(i)), vm[0]));
            _mm_storeu_ps(x0.as_mut_ptr().add(i), r0);
            sum_acc[0] = _mm_add_ps(sum_acc[0], r0);
            let r1 = exp128_ps(_mm_sub_ps(_mm_loadu_ps(x1.as_ptr().add(i)), vm[1]));
            _mm_storeu_ps(x1.as_mut_ptr().add(i), r1);
            sum_acc[1] = _mm_add_ps(sum_acc[1], r1);
            let r2 = exp128_ps(_mm_sub_ps(_mm_loadu_ps(x2.as_ptr().add(i)), vm[2]));
            _mm_storeu_ps(x2.as_mut_ptr().add(i), r2);
            sum_acc[2] = _mm_add_ps(sum_acc[2], r2);
            let r3 = exp128_ps(_mm_sub_ps(_mm_loadu_ps(x3.as_ptr().add(i)), vm[3]));
            _mm_storeu_ps(x3.as_mut_ptr().add(i), r3);
            sum_acc[3] = _mm_add_ps(sum_acc[3], r3);
            i += 4;
        }
        let mut sum: [f32; 4] = std::array::from_fn(|r| hsum128_ps(sum_acc[r]));
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
}

#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn axpy_sse41(dst: &mut [f32], src: &[f32], scale: f32) {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers both the `dst` load/store and the `src` load (asserted equal length above).
    unsafe {
        debug_assert_eq!(dst.len(), src.len());
        let len = dst.len();
        let vscale = _mm_set1_ps(scale);
        let mut i = 0usize;
        while i + 4 <= len {
            let d = _mm_loadu_ps(dst.as_ptr().add(i));
            let s = _mm_loadu_ps(src.as_ptr().add(i));
            let r = _mm_add_ps(d, _mm_mul_ps(s, vscale));
            _mm_storeu_ps(dst.as_mut_ptr().add(i), r);
            i += 4;
        }
        while i < len {
            dst[i] += src[i] * scale;
            i += 1;
        }
    }
}

/// PV accumulation for a 4-row group against a whole KV tile, `d_head`-chunk
/// outer / V-row inner: each chunk's accumulator registers stay resident
/// across the entire `bc` sweep and are written back to `acc` only once per
/// chunk, instead of once per V-row — see [`crate::kernel::Kernel::pv4`].
///
/// Processes 2 lanes-of-4 (8 lanes, 8 independent accumulator chains) per
/// outer step rather than 1 (4 chains): see `neon::pv4_neon`'s docs for why
/// 4 chains isn't enough concurrent independent work to hide the FMA
/// latency of each row's `bc`-long sequential accumulation (no native FMA
/// here — see module docs — but the same latency-hiding logic applies to
/// the separate mul+add). A single-chunk (4-chain) fallback handles one
/// leftover lane-of-4, then a scalar tail.
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn pv4_sse41(acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `chunk + 4/8 <= d` guards cover the `a0..a3` accumulator loads/stores; the inner `j < bc` loop's `v_block` index stays in bounds since `v_block.len() == bc * d` is asserted above and `chunk (+ 4/8) < d` from the outer guard.
    unsafe {
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
            let mut acc0a = _mm_loadu_ps(a0.as_ptr().add(chunk));
            let mut acc0b = _mm_loadu_ps(a0.as_ptr().add(chunk + 4));
            let mut acc1a = _mm_loadu_ps(a1.as_ptr().add(chunk));
            let mut acc1b = _mm_loadu_ps(a1.as_ptr().add(chunk + 4));
            let mut acc2a = _mm_loadu_ps(a2.as_ptr().add(chunk));
            let mut acc2b = _mm_loadu_ps(a2.as_ptr().add(chunk + 4));
            let mut acc3a = _mm_loadu_ps(a3.as_ptr().add(chunk));
            let mut acc3b = _mm_loadu_ps(a3.as_ptr().add(chunk + 4));
            let mut j = 0usize;
            while j < bc {
                let vva = _mm_loadu_ps(v_block.as_ptr().add(j * d + chunk));
                let vvb = _mm_loadu_ps(v_block.as_ptr().add(j * d + chunk + 4));
                let s0 = _mm_set1_ps(p[0][j]);
                let s1 = _mm_set1_ps(p[1][j]);
                let s2 = _mm_set1_ps(p[2][j]);
                let s3 = _mm_set1_ps(p[3][j]);
                acc0a = _mm_add_ps(acc0a, _mm_mul_ps(vva, s0));
                acc0b = _mm_add_ps(acc0b, _mm_mul_ps(vvb, s0));
                acc1a = _mm_add_ps(acc1a, _mm_mul_ps(vva, s1));
                acc1b = _mm_add_ps(acc1b, _mm_mul_ps(vvb, s1));
                acc2a = _mm_add_ps(acc2a, _mm_mul_ps(vva, s2));
                acc2b = _mm_add_ps(acc2b, _mm_mul_ps(vvb, s2));
                acc3a = _mm_add_ps(acc3a, _mm_mul_ps(vva, s3));
                acc3b = _mm_add_ps(acc3b, _mm_mul_ps(vvb, s3));
                j += 1;
            }
            _mm_storeu_ps(a0.as_mut_ptr().add(chunk), acc0a);
            _mm_storeu_ps(a0.as_mut_ptr().add(chunk + 4), acc0b);
            _mm_storeu_ps(a1.as_mut_ptr().add(chunk), acc1a);
            _mm_storeu_ps(a1.as_mut_ptr().add(chunk + 4), acc1b);
            _mm_storeu_ps(a2.as_mut_ptr().add(chunk), acc2a);
            _mm_storeu_ps(a2.as_mut_ptr().add(chunk + 4), acc2b);
            _mm_storeu_ps(a3.as_mut_ptr().add(chunk), acc3a);
            _mm_storeu_ps(a3.as_mut_ptr().add(chunk + 4), acc3b);
            chunk += 8;
        }
        if chunk + 4 <= d {
            let mut acc0 = _mm_loadu_ps(a0.as_ptr().add(chunk));
            let mut acc1 = _mm_loadu_ps(a1.as_ptr().add(chunk));
            let mut acc2 = _mm_loadu_ps(a2.as_ptr().add(chunk));
            let mut acc3 = _mm_loadu_ps(a3.as_ptr().add(chunk));
            let mut j = 0usize;
            while j < bc {
                let vv = _mm_loadu_ps(v_block.as_ptr().add(j * d + chunk));
                acc0 = _mm_add_ps(acc0, _mm_mul_ps(vv, _mm_set1_ps(p[0][j])));
                acc1 = _mm_add_ps(acc1, _mm_mul_ps(vv, _mm_set1_ps(p[1][j])));
                acc2 = _mm_add_ps(acc2, _mm_mul_ps(vv, _mm_set1_ps(p[2][j])));
                acc3 = _mm_add_ps(acc3, _mm_mul_ps(vv, _mm_set1_ps(p[3][j])));
                j += 1;
            }
            _mm_storeu_ps(a0.as_mut_ptr().add(chunk), acc0);
            _mm_storeu_ps(a1.as_mut_ptr().add(chunk), acc1);
            _mm_storeu_ps(a2.as_mut_ptr().add(chunk), acc2);
            _mm_storeu_ps(a3.as_mut_ptr().add(chunk), acc3);
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
}

#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn scale_sse41(dst: &mut [f32], scale: f32) {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the load and the store back to the same index.
    unsafe {
        let len = dst.len();
        let vscale = _mm_set1_ps(scale);
        let mut i = 0usize;
        while i + 4 <= len {
            let d = _mm_loadu_ps(dst.as_ptr().add(i));
            let r = _mm_mul_ps(d, vscale);
            _mm_storeu_ps(dst.as_mut_ptr().add(i), r);
            i += 4;
        }
        while i < len {
            dst[i] *= scale;
            i += 1;
        }
    }
}

#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn max_reduce_sse41(x: &[f32]) -> f32 {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the load; the `len == 0` early return above avoids reducing an empty accumulator.
    unsafe {
        let len = x.len();
        if len == 0 {
            return f32::NEG_INFINITY;
        }
        let mut acc = _mm_set1_ps(f32::NEG_INFINITY);
        let mut i = 0usize;
        while i + 4 <= len {
            let v = _mm_loadu_ps(x.as_ptr().add(i));
            acc = _mm_max_ps(acc, v);
            i += 4;
        }
        let mut m = hmax128_ps(acc);
        while i < len {
            m = m.max(x[i]);
            i += 1;
        }
        m
    }
}

/// [`max_reduce_sse41`], 4 rows at once, interleaved into 4 independent
/// chains — see [`crate::kernel::Kernel::max_reduce4`] for why.
#[target_feature(enable = "sse2,sse3,sse4.1")]
unsafe fn max_reduce4_sse41(x: [&[f32]; 4]) -> [f32; 4] {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this crate only ever calls it after confirming `is_x86_feature_detected!("sse4.1")` (see `Sse41Kernel`'s callers in `v1.rs`/`v2.rs`/`v3.rs`).
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the load for all four rows (asserted equal length above); the `len == 0` early return above avoids reducing an empty accumulator.
    unsafe {
        let len = x[0].len();
        debug_assert_eq!(x[1].len(), len);
        debug_assert_eq!(x[2].len(), len);
        debug_assert_eq!(x[3].len(), len);
        if len == 0 {
            return [f32::NEG_INFINITY; 4];
        }
        let mut acc = [_mm_set1_ps(f32::NEG_INFINITY); 4];
        let mut i = 0usize;
        while i + 4 <= len {
            for r in 0..4 {
                acc[r] = _mm_max_ps(acc[r], _mm_loadu_ps(x[r].as_ptr().add(i)));
            }
            i += 4;
        }
        let mut m: [f32; 4] = std::array::from_fn(|r| hmax128_ps(acc[r]));
        while i < len {
            for r in 0..4 {
                m[r] = m[r].max(x[r][i]);
            }
            i += 1;
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse41_available() -> bool {
        is_x86_feature_detected!("sse4.1")
    }

    #[test]
    fn exp_matches_std() {
        if !sse41_available() {
            return;
        }
        let xs: Vec<f32> = (-800..800).map(|i| i as f32 * 0.1).collect();
        let mut got = xs.clone();
        // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
        let sum = unsafe { sub_exp_sum_inplace_sse41(&mut got, 0.0) };
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
        if !sse41_available() {
            return;
        }
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.71).cos()).collect();
            let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            let simd = unsafe { dot_sse41(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3 * (scalar.abs() + 1.0),
                "len={len} scalar={scalar} simd={simd}"
            );
        }
    }

    #[test]
    fn dot4_matches_four_dots() {
        if !sse41_available() {
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
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { dot_sse41(&a0, &b) },
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { dot_sse41(&a1, &b) },
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { dot_sse41(&a2, &b) },
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { dot_sse41(&a3, &b) },
            ];
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            let got = unsafe { dot4_sse41(&a0, &a1, &a2, &a3, &b) };
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
        if !sse41_available() {
            return;
        }
        for d in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk = |seed: f32| -> Vec<f32> { (0..d).map(|i| (i as f32 * seed).sin()).collect() };
            let q = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let k = [mk(0.61), mk(0.67), mk(0.73), mk(0.79)];

            let want: [[f32; 4]; 4] = std::array::from_fn(|r| {
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                std::array::from_fn(|c| unsafe { dot_sse41(&q[r], &k[c]) })
            });
            let got =
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { dot4x4_sse41([&q[0], &q[1], &q[2], &q[3]], [&k[0], &k[1], &k[2], &k[3]]) };
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
        if !sse41_available() {
            return;
        }
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
                    // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                    unsafe { axpy_sse41(row, v_row, pr[j]) };
                }
            }

            let mut got = init;
            let [g0, g1, g2, g3] = &mut got;
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            unsafe { pv4_sse41([g0, g1, g2, g3], &v_block, [&p[0], &p[1], &p[2], &p[3]]) };

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
        if !sse41_available() {
            return;
        }
        for len in [0usize, 1, 5, 8, 13, 16, 33] {
            let x: Vec<f32> = (0..len).map(|i| ((i as f32) * 1.3).sin() * 5.0).collect();
            let want_max: f32 = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            let got_max = unsafe { max_reduce_sse41(&x) };
            assert_eq!(want_max, got_max, "len={len} max");
        }
    }

    #[test]
    fn max_reduce4_matches_four_max_reduces() {
        if !sse41_available() {
            return;
        }
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk =
                |seed: f32| -> Vec<f32> { (0..len).map(|i| (i as f32 * seed).sin()).collect() };
            let rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let want = [
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { max_reduce_sse41(&rows[0]) },
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { max_reduce_sse41(&rows[1]) },
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { max_reduce_sse41(&rows[2]) },
                // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
                unsafe { max_reduce_sse41(&rows[3]) },
            ];
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            let got = unsafe { max_reduce4_sse41([&rows[0], &rows[1], &rows[2], &rows[3]]) };
            assert_eq!(want, got, "len={len}");
        }
    }

    #[test]
    fn sub_exp_sum_inplace4_matches_four_calls() {
        if !sse41_available() {
            return;
        }
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk = |seed: f32| -> Vec<f32> {
                (0..len).map(|i| (i as f32 * seed).sin() * 4.0).collect()
            };
            let m = [0.1f32, -0.3, 0.5, 0.0];

            let mut want_rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            let want_sums: [f32; 4] = std::array::from_fn(|r| unsafe {
                sub_exp_sum_inplace_sse41(&mut want_rows[r], m[r])
            });

            let mut got_rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let [g0, g1, g2, g3] = &mut got_rows;
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            let got_sums = unsafe { sub_exp_sum_inplace4_sse41([g0, g1, g2, g3], m) };

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
        if !sse41_available() {
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
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            unsafe { axpy_sse41(&mut dst, &src, scale) };
            for (w, g) in want.iter().zip(dst.iter()) {
                assert!((w - g).abs() < 1e-4, "axpy len={len}");
            }

            let mut d2 = want.clone();
            let want2: Vec<f32> = d2.iter().map(|v| v * 2.5).collect();
            // SAFETY: test-only; guarded by the `sse41_available()` check (`is_x86_feature_detected!("sse4.1")`) at the top of this test, matching the same precondition the real dispatch in `v1.rs`/`v2.rs`/`v3.rs` enforces.
            unsafe { scale_sse41(&mut d2, 2.5) };
            for (w, g) in want2.iter().zip(d2.iter()) {
                assert!((w - g).abs() < 1e-4, "scale len={len}");
            }
        }
    }
}
