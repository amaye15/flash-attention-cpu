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
//! **Fused multiply-add, opt-in.** WASM SIMD128's baseline instruction set
//! (unlike AVX2/AVX-512's `vfmadd*`/NEON's `vfmaq_f32`) has no fused
//! multiply-add of its own — every accumulation here is structured as a
//! separate multiply then add, two roundings instead of one, *unless* the
//! further opt-in `relaxed-simd` target feature is also enabled at compile
//! time (WASM has no runtime feature detection, so this is a build-time
//! choice — see [`fma128_ps`]). `relaxed-simd` reached full
//! standardization (Phase 4) in 2024 and Rust stabilized the corresponding
//! `core::arch::wasm32` intrinsics in 1.82, comfortably under this crate's
//! MSRV — but it's a narrower guarantee than baseline `simd128`: Chrome and
//! Firefox (and Node.js/V8, which is what `wasm-pack test --node` runs
//! against) support it, Safari doesn't yet. That's why it's a second,
//! separate opt-in layered on top of `simd128` rather than folded into the
//! default `.cargo/config.toml` flags this repo's own tests build with —
//! see [ROADMAP.md](https://github.com/amaye15/flash-attention-cpu/blob/main/ROADMAP.md#1-wasm-relaxed-simd-real-fma-doc-correction--open-opportunity)
//! and CI's `wasm-relaxed-simd` job for how the fused path gets exercised.
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

impl Simd128Kernel {
    /// Always succeeds — this module only exists when compiled with
    /// `target_feature = "simd128"` (see the `#[cfg(...)]` gate in
    /// `lib.rs`), so the precondition is already guaranteed at compile
    /// time, not something to check at runtime.
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Kernel for Simd128Kernel {
    #[inline]
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { dot_simd128(a, b) }
    }

    #[inline]
    fn dot4(&self, a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { dot4_simd128(a0, a1, a2, a3, b) }
    }

    #[inline]
    fn dot4x4(&self, q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { dot4x4_simd128(q, k) }
    }

    #[inline]
    fn sub_exp_sum_inplace(&self, x: &mut [f32], m: f32) -> f32 {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { sub_exp_sum_inplace_simd128(x, m) }
    }

    #[inline]
    fn sub_exp_sum_inplace4(&self, x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { sub_exp_sum_inplace4_simd128(x, m) }
    }

    #[inline]
    fn axpy(&self, dst: &mut [f32], src: &[f32], scale: f32) {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { axpy_simd128(dst, src, scale) }
    }

    #[inline]
    fn pv4(&self, acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { pv4_simd128(acc, v_block, p) }
    }

    #[inline]
    fn scale_inplace(&self, dst: &mut [f32], scale: f32) {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { scale_simd128(dst, scale) }
    }

    #[inline]
    fn max_reduce(&self, x: &[f32]) -> f32 {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { max_reduce_simd128(x) }
    }

    #[inline]
    fn max_reduce4(&self, x: [&[f32]; 4]) -> [f32; 4] {
        // SAFETY: `Self` is only constructible via `Simd128Kernel::new()` (see its docs), which already confirmed the precondition below.
        unsafe { max_reduce4_simd128(x) }
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

/// `a * b + c`, fused when the `relaxed-simd` target feature is enabled at
/// compile time (`f32x4_relaxed_madd` — one rounding), falling back to a
/// separate multiply then add (two roundings) when it isn't — see module
/// docs. Every accumulation site in this file goes through this one
/// function so the fused/unfused choice is made in exactly one place.
///
/// "Relaxed" here refers to the *specification* allowing either
/// fused-or-not codegen depending on the target (that's the portability
/// tradeoff, not a precision one on any single run) — on hardware that
/// actually executes it fused (`f32x4.relaxed_madd` compiles to real FMA
/// on both x86 and Arm backends), results are at least as accurate as the
/// unfused path, never less.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn fma128_ps(a: v128, b: v128, c: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    {
        f32x4_relaxed_madd(a, b, c)
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        f32x4_add(c, f32x4_mul(a, b))
    }
}

/// Dot product, 2-way accumulator unrolled (8 f32 / iteration) — same idea
/// as `avx2::dot_avx2`/`neon::dot_neon`, just without a fused multiply-add
/// (see module docs), so each step is a separate `mul` + `add`.
#[target_feature(enable = "simd128")]
unsafe fn dot_simd128(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: each load reads a fixed-width window starting at `i`; the `i + 8 <= len` guards above ensure every load stays within `a`/`b` (asserted equal length above).
    unsafe {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len();
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut i = 0usize;
        while i + 8 <= len {
            let a0 = v128_load(a.as_ptr().add(i) as *const v128);
            let b0 = v128_load(b.as_ptr().add(i) as *const v128);
            acc0 = fma128_ps(a0, b0, acc0);
            let a1 = v128_load(a.as_ptr().add(i + 4) as *const v128);
            let b1 = v128_load(b.as_ptr().add(i + 4) as *const v128);
            acc1 = fma128_ps(a1, b1, acc1);
            i += 8;
        }
        while i + 4 <= len {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            acc0 = fma128_ps(av, bv, acc0);
            i += 4;
        }
        let mut sum = hsum128_ps(f32x4_add(acc0, acc1));
        while i < len {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// Four dot products sharing `b`'s vector loads across four independent
/// mul+add accumulator chains — see [`crate::kernel::Kernel::dot4`] for why
/// this is faster than four separate [`dot_simd128`] calls.
#[target_feature(enable = "simd128")]
unsafe fn dot4_simd128(a0: &[f32], a1: &[f32], a2: &[f32], a3: &[f32], b: &[f32]) -> [f32; 4] {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the shared `b` load and all four `a0..a3` loads each iteration (all four asserted equal length to `b` above).
    unsafe {
        debug_assert_eq!(a0.len(), b.len());
        debug_assert_eq!(a1.len(), b.len());
        debug_assert_eq!(a2.len(), b.len());
        debug_assert_eq!(a3.len(), b.len());
        let len = b.len();
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut acc2 = f32x4_splat(0.0);
        let mut acc3 = f32x4_splat(0.0);
        let mut i = 0usize;
        while i + 4 <= len {
            let bv = v128_load(b.as_ptr().add(i) as *const v128); // loaded once, shared 4 ways
            acc0 = fma128_ps(v128_load(a0.as_ptr().add(i) as *const v128), bv, acc0);
            acc1 = fma128_ps(v128_load(a1.as_ptr().add(i) as *const v128), bv, acc1);
            acc2 = fma128_ps(v128_load(a2.as_ptr().add(i) as *const v128), bv, acc2);
            acc3 = fma128_ps(v128_load(a3.as_ptr().add(i) as *const v128), bv, acc3);
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
#[target_feature(enable = "simd128")]
unsafe fn dot4x4_simd128(q: [&[f32]; 4], k: [&[f32]; 4]) -> [[f32; 4]; 4] {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: the `p + 4 <= d` guard covers all four `q` and four `k` row loads each iteration.
    unsafe {
        let d = q[0].len();
        let mut acc = [[f32x4_splat(0.0); 4]; 4];
        let mut p = 0usize;
        while p + 4 <= d {
            let qv = [
                v128_load(q[0].as_ptr().add(p) as *const v128),
                v128_load(q[1].as_ptr().add(p) as *const v128),
                v128_load(q[2].as_ptr().add(p) as *const v128),
                v128_load(q[3].as_ptr().add(p) as *const v128),
            ];
            for c in 0..4 {
                let kv = v128_load(k[c].as_ptr().add(p) as *const v128); // loaded once, shared 4 ways
                for r in 0..4 {
                    acc[r][c] = fma128_ps(qv[r], kv, acc[r][c]);
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
/// minimax fit `avx2::exp256_ps`/`neon::exp128_ps` use, kept at full
/// precision as documented there rather than trimmed to placate
/// `clippy::excessive_precision`.
#[target_feature(enable = "simd128")]
#[allow(clippy::excessive_precision)]
unsafe fn exp128_ps(x: v128) -> v128 {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // No raw pointers here (unlike most other functions in this file) — every operation is pure register arithmetic on already-valid `v128` values; the only unsafe requirement is the same CPU-feature precondition, inherited by the `fma128_ps` calls below.
    unsafe {
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
        let fx = fma128_ps(x, log2ef, half);
        let fx = f32x4_floor(fx);

        // r = x - n*ln(2), split hi/lo for precision
        let tmp = f32x4_mul(fx, c1);
        let z = f32x4_mul(fx, c2);
        let x = f32x4_sub(x, tmp);
        let x = f32x4_sub(x, z);

        let z = f32x4_mul(x, x);

        // degree-5 minimax polynomial for exp(r)
        let mut y = p0;
        y = fma128_ps(y, x, p1);
        y = fma128_ps(y, x, p2);
        y = fma128_ps(y, x, p3);
        y = fma128_ps(y, x, p4);
        y = fma128_ps(y, x, p5);
        y = fma128_ps(y, z, x);
        y = f32x4_add(y, one);

        // 2^n via direct exponent-bit construction. No bitcast needed: `v128`
        // is a single type for every lane interpretation in this API.
        let imm0 = i32x4_trunc_sat_f32x4(fx);
        let imm0 = i32x4_add(imm0, i32x4_splat(0x7f));
        let pow2n = i32x4_shl(imm0, 23);

        f32x4_mul(y, pow2n)
    }
}

/// Fused `x[i] = exp(x[i] - m)`, returning `sum(x)` after the exponential:
/// subtract, exponential, and sum accumulation all in the same pass over
/// `x` (one load/store per lane, plus the sum, instead of two separate
/// passes).
#[target_feature(enable = "simd128")]
unsafe fn sub_exp_sum_inplace_simd128(x: &mut [f32], m: f32) -> f32 {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers both the load and the store back to the same index range.
    unsafe {
        let len = x.len();
        let vm = f32x4_splat(m);
        let mut sum_acc = f32x4_splat(0.0);
        let mut i = 0usize;
        while i + 4 <= len {
            let v = v128_load(x.as_ptr().add(i) as *const v128);
            let v = f32x4_sub(v, vm);
            let r = exp128_ps(v);
            v128_store(x.as_mut_ptr().add(i) as *mut v128, r);
            sum_acc = f32x4_add(sum_acc, r);
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

/// [`sub_exp_sum_inplace_simd128`], 4 rows at once with per-row `m` values,
/// interleaved into 4 independent chains — see
/// [`crate::kernel::Kernel::sub_exp_sum_inplace4`] for why.
#[target_feature(enable = "simd128")]
unsafe fn sub_exp_sum_inplace4_simd128(x: [&mut [f32]; 4], m: [f32; 4]) -> [f32; 4] {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the load/store for all four rows (asserted equal length above).
    unsafe {
        let [x0, x1, x2, x3] = x;
        let len = x0.len();
        debug_assert_eq!(x1.len(), len);
        debug_assert_eq!(x2.len(), len);
        debug_assert_eq!(x3.len(), len);
        let vm = [
            f32x4_splat(m[0]),
            f32x4_splat(m[1]),
            f32x4_splat(m[2]),
            f32x4_splat(m[3]),
        ];
        let mut sum_acc = [f32x4_splat(0.0); 4];
        let mut i = 0usize;
        while i + 4 <= len {
            let r0 = exp128_ps(f32x4_sub(
                v128_load(x0.as_ptr().add(i) as *const v128),
                vm[0],
            ));
            v128_store(x0.as_mut_ptr().add(i) as *mut v128, r0);
            sum_acc[0] = f32x4_add(sum_acc[0], r0);
            let r1 = exp128_ps(f32x4_sub(
                v128_load(x1.as_ptr().add(i) as *const v128),
                vm[1],
            ));
            v128_store(x1.as_mut_ptr().add(i) as *mut v128, r1);
            sum_acc[1] = f32x4_add(sum_acc[1], r1);
            let r2 = exp128_ps(f32x4_sub(
                v128_load(x2.as_ptr().add(i) as *const v128),
                vm[2],
            ));
            v128_store(x2.as_mut_ptr().add(i) as *mut v128, r2);
            sum_acc[2] = f32x4_add(sum_acc[2], r2);
            let r3 = exp128_ps(f32x4_sub(
                v128_load(x3.as_ptr().add(i) as *const v128),
                vm[3],
            ));
            v128_store(x3.as_mut_ptr().add(i) as *mut v128, r3);
            sum_acc[3] = f32x4_add(sum_acc[3], r3);
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

#[target_feature(enable = "simd128")]
unsafe fn axpy_simd128(dst: &mut [f32], src: &[f32], scale: f32) {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers both the `dst` load/store and the `src` load (asserted equal length above).
    unsafe {
        debug_assert_eq!(dst.len(), src.len());
        let len = dst.len();
        let vscale = f32x4_splat(scale);
        let mut i = 0usize;
        while i + 4 <= len {
            let d = v128_load(dst.as_ptr().add(i) as *const v128);
            let s = v128_load(src.as_ptr().add(i) as *const v128);
            let r = fma128_ps(s, vscale, d);
            v128_store(dst.as_mut_ptr().add(i) as *mut v128, r);
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
#[target_feature(enable = "simd128")]
unsafe fn pv4_simd128(acc: [&mut [f32]; 4], v_block: &[f32], p: [&[f32]; 4]) {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
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
            let mut acc0a = v128_load(a0.as_ptr().add(chunk) as *const v128);
            let mut acc0b = v128_load(a0.as_ptr().add(chunk + 4) as *const v128);
            let mut acc1a = v128_load(a1.as_ptr().add(chunk) as *const v128);
            let mut acc1b = v128_load(a1.as_ptr().add(chunk + 4) as *const v128);
            let mut acc2a = v128_load(a2.as_ptr().add(chunk) as *const v128);
            let mut acc2b = v128_load(a2.as_ptr().add(chunk + 4) as *const v128);
            let mut acc3a = v128_load(a3.as_ptr().add(chunk) as *const v128);
            let mut acc3b = v128_load(a3.as_ptr().add(chunk + 4) as *const v128);
            let mut j = 0usize;
            while j < bc {
                let vva = v128_load(v_block.as_ptr().add(j * d + chunk) as *const v128);
                let vvb = v128_load(v_block.as_ptr().add(j * d + chunk + 4) as *const v128);
                let s0 = f32x4_splat(p[0][j]);
                let s1 = f32x4_splat(p[1][j]);
                let s2 = f32x4_splat(p[2][j]);
                let s3 = f32x4_splat(p[3][j]);
                acc0a = fma128_ps(vva, s0, acc0a);
                acc0b = fma128_ps(vvb, s0, acc0b);
                acc1a = fma128_ps(vva, s1, acc1a);
                acc1b = fma128_ps(vvb, s1, acc1b);
                acc2a = fma128_ps(vva, s2, acc2a);
                acc2b = fma128_ps(vvb, s2, acc2b);
                acc3a = fma128_ps(vva, s3, acc3a);
                acc3b = fma128_ps(vvb, s3, acc3b);
                j += 1;
            }
            v128_store(a0.as_mut_ptr().add(chunk) as *mut v128, acc0a);
            v128_store(a0.as_mut_ptr().add(chunk + 4) as *mut v128, acc0b);
            v128_store(a1.as_mut_ptr().add(chunk) as *mut v128, acc1a);
            v128_store(a1.as_mut_ptr().add(chunk + 4) as *mut v128, acc1b);
            v128_store(a2.as_mut_ptr().add(chunk) as *mut v128, acc2a);
            v128_store(a2.as_mut_ptr().add(chunk + 4) as *mut v128, acc2b);
            v128_store(a3.as_mut_ptr().add(chunk) as *mut v128, acc3a);
            v128_store(a3.as_mut_ptr().add(chunk + 4) as *mut v128, acc3b);
            chunk += 8;
        }
        if chunk + 4 <= d {
            let mut acc0 = v128_load(a0.as_ptr().add(chunk) as *const v128);
            let mut acc1 = v128_load(a1.as_ptr().add(chunk) as *const v128);
            let mut acc2 = v128_load(a2.as_ptr().add(chunk) as *const v128);
            let mut acc3 = v128_load(a3.as_ptr().add(chunk) as *const v128);
            let mut j = 0usize;
            while j < bc {
                let vv = v128_load(v_block.as_ptr().add(j * d + chunk) as *const v128);
                acc0 = fma128_ps(vv, f32x4_splat(p[0][j]), acc0);
                acc1 = fma128_ps(vv, f32x4_splat(p[1][j]), acc1);
                acc2 = fma128_ps(vv, f32x4_splat(p[2][j]), acc2);
                acc3 = fma128_ps(vv, f32x4_splat(p[3][j]), acc3);
                j += 1;
            }
            v128_store(a0.as_mut_ptr().add(chunk) as *mut v128, acc0);
            v128_store(a1.as_mut_ptr().add(chunk) as *mut v128, acc1);
            v128_store(a2.as_mut_ptr().add(chunk) as *mut v128, acc2);
            v128_store(a3.as_mut_ptr().add(chunk) as *mut v128, acc3);
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

#[target_feature(enable = "simd128")]
unsafe fn scale_simd128(dst: &mut [f32], scale: f32) {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the load and the store back to the same index.
    unsafe {
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
}

#[target_feature(enable = "simd128")]
unsafe fn max_reduce_simd128(x: &[f32]) -> f32 {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the load; the `len == 0` early return above avoids reducing an empty accumulator.
    unsafe {
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
}

/// [`max_reduce_simd128`], 4 rows at once, interleaved into 4 independent
/// chains — see [`crate::kernel::Kernel::max_reduce4`] for why.
#[target_feature(enable = "simd128")]
unsafe fn max_reduce4_simd128(x: [&[f32]; 4]) -> [f32; 4] {
    // SAFETY: `#[target_feature(enable = "...")]` requires the CPU to
    // actually support it, which holds because this module only exists when compiled with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, not checked at runtime.
    // All raw-pointer loads/stores below stay in bounds: the `i + 4 <= len` guard covers the load for all four rows (asserted equal length above); the `len == 0` early return above avoids reducing an empty accumulator.
    unsafe {
        let len = x[0].len();
        debug_assert_eq!(x[1].len(), len);
        debug_assert_eq!(x[2].len(), len);
        debug_assert_eq!(x[3].len(), len);
        if len == 0 {
            return [f32::NEG_INFINITY; 4];
        }
        let mut acc = [f32x4_splat(f32::NEG_INFINITY); 4];
        let mut i = 0usize;
        while i + 4 <= len {
            for r in 0..4 {
                acc[r] = f32x4_max(acc[r], v128_load(x[r].as_ptr().add(i) as *const v128));
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

// `wasm-bindgen-test`'s harness only discovers `#[wasm_bindgen_test]`-marked
// functions on wasm32 — plain `#[test]` silently never runs under
// `wasm-pack test --node` (no OS process/argv support for the default
// libtest harness on `wasm32-unknown-unknown`; see `tests/wasm_simd.rs`'s
// module docs for the same point about integration tests). Every test in
// this module uses `#[wasm_bindgen_test]` instead for exactly that reason —
// these used to be plain `#[test]` and were silently never executed.
#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    /// `fma128_ps` must equal `a*b+c` regardless of which branch (fused or
    /// unfused) it compiles to — this is what makes swapping in the
    /// `relaxed-simd` path a pure codegen choice rather than a behavior
    /// change. Tolerance is loose enough to accept either rounding.
    #[wasm_bindgen_test]
    fn fma_matches_mul_add() {
        let cases: [(f32, f32, f32); 5] = [
            (1.5, 2.5, 0.5),
            (-3.25, 4.0, 1.0),
            (0.0, 123.456, -7.0),
            (1e-6, 1e6, 2.0),
            (-1.0, -1.0, -1.0),
        ];
        for (a, b, c) in cases {
            let want = a * b + c;
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            let got = unsafe {
                let r = fma128_ps(f32x4_splat(a), f32x4_splat(b), f32x4_splat(c));
                f32x4_extract_lane::<0>(r)
            };
            assert!(
                (want - got).abs() < 1e-3 * (want.abs() + 1.0),
                "a={a} b={b} c={c} want={want} got={got}"
            );
        }
    }

    #[wasm_bindgen_test]
    fn exp_matches_std() {
        let xs: Vec<f32> = (-800..800).map(|i| i as f32 * 0.1).collect();
        let mut got = xs.clone();
        // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
        let sum = unsafe { sub_exp_sum_inplace_simd128(&mut got, 0.0) };
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

    #[wasm_bindgen_test]
    fn dot_matches_scalar() {
        for len in [0usize, 1, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.71).cos()).collect();
            let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            let simd = unsafe { dot_simd128(&a, &b) };
            assert!(
                (scalar - simd).abs() < 1e-3 * (scalar.abs() + 1.0),
                "len={len} scalar={scalar} simd={simd}"
            );
        }
    }

    #[wasm_bindgen_test]
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
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                unsafe { dot_simd128(&a0, &b) },
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                unsafe { dot_simd128(&a1, &b) },
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                unsafe { dot_simd128(&a2, &b) },
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                unsafe { dot_simd128(&a3, &b) },
            ];
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            let got = unsafe { dot4_simd128(&a0, &a1, &a2, &a3, &b) };
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

    #[wasm_bindgen_test]
    fn dot4x4_matches_naive() {
        for d in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk = |seed: f32| -> Vec<f32> { (0..d).map(|i| (i as f32 * seed).sin()).collect() };
            let q = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let k = [mk(0.61), mk(0.67), mk(0.73), mk(0.79)];

            let want: [[f32; 4]; 4] = std::array::from_fn(|r| {
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                std::array::from_fn(|c| unsafe { dot_simd128(&q[r], &k[c]) })
            });
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            let got = unsafe {
                dot4x4_simd128([&q[0], &q[1], &q[2], &q[3]], [&k[0], &k[1], &k[2], &k[3]])
            };
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

    #[wasm_bindgen_test]
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
                    // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                    unsafe { axpy_simd128(row, v_row, pr[j]) };
                }
            }

            let mut got = init;
            let [g0, g1, g2, g3] = &mut got;
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            unsafe { pv4_simd128([g0, g1, g2, g3], &v_block, [&p[0], &p[1], &p[2], &p[3]]) };

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

    #[wasm_bindgen_test]
    fn reductions_match_scalar() {
        for len in [0usize, 1, 5, 8, 13, 16, 33] {
            let x: Vec<f32> = (0..len).map(|i| ((i as f32) * 1.3).sin() * 5.0).collect();
            let want_max: f32 = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            let got_max = unsafe { max_reduce_simd128(&x) };
            assert_eq!(want_max, got_max, "len={len} max");
        }
    }

    #[wasm_bindgen_test]
    fn max_reduce4_matches_four_max_reduces() {
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk =
                |seed: f32| -> Vec<f32> { (0..len).map(|i| (i as f32 * seed).sin()).collect() };
            let rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let want = [
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                unsafe { max_reduce_simd128(&rows[0]) },
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                unsafe { max_reduce_simd128(&rows[1]) },
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                unsafe { max_reduce_simd128(&rows[2]) },
                // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
                unsafe { max_reduce_simd128(&rows[3]) },
            ];
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            let got = unsafe { max_reduce4_simd128([&rows[0], &rows[1], &rows[2], &rows[3]]) };
            assert_eq!(want, got, "len={len}");
        }
    }

    #[wasm_bindgen_test]
    fn sub_exp_sum_inplace4_matches_four_calls() {
        for len in [0usize, 1, 3, 4, 5, 7, 8, 9, 33] {
            let mk = |seed: f32| -> Vec<f32> {
                (0..len).map(|i| (i as f32 * seed).sin() * 4.0).collect()
            };
            let m = [0.1f32, -0.3, 0.5, 0.0];

            let mut want_rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            let want_sums: [f32; 4] = std::array::from_fn(|r| unsafe {
                sub_exp_sum_inplace_simd128(&mut want_rows[r], m[r])
            });

            let mut got_rows = [mk(0.11), mk(0.23), mk(0.37), mk(0.51)];
            let [g0, g1, g2, g3] = &mut got_rows;
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            let got_sums = unsafe { sub_exp_sum_inplace4_simd128([g0, g1, g2, g3], m) };

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

    #[wasm_bindgen_test]
    fn axpy_and_scale_match_scalar() {
        for len in [0usize, 1, 7, 8, 9, 31, 32, 33] {
            let mut dst: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let src: Vec<f32> = (0..len).map(|i| (i as f32 * 0.2).cos()).collect();
            let scale = 1.37f32;
            let mut want = dst.clone();
            for (d, s) in want.iter_mut().zip(src.iter()) {
                *d += s * scale;
            }
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            unsafe { axpy_simd128(&mut dst, &src, scale) };
            for (w, g) in want.iter().zip(dst.iter()) {
                assert!((w - g).abs() < 1e-4, "axpy len={len}");
            }

            let mut d2 = want.clone();
            let want2: Vec<f32> = d2.iter().map(|v| v * 2.5).collect();
            // SAFETY: test-only; this module only compiles in when built with `target_feature = "simd128"` (see the `#[cfg(...)]` gate in `lib.rs`), so the feature is guaranteed at compile time, same as the real dispatch.
            unsafe { scale_simd128(&mut d2, 2.5) };
            for (w, g) in want2.iter().zip(d2.iter()) {
                assert!((w - g).abs() < 1e-4, "scale len={len}");
            }
        }
    }
}
