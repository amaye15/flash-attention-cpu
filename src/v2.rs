//! FlashAttention-2 (Dao, 2023): tiling + online softmax with the query
//! block as the outer (parallel) loop and the KV block as the inner loop,
//! a single deferred normalization after the whole KV sweep, and a causal
//! early-exit that skips fully-future KV tiles outright rather than
//! computing and masking them.
//!
//! This was the crate's original (unversioned) algorithm; see the crate
//! docs and README for how it compares to [`crate::v1`] and [`crate::v3`].

#[cfg(target_arch = "x86_64")]
use crate::avx2::Avx2Kernel;
#[cfg(target_arch = "x86_64")]
use crate::avx512::Avx512Kernel;
use crate::common::{check_shapes, multihead_dispatch, FlashAttentionConfig};
use crate::kernel::Kernel;
#[cfg(target_arch = "aarch64")]
use crate::neon::NeonKernel;
#[cfg(any(
    test,
    not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))
))]
use crate::scalar::ScalarKernel;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::simd128::Simd128Kernel;
use rayon::prelude::*;

/// Single-head scaled dot-product attention via tiling + online softmax
/// ("flash attention"), dispatched at runtime/compile-time to the fastest
/// available SIMD kernel (AVX-512F/AVX2+FMA on x86_64, NEON on aarch64,
/// SIMD128 on wasm32 when built with that feature), and a portable scalar
/// kernel otherwise.
///
/// `q`: `[seq_len_q, d_head]`, `k`/`v`: `[seq_len_k, d_head]`, row-major.
/// `out`: `[seq_len_q, d_head]`, overwritten. Peak extra memory is
/// `O(block_size_q * (d_head + block_size_kv))`, independent of the full
/// sequence length — the whole point of tiling versus materializing an
/// `O(seq_len_q * seq_len_k)` score matrix.
///
/// # Panics
///
/// Panics if `q.len() != seq_len_q * d_head`, `k.len() != seq_len_k * d_head`,
/// `v.len() != seq_len_k * d_head`, or `out.len() != seq_len_q * d_head`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_v2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len_q: usize,
    seq_len_k: usize,
    d_head: usize,
    config: &FlashAttentionConfig,
    out: &mut [f32],
) {
    if check_shapes(q, k, v, seq_len_q, seq_len_k, d_head, out) {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            run_v2::<Avx512Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            run_v2::<Avx2Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        run_v2::<NeonKernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        run_v2::<Simd128Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    run_v2::<ScalarKernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
}

/// Batched multi-head attention. See [`flash_attention_v2`] for the
/// single-head algorithm; this parallelizes over `batch * heads` and,
/// within each head, over query blocks.
///
/// # Panics
///
/// Panics if `q`/`k`/`v`/`out` don't match `batch * heads * seq_len_q *
/// d_head` (`q`/`out`) or `batch * heads * seq_len_k * d_head` (`k`/`v`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_multihead_v2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq_len_q: usize,
    seq_len_k: usize,
    d_head: usize,
    config: &FlashAttentionConfig,
    out: &mut [f32],
) {
    multihead_dispatch(
        q,
        k,
        v,
        batch,
        heads,
        seq_len_q,
        seq_len_k,
        d_head,
        config,
        out,
        flash_attention_v2,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_v2<K: Kernel + Sync>(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len_q: usize,
    seq_len_k: usize,
    d_head: usize,
    config: &FlashAttentionConfig,
    out: &mut [f32],
) {
    let scale = 1.0 / (d_head as f32).sqrt();
    let br = config.block_size_q.max(1).min(seq_len_q);
    let bc = config.block_size_kv.max(1).min(seq_len_k);

    out.par_chunks_mut(br * d_head)
        .enumerate()
        .for_each(|(qi, out_block)| {
            let q_start = qi * br;
            let this_br = out_block.len() / d_head;
            let q_block = &q[q_start * d_head..(q_start + this_br) * d_head];

            // Running online-softmax state for this query block.
            let mut m = vec![f32::NEG_INFINITY; this_br]; // running row max
            let mut l = vec![0.0f32; this_br]; // running row sum (of exp)
            let mut acc = vec![0.0f32; this_br * d_head]; // unnormalized output accumulator
            let mut scores = vec![0.0f32; this_br * bc]; // scratch S tile, reused per kv-block

            let num_kv_blocks = seq_len_k.div_ceil(bc);
            for kj in 0..num_kv_blocks {
                let k_start = kj * bc;
                let this_bc = bc.min(seq_len_k - k_start);

                // Entire tile is strictly future: skip, and since kv blocks
                // are visited in increasing order, so is everything after it.
                if config.causal && k_start > q_start + this_br - 1 {
                    break;
                }

                let k_block = &k[k_start * d_head..(k_start + this_bc) * d_head];
                let v_block = &v[k_start * d_head..(k_start + this_bc) * d_head];
                let scores_slice = &mut scores[..this_br * this_bc];

                // S_ij = scale * Q_i . K_j  (Q/K rows are contiguous, so this
                // is a plain dot product — no transpose needed).
                for i in 0..this_br {
                    let qi_row = &q_block[i * d_head..(i + 1) * d_head];
                    let s_row = &mut scores_slice[i * this_bc..(i + 1) * this_bc];
                    for (s, kj_row) in s_row.iter_mut().zip(k_block.chunks_exact(d_head)) {
                        *s = unsafe { K::dot(qi_row, kj_row) } * scale;
                    }
                }

                // Only touch the mask if this tile actually straddles the
                // diagonal; fully-visible tiles skip it entirely.
                if config.causal && k_start + this_bc - 1 > q_start {
                    for i in 0..this_br {
                        let global_i = q_start + i;
                        let s_row = &mut scores_slice[i * this_bc..(i + 1) * this_bc];
                        for (j, s) in s_row.iter_mut().enumerate() {
                            if k_start + j > global_i {
                                *s = f32::NEG_INFINITY;
                            }
                        }
                    }
                }

                // Online softmax update + PV accumulation, fused per row.
                for i in 0..this_br {
                    let s_row = &mut scores_slice[i * this_bc..(i + 1) * this_bc];
                    let block_max = unsafe { K::max_reduce(s_row) };
                    let new_m = m[i].max(block_max);

                    for x in s_row.iter_mut() {
                        *x -= new_m;
                    }
                    unsafe { K::exp_inplace(s_row) };

                    let block_sum = unsafe { K::sum_reduce(s_row) };
                    let correction = (m[i] - new_m).exp();

                    l[i] = correction * l[i] + block_sum;

                    let acc_row = &mut acc[i * d_head..(i + 1) * d_head];
                    unsafe { K::scale_inplace(acc_row, correction) };

                    for (v_row, &p) in v_block.chunks_exact(d_head).zip(s_row.iter()) {
                        unsafe { K::axpy(acc_row, v_row, p) };
                    }

                    m[i] = new_m;
                }
            }

            for i in 0..this_br {
                let inv_l = if l[i] > 0.0 { 1.0 / l[i] } else { 0.0 };
                let acc_row = &acc[i * d_head..(i + 1) * d_head];
                let out_row = &mut out_block[i * d_head..(i + 1) * d_head];
                for (o, a) in out_row.iter_mut().zip(acc_row.iter()) {
                    *o = a * inv_l;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    // These call `run_v2::<K>` directly for a specific kernel, bypassing
    // runtime feature detection, so both paths get exercised regardless of
    // which one `flash_attention_v2`'s public dispatch would pick on the
    // machine running the tests.
    use super::*;
    use crate::naive::naive_attention;
    use rand::{Rng, SeedableRng};

    fn random_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    fn check_kernel<K: Kernel + Sync>(seq_q: usize, seq_k: usize, d: usize, causal: bool) {
        let q = random_vec(seq_q * d, 1);
        let k = random_vec(seq_k * d, 2);
        let v = random_vec(seq_k * d, 3);
        let config = FlashAttentionConfig {
            block_size_q: 16,
            block_size_kv: 24,
            causal,
        };

        let mut out = vec![0.0f32; seq_q * d];
        run_v2::<K>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out);

        let mut out_naive = vec![0.0f32; seq_q * d];
        naive_attention(&q, &k, &v, seq_q, seq_k, d, causal, &mut out_naive);

        let diff = out
            .iter()
            .zip(out_naive.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-3, "diff {diff} too large");
    }

    #[test]
    fn scalar_kernel_matches_naive() {
        check_kernel::<ScalarKernel>(53, 71, 40, false);
        check_kernel::<ScalarKernel>(53, 71, 40, true);
        check_kernel::<ScalarKernel>(1, 1, 8, true);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_kernel_matches_naive() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        check_kernel::<Avx2Kernel>(53, 71, 40, false);
        check_kernel::<Avx2Kernel>(53, 71, 40, true);
        check_kernel::<Avx2Kernel>(1, 1, 8, true);
    }

    #[test]
    fn scalar_and_avx2_agree_with_each_other() {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let (seq_q, seq_k, d) = (65, 90, 48);
                let q = random_vec(seq_q * d, 4);
                let k = random_vec(seq_k * d, 5);
                let v = random_vec(seq_k * d, 6);
                let config = FlashAttentionConfig {
                    block_size_q: 32,
                    block_size_kv: 32,
                    causal: true,
                };

                let mut out_scalar = vec![0.0f32; seq_q * d];
                run_v2::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

                let mut out_avx2 = vec![0.0f32; seq_q * d];
                run_v2::<Avx2Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_avx2);

                let diff = out_scalar
                    .iter()
                    .zip(out_avx2.iter())
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(diff < 1e-3, "scalar/avx2 diff {diff} too large");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_kernel_matches_naive() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        check_kernel::<Avx512Kernel>(53, 71, 40, false);
        check_kernel::<Avx512Kernel>(53, 71, 40, true);
        check_kernel::<Avx512Kernel>(1, 1, 8, true);
    }

    #[test]
    fn scalar_and_avx512_agree_with_each_other() {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                let (seq_q, seq_k, d) = (65, 90, 48);
                let q = random_vec(seq_q * d, 4);
                let k = random_vec(seq_k * d, 5);
                let v = random_vec(seq_k * d, 6);
                let config = FlashAttentionConfig {
                    block_size_q: 32,
                    block_size_kv: 32,
                    causal: true,
                };

                let mut out_scalar = vec![0.0f32; seq_q * d];
                run_v2::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

                let mut out_avx512 = vec![0.0f32; seq_q * d];
                run_v2::<Avx512Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_avx512);

                let diff = out_scalar
                    .iter()
                    .zip(out_avx512.iter())
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(diff < 1e-3, "scalar/avx512 diff {diff} too large");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn neon_kernel_matches_naive() {
        check_kernel::<NeonKernel>(53, 71, 40, false);
        check_kernel::<NeonKernel>(53, 71, 40, true);
        check_kernel::<NeonKernel>(1, 1, 8, true);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn scalar_and_neon_agree_with_each_other() {
        let (seq_q, seq_k, d) = (65, 90, 48);
        let q = random_vec(seq_q * d, 4);
        let k = random_vec(seq_k * d, 5);
        let v = random_vec(seq_k * d, 6);
        let config = FlashAttentionConfig {
            block_size_q: 32,
            block_size_kv: 32,
            causal: true,
        };

        let mut out_scalar = vec![0.0f32; seq_q * d];
        run_v2::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

        let mut out_neon = vec![0.0f32; seq_q * d];
        run_v2::<NeonKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_neon);

        let diff = out_scalar
            .iter()
            .zip(out_neon.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-3, "scalar/neon diff {diff} too large");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn simd128_kernel_matches_naive() {
        check_kernel::<Simd128Kernel>(53, 71, 40, false);
        check_kernel::<Simd128Kernel>(53, 71, 40, true);
        check_kernel::<Simd128Kernel>(1, 1, 8, true);
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn scalar_and_simd128_agree_with_each_other() {
        let (seq_q, seq_k, d) = (65, 90, 48);
        let q = random_vec(seq_q * d, 4);
        let k = random_vec(seq_k * d, 5);
        let v = random_vec(seq_k * d, 6);
        let config = FlashAttentionConfig {
            block_size_q: 32,
            block_size_kv: 32,
            causal: true,
        };

        let mut out_scalar = vec![0.0f32; seq_q * d];
        run_v2::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

        let mut out_simd128 = vec![0.0f32; seq_q * d];
        run_v2::<Simd128Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_simd128);

        let diff = out_scalar
            .iter()
            .zip(out_simd128.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-3, "scalar/simd128 diff {diff} too large");
    }
}
