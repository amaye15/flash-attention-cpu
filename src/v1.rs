//! FlashAttention-1 (Dao et al., 2022, Algorithm 1): tiling + online
//! softmax where the output accumulator is fully normalized and
//! re-normalized on *every* KV-block step, rather than deferred to a
//! single division at the end (contrast [`crate::v2`]). This mirrors the
//! original GPU kernel's need to write a consistent, normalized `O` back to
//! HBM after each step touching it — on CPU there's no such HBM
//! round-trip, so this is strictly extra non-matmul FLOPs kept here for a
//! faithful, honestly-slower v1 comparison point.
//!
//! Causal masking also has no early-exit here: fully-future KV tiles are
//! still computed and then masked (rather than skipped via a `break`, as
//! [`crate::v2`] and [`crate::v3`] do), matching the original paper, which
//! didn't have that optimization.

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

/// Single-head scaled dot-product attention via FlashAttention-1's
/// per-step-normalized tiling algorithm. See the module docs for how this
/// differs from [`crate::v2::flash_attention_v2`].
///
/// `q`: `[seq_len_q, d_head]`, `k`/`v`: `[seq_len_k, d_head]`, row-major.
/// `out`: `[seq_len_q, d_head]`, overwritten. Peak extra memory is
/// `O(block_size_q * (d_head + block_size_kv))`, independent of the full
/// sequence length.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_v1(
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
            run_v1::<Avx512Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            run_v1::<Avx2Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        run_v1::<NeonKernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        run_v1::<Simd128Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    run_v1::<ScalarKernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
}

/// Batched multi-head attention. See [`flash_attention_v1`] for the
/// single-head algorithm; this parallelizes over `batch * heads` and,
/// within each head, over query blocks.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_multihead_v1(
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
        flash_attention_v1,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_v1<K: Kernel + Sync>(
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

            // Running online-softmax state for this query block. Unlike
            // v2, `acc` is kept *normalized* between kv-block steps (it's
            // written back in a consistent state after every step, mirroring
            // the original paper's HBM round-trip), not deferred.
            let mut m = vec![f32::NEG_INFINITY; this_br]; // running row max
            let mut l = vec![0.0f32; this_br]; // running row sum (of exp)
            let mut acc = vec![0.0f32; this_br * d_head]; // normalized output, updated every step
            let mut scores = vec![0.0f32; this_br * bc]; // scratch S tile, reused per kv-block

            let num_kv_blocks = seq_len_k.div_ceil(bc);
            for kj in 0..num_kv_blocks {
                // No causal early-`break` here (contrast v2/v3): every
                // kv-block is visited, and fully-future tiles are masked to
                // all -inf below rather than skipped. This is the intended
                // "v1 has no causal-skip optimization" comparison point —
                // it costs a wasted K::dot pass over any fully-future tile.
                let k_start = kj * bc;
                let this_bc = bc.min(seq_len_k - k_start);

                let k_block = &k[k_start * d_head..(k_start + this_bc) * d_head];
                let v_block = &v[k_start * d_head..(k_start + this_bc) * d_head];
                let scores_slice = &mut scores[..this_br * this_bc];

                // S_ij = scale * Q_i . K_j
                for i in 0..this_br {
                    let qi_row = &q_block[i * d_head..(i + 1) * d_head];
                    let s_row = &mut scores_slice[i * this_bc..(i + 1) * this_bc];
                    for (s, kj_row) in s_row.iter_mut().zip(k_block.chunks_exact(d_head)) {
                        *s = unsafe { K::dot(qi_row, kj_row) } * scale;
                    }
                }

                if config.causal {
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

                // Online softmax update, fused with the per-step
                // normalize/denormalize round-trip and PV accumulation.
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

                    let old_l = l[i];
                    let new_l = correction * old_l + block_sum;

                    let acc_row = &mut acc[i * d_head..(i + 1) * d_head];
                    // Un-normalize the previous (already-normalized) output
                    // by its old sum, apply the running-max correction, in
                    // one fused scale — this is the extra work v2 defers.
                    unsafe { K::scale_inplace(acc_row, old_l * correction) };

                    for (v_row, &p) in v_block.chunks_exact(d_head).zip(s_row.iter()) {
                        unsafe { K::axpy(acc_row, v_row, p) };
                    }

                    // Re-normalize immediately, so `acc` is consistent
                    // (normalized) again before the next kv-block step.
                    let inv_new_l = if new_l > 0.0 { 1.0 / new_l } else { 0.0 };
                    unsafe { K::scale_inplace(acc_row, inv_new_l) };

                    m[i] = new_m;
                    l[i] = new_l;
                }
            }

            // `acc` is already normalized — unlike v2, no final division.
            for i in 0..this_br {
                let acc_row = &acc[i * d_head..(i + 1) * d_head];
                let out_row = &mut out_block[i * d_head..(i + 1) * d_head];
                out_row.copy_from_slice(acc_row);
            }
        });
}

#[cfg(test)]
mod tests {
    // These call `run_v1::<K>` directly for a specific kernel, bypassing
    // runtime feature detection, so both paths get exercised regardless of
    // which one `flash_attention_v1`'s public dispatch would pick on the
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
        run_v1::<K>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out);

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
                run_v1::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

                let mut out_avx2 = vec![0.0f32; seq_q * d];
                run_v1::<Avx2Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_avx2);

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
                run_v1::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

                let mut out_avx512 = vec![0.0f32; seq_q * d];
                run_v1::<Avx512Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_avx512);

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
        run_v1::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

        let mut out_neon = vec![0.0f32; seq_q * d];
        run_v1::<NeonKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_neon);

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
        run_v1::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

        let mut out_simd128 = vec![0.0f32; seq_q * d];
        run_v1::<Simd128Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_simd128);

        let diff = out_scalar
            .iter()
            .zip(out_simd128.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-3, "scalar/simd128 diff {diff} too large");
    }

    /// v1-specific: a fully-future tile (no causal skip) must still
    /// contribute nothing to the output — the normalize/denormalize
    /// round-trip on an all -inf-masked row should be a no-op up to
    /// floating-point rounding.
    #[test]
    fn fully_future_tile_is_a_noop() {
        check_kernel::<ScalarKernel>(4, 64, 16, true);
    }
}
