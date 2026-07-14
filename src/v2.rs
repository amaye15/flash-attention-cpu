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
#[cfg(target_arch = "x86_64")]
use crate::sse41::Sse41Kernel;
use rayon::prelude::*;

/// Single-head scaled dot-product attention via tiling + online softmax
/// ("flash attention"), dispatched at runtime/compile-time to the fastest
/// available SIMD kernel (AVX-512F/AVX2+FMA/SSE4.1 on x86_64, NEON on
/// aarch64, SIMD128 on wasm32 when built with that feature), and a portable
/// scalar kernel otherwise.
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
        if let Some(kernel) = Avx512Kernel::new() {
            run_v2(&kernel, q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
        if let Some(kernel) = Avx2Kernel::new() {
            run_v2(&kernel, q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
        if let Some(kernel) = Sse41Kernel::new() {
            run_v2(&kernel, q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        run_v2(
            &NeonKernel::new(),
            q,
            k,
            v,
            seq_len_q,
            seq_len_k,
            d_head,
            config,
            out,
        );
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        run_v2(
            &Simd128Kernel::new(),
            q,
            k,
            v,
            seq_len_q,
            seq_len_k,
            d_head,
            config,
            out,
        );
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    run_v2(
        &ScalarKernel::new(),
        q,
        k,
        v,
        seq_len_q,
        seq_len_k,
        d_head,
        config,
        out,
    );
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
    kernel: &K,
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

    out.par_chunks_mut(br * d_head).enumerate().for_each_init(
        // Rayon calls this once per work-stealing split rather than
        // once per query block, so scratch buffers get reused across
        // (typically many) blocks handled by the same split instead of
        // being freshly heap-allocated for every single one. Sized to
        // the configured `br`/`bc` (the max any block can be); each
        // call below slices down to that call's actual `this_br`.
        || {
            (
                vec![f32::NEG_INFINITY; br],
                vec![0.0f32; br],
                vec![0.0f32; br * d_head],
                vec![0.0f32; br * bc],
            )
        },
        |(m_buf, l_buf, acc_buf, scores_buf), (qi, out_block)| {
            let q_start = qi * br;
            let this_br = out_block.len() / d_head;
            let q_block = &q[q_start * d_head..(q_start + this_br) * d_head];

            // Running online-softmax state for this query block.
            let m = &mut m_buf[..this_br]; // running row max
            let l = &mut l_buf[..this_br]; // running row sum (of exp)
            let acc = &mut acc_buf[..this_br * d_head]; // unnormalized output accumulator
            m.fill(f32::NEG_INFINITY);
            l.fill(0.0);
            acc.fill(0.0);

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
                let scores_slice = &mut scores_buf[..this_br * this_bc];

                // S_ij = scale * Q_i . K_j  (Q/K rows are contiguous, so this
                // is a plain dot product — no transpose needed). Three-tier
                // register blocking: the bulk of the tile goes through
                // `dot4x4` (4 query rows x 4 key rows, both operands' loads
                // shared — see its docs for the measured speedup over
                // `dot4` alone), leftover columns (`this_bc % 4`) for a full
                // row-group fall back to `dot4`, and leftover rows
                // (`this_br % 4`) fall back to the scalar `dot`.
                let mut i = 0;
                while i + 4 <= this_br {
                    let q0 = &q_block[i * d_head..(i + 1) * d_head];
                    let q1 = &q_block[(i + 1) * d_head..(i + 2) * d_head];
                    let q2 = &q_block[(i + 2) * d_head..(i + 3) * d_head];
                    let q3 = &q_block[(i + 3) * d_head..(i + 4) * d_head];
                    let qg = [q0, q1, q2, q3];

                    let mut j = 0;
                    while j + 4 <= this_bc {
                        let k0 = &k_block[j * d_head..(j + 1) * d_head];
                        let k1 = &k_block[(j + 1) * d_head..(j + 2) * d_head];
                        let k2 = &k_block[(j + 2) * d_head..(j + 3) * d_head];
                        let k3 = &k_block[(j + 3) * d_head..(j + 4) * d_head];
                        let s = kernel.dot4x4(qg, [k0, k1, k2, k3]);
                        for r in 0..4 {
                            for c in 0..4 {
                                scores_slice[(i + r) * this_bc + (j + c)] = s[r][c] * scale;
                            }
                        }
                        j += 4;
                    }
                    while j < this_bc {
                        let kj_row = &k_block[j * d_head..(j + 1) * d_head];
                        let [s0, s1, s2, s3] = kernel.dot4(q0, q1, q2, q3, kj_row);
                        scores_slice[i * this_bc + j] = s0 * scale;
                        scores_slice[(i + 1) * this_bc + j] = s1 * scale;
                        scores_slice[(i + 2) * this_bc + j] = s2 * scale;
                        scores_slice[(i + 3) * this_bc + j] = s3 * scale;
                        j += 1;
                    }
                    i += 4;
                }
                while i < this_br {
                    let qi_row = &q_block[i * d_head..(i + 1) * d_head];
                    let s_row = &mut scores_slice[i * this_bc..(i + 1) * this_bc];
                    for (s, kj_row) in s_row.iter_mut().zip(k_block.chunks_exact(d_head)) {
                        *s = kernel.dot(qi_row, kj_row) * scale;
                    }
                    i += 1;
                }

                // Only touch the mask if this tile actually straddles the
                // diagonal; fully-visible tiles skip it entirely.
                if config.causal && k_start + this_bc - 1 > q_start {
                    for i in 0..this_br {
                        let global_i = q_start + i;
                        // Smallest j such that k_start + j > global_i,
                        // clamped to [0, this_bc]. The mask condition is
                        // monotonic in j for a fixed row, so everything from
                        // this cutoff onward is masked — one arithmetic
                        // computation + a slice fill instead of a branch per
                        // element.
                        let cutoff = if global_i + 1 >= k_start {
                            (global_i + 1 - k_start).min(this_bc)
                        } else {
                            0
                        };
                        let s_row = &mut scores_slice[i * this_bc..(i + 1) * this_bc];
                        s_row[cutoff..].fill(f32::NEG_INFINITY);
                    }
                }

                // Online softmax update, 4 rows at a time via `max_reduce4`/
                // `sub_exp_sum_inplace4`: row-max and subtract+exp+sum
                // reductions are interleaved across 4 independent chains
                // instead of processed one row at a time — see
                // `Kernel::max_reduce4`'s docs for why a single row's
                // reduction doesn't have enough independent work on its own
                // to hide latency. The rest (correction, `l` update,
                // `scale_inplace`) stays per-row afterward: already
                // independent and throughput-bound, nothing to interleave.
                // Leaves the PV accumulation itself to the blocked pass
                // below — `scores_slice` still holds each row's exp'd
                // probabilities afterward, unchanged from before.
                let mut i = 0;
                while i + 4 <= this_br {
                    let mut chunks =
                        scores_slice[i * this_bc..(i + 4) * this_bc].chunks_exact_mut(this_bc);
                    let s0 = chunks.next().unwrap();
                    let s1 = chunks.next().unwrap();
                    let s2 = chunks.next().unwrap();
                    let s3 = chunks.next().unwrap();

                    let block_max = kernel.max_reduce4([&*s0, &*s1, &*s2, &*s3]);
                    let new_m: [f32; 4] = std::array::from_fn(|r| m[i + r].max(block_max[r]));

                    let block_sum = kernel.sub_exp_sum_inplace4([s0, s1, s2, s3], new_m);

                    for r in 0..4 {
                        let correction = (m[i + r] - new_m[r]).exp();
                        l[i + r] = correction * l[i + r] + block_sum[r];
                        let acc_row = &mut acc[(i + r) * d_head..(i + r + 1) * d_head];
                        kernel.scale_inplace(acc_row, correction);
                        m[i + r] = new_m[r];
                    }
                    i += 4;
                }
                while i < this_br {
                    let s_row = &mut scores_slice[i * this_bc..(i + 1) * this_bc];
                    let block_max = kernel.max_reduce(s_row);
                    let new_m = m[i].max(block_max);

                    let block_sum = kernel.sub_exp_sum_inplace(s_row, new_m);
                    let correction = (m[i] - new_m).exp();

                    l[i] = correction * l[i] + block_sum;

                    let acc_row = &mut acc[i * d_head..(i + 1) * d_head];
                    kernel.scale_inplace(acc_row, correction);

                    m[i] = new_m;
                    i += 1;
                }

                // PV accumulation, 4 rows at a time via `pv4`: each
                // `d_head`-chunk's accumulator registers stay resident
                // across the whole `this_bc` sweep instead of round-tripping
                // through memory once per V row — see `Kernel::pv4`'s docs
                // for the measured speedup. Rows are fully independent here
                // (each only needs its own just-finished softmax state
                // above), so grouping them by 4 changes nothing but the
                // order this work happens in.
                let mut i = 0;
                while i + 4 <= this_br {
                    let mut chunks = acc[i * d_head..(i + 4) * d_head].chunks_exact_mut(d_head);
                    let d0 = chunks.next().unwrap();
                    let d1 = chunks.next().unwrap();
                    let d2 = chunks.next().unwrap();
                    let d3 = chunks.next().unwrap();

                    let p0 = &scores_slice[i * this_bc..(i + 1) * this_bc];
                    let p1 = &scores_slice[(i + 1) * this_bc..(i + 2) * this_bc];
                    let p2 = &scores_slice[(i + 2) * this_bc..(i + 3) * this_bc];
                    let p3 = &scores_slice[(i + 3) * this_bc..(i + 4) * this_bc];

                    kernel.pv4([d0, d1, d2, d3], v_block, [p0, p1, p2, p3]);
                    i += 4;
                }
                while i < this_br {
                    let s_row = &scores_slice[i * this_bc..(i + 1) * this_bc];
                    let acc_row = &mut acc[i * d_head..(i + 1) * d_head];
                    for (v_row, &p) in v_block.chunks_exact(d_head).zip(s_row.iter()) {
                        kernel.axpy(acc_row, v_row, p);
                    }
                    i += 1;
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
        },
    );
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

    fn check_kernel<K: Kernel + Sync>(
        kernel: &K,
        seq_q: usize,
        seq_k: usize,
        d: usize,
        causal: bool,
    ) {
        let q = random_vec(seq_q * d, 1);
        let k = random_vec(seq_k * d, 2);
        let v = random_vec(seq_k * d, 3);
        let config = FlashAttentionConfig {
            block_size_q: 16,
            block_size_kv: 24,
            causal,
        };

        let mut out = vec![0.0f32; seq_q * d];
        run_v2(kernel, &q, &k, &v, seq_q, seq_k, d, &config, &mut out);

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
        check_kernel(&ScalarKernel::new(), 53, 71, 40, false);
        check_kernel(&ScalarKernel::new(), 53, 71, 40, true);
        check_kernel(&ScalarKernel::new(), 1, 1, 8, true);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_kernel_matches_naive() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        check_kernel(&Avx2Kernel::new().unwrap(), 53, 71, 40, false);
        check_kernel(&Avx2Kernel::new().unwrap(), 53, 71, 40, true);
        check_kernel(&Avx2Kernel::new().unwrap(), 1, 1, 8, true);
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
                run_v2(
                    &ScalarKernel::new(),
                    &q,
                    &k,
                    &v,
                    seq_q,
                    seq_k,
                    d,
                    &config,
                    &mut out_scalar,
                );

                let mut out_avx2 = vec![0.0f32; seq_q * d];
                run_v2(
                    &Avx2Kernel::new().unwrap(),
                    &q,
                    &k,
                    &v,
                    seq_q,
                    seq_k,
                    d,
                    &config,
                    &mut out_avx2,
                );

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
    fn sse41_kernel_matches_naive() {
        if !is_x86_feature_detected!("sse4.1") {
            return;
        }
        check_kernel(&Sse41Kernel::new().unwrap(), 53, 71, 40, false);
        check_kernel(&Sse41Kernel::new().unwrap(), 53, 71, 40, true);
        check_kernel(&Sse41Kernel::new().unwrap(), 1, 1, 8, true);
    }

    #[test]
    fn scalar_and_sse41_agree_with_each_other() {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
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
                run_v2(
                    &ScalarKernel::new(),
                    &q,
                    &k,
                    &v,
                    seq_q,
                    seq_k,
                    d,
                    &config,
                    &mut out_scalar,
                );

                let mut out_sse41 = vec![0.0f32; seq_q * d];
                run_v2(
                    &Sse41Kernel::new().unwrap(),
                    &q,
                    &k,
                    &v,
                    seq_q,
                    seq_k,
                    d,
                    &config,
                    &mut out_sse41,
                );

                let diff = out_scalar
                    .iter()
                    .zip(out_sse41.iter())
                    .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(diff < 1e-3, "scalar/sse41 diff {diff} too large");
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_kernel_matches_naive() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        check_kernel(&Avx512Kernel::new().unwrap(), 53, 71, 40, false);
        check_kernel(&Avx512Kernel::new().unwrap(), 53, 71, 40, true);
        check_kernel(&Avx512Kernel::new().unwrap(), 1, 1, 8, true);
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
                run_v2(
                    &ScalarKernel::new(),
                    &q,
                    &k,
                    &v,
                    seq_q,
                    seq_k,
                    d,
                    &config,
                    &mut out_scalar,
                );

                let mut out_avx512 = vec![0.0f32; seq_q * d];
                run_v2(
                    &Avx512Kernel::new().unwrap(),
                    &q,
                    &k,
                    &v,
                    seq_q,
                    seq_k,
                    d,
                    &config,
                    &mut out_avx512,
                );

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
        check_kernel(&NeonKernel::new(), 53, 71, 40, false);
        check_kernel(&NeonKernel::new(), 53, 71, 40, true);
        check_kernel(&NeonKernel::new(), 1, 1, 8, true);
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
        run_v2(
            &ScalarKernel::new(),
            &q,
            &k,
            &v,
            seq_q,
            seq_k,
            d,
            &config,
            &mut out_scalar,
        );

        let mut out_neon = vec![0.0f32; seq_q * d];
        run_v2(
            &NeonKernel::new(),
            &q,
            &k,
            &v,
            seq_q,
            seq_k,
            d,
            &config,
            &mut out_neon,
        );

        let diff = out_scalar
            .iter()
            .zip(out_neon.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-3, "scalar/neon diff {diff} too large");
    }

    #[test]
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn simd128_kernel_matches_naive() {
        check_kernel(&Simd128Kernel::new(), 53, 71, 40, false);
        check_kernel(&Simd128Kernel::new(), 53, 71, 40, true);
        check_kernel(&Simd128Kernel::new(), 1, 1, 8, true);
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
        run_v2(
            &ScalarKernel::new(),
            &q,
            &k,
            &v,
            seq_q,
            seq_k,
            d,
            &config,
            &mut out_scalar,
        );

        let mut out_simd128 = vec![0.0f32; seq_q * d];
        run_v2(
            &Simd128Kernel::new(),
            &q,
            &k,
            &v,
            seq_q,
            seq_k,
            d,
            &config,
            &mut out_simd128,
        );

        let diff = out_scalar
            .iter()
            .zip(out_simd128.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-3, "scalar/simd128 diff {diff} too large");
    }
}
