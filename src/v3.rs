//! This crate's same-core, instruction-level-parallelism analog of
//! FlashAttention-3's compute/softmax overlap: [`crate::v2`]'s algorithm
//! (outer loop over query blocks, deferred single normalization, causal
//! tile-skip) with the next KV tile's score computation (`QK^T`, a
//! FMA-heavy independent computation) software-pipelined one step ahead of
//! the current tile's online-softmax-update + PV finish (which has a
//! longer-latency `exp` in its dependency chain).
//!
//! **This is not a port of FlashAttention-3's actual mechanism.** FA3's
//! real headline feature is Hopper-specific warp specialization with
//! asynchronous tensor-core (WGMMA/TMA) pipelines, plus FP8 low-precision
//! numerics — both are GPU-hardware-specific concepts with no same-core
//! CPU equivalent. What's implemented here is a program-order restructuring
//! that gives the compiler's instruction scheduler and the CPU's
//! out-of-order execution window more independent work to interleave,
//! which is the closest same-core analog of "overlap softmax with the next
//! matmul." It is **not** a hardware guarantee of overlap: modern
//! out-of-order cores may already extract much of this from
//! [`crate::v2`]'s code as compiled, so the measured win over v2 can be
//! small, or compiler/CPU-dependent — see the README for real measurements
//! rather than an assumed speedup.

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

/// Single-head scaled dot-product attention via v2's algorithm with
/// software-pipelined score computation. See the module docs for what this
/// does and does not carry over from FlashAttention-3.
///
/// `q`: `[seq_len_q, d_head]`, `k`/`v`: `[seq_len_k, d_head]`, row-major.
/// `out`: `[seq_len_q, d_head]`, overwritten. Peak extra memory is
/// `O(block_size_q * (d_head + block_size_kv))`, independent of the full
/// sequence length — the score scratch is double-buffered (two tiles
/// resident at once for the pipeline), so this is roughly 2x v1/v2's tile
/// scratch, still independent of `seq_len`.
///
/// # Panics
///
/// Panics if `q.len() != seq_len_q * d_head`, `k.len() != seq_len_k * d_head`,
/// `v.len() != seq_len_k * d_head`, or `out.len() != seq_len_q * d_head`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_v3(
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
            run_v3::<Avx512Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            run_v3::<Avx2Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
        if is_x86_feature_detected!("sse4.1") {
            run_v3::<Sse41Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        run_v3::<NeonKernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        run_v3::<Simd128Kernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    run_v3::<ScalarKernel>(q, k, v, seq_len_q, seq_len_k, d_head, config, out);
}

/// Batched multi-head attention. See [`flash_attention_v3`] for the
/// single-head algorithm; this parallelizes over `batch * heads` and,
/// within each head, over query blocks.
///
/// # Panics
///
/// Panics if `q`/`k`/`v`/`out` don't match `batch * heads * seq_len_q *
/// d_head` (`q`/`out`) or `batch * heads * seq_len_k * d_head` (`k`/`v`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_multihead_v3(
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
        flash_attention_v3,
    );
}

/// Compute `S_kj = scale * Q_i . K_kj` for every row of the query block into
/// `dst[..this_br*this_bc]`, then mask in place if this tile straddles the
/// causal diagonal. Pure function of `(kj, q_block, k, v)` — no dependency
/// on any other tile's softmax state, which is what makes it safe to issue
/// ahead of the previous tile's [`finish_tile`].
#[allow(clippy::too_many_arguments)]
fn compute_tile<K: Kernel + Sync>(
    q_block: &[f32],
    k: &[f32],
    d_head: usize,
    this_br: usize,
    k_start: usize,
    this_bc: usize,
    scale: f32,
    causal: bool,
    q_start: usize,
    dst: &mut [f32],
) {
    let k_block = &k[k_start * d_head..(k_start + this_bc) * d_head];
    let scores_slice = &mut dst[..this_br * this_bc];

    // Three-tier register blocking — see `v2.rs`'s identical QK^T loop /
    // `Kernel::dot4x4`'s docs for why.
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
            let s = unsafe { K::dot4x4(qg, [k0, k1, k2, k3]) };
            for r in 0..4 {
                for c in 0..4 {
                    scores_slice[(i + r) * this_bc + (j + c)] = s[r][c] * scale;
                }
            }
            j += 4;
        }
        while j < this_bc {
            let kj_row = &k_block[j * d_head..(j + 1) * d_head];
            let [s0, s1, s2, s3] = unsafe { K::dot4(q0, q1, q2, q3, kj_row) };
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
            *s = unsafe { K::dot(qi_row, kj_row) } * scale;
        }
        i += 1;
    }

    // Only touch the mask if this tile actually straddles the diagonal;
    // fully-visible tiles skip it entirely. Fully-future tiles are never
    // reached at all — see `last_kj` in `run_v3`.
    if causal && k_start + this_bc - 1 > q_start {
        for i in 0..this_br {
            let global_i = q_start + i;
            // Smallest j such that k_start + j > global_i, clamped to
            // [0, this_bc]. The mask condition is monotonic in j for a
            // fixed row, so everything from this cutoff onward is masked —
            // one arithmetic computation + a slice fill instead of a branch
            // per element.
            let cutoff = if global_i + 1 >= k_start {
                (global_i + 1 - k_start).min(this_bc)
            } else {
                0
            };
            let s_row = &mut scores_slice[i * this_bc..(i + 1) * this_bc];
            s_row[cutoff..].fill(f32::NEG_INFINITY);
        }
    }
}

/// Finish a tile whose scores are already computed (and masked) in
/// `scores`: the online-softmax update fused with PV accumulation,
/// identical to v2's per-row finish math, mutating `scores` in place
/// (max-subtract + exp) exactly as v2 does — no extra allocation. Reads/
/// mutates the running `m`/`l`/`acc` state, which carries a genuine serial
/// dependency from tile to tile — unlike `compute_tile`, this cannot be
/// reordered across tiles.
#[allow(clippy::too_many_arguments)]
fn finish_tile<K: Kernel + Sync>(
    scores: &mut [f32],
    v_block: &[f32],
    d_head: usize,
    this_br: usize,
    this_bc: usize,
    m: &mut [f32],
    l: &mut [f32],
    acc: &mut [f32],
) {
    // Online softmax update, 4 rows at a time via `max_reduce4`/
    // `sub_exp_sum_inplace4`, leaving the PV accumulation to the blocked
    // pass below — see `v2.rs`'s identical split / `Kernel::max_reduce4`'s
    // docs for why this is still exactly equivalent to doing both in one
    // fused per-row pass.
    let mut i = 0;
    while i + 4 <= this_br {
        let mut chunks = scores[i * this_bc..(i + 4) * this_bc].chunks_exact_mut(this_bc);
        let s0 = chunks.next().unwrap();
        let s1 = chunks.next().unwrap();
        let s2 = chunks.next().unwrap();
        let s3 = chunks.next().unwrap();

        let block_max = unsafe { K::max_reduce4([&*s0, &*s1, &*s2, &*s3]) };
        let new_m: [f32; 4] = std::array::from_fn(|r| m[i + r].max(block_max[r]));

        let block_sum = unsafe { K::sub_exp_sum_inplace4([s0, s1, s2, s3], new_m) };

        for r in 0..4 {
            let correction = (m[i + r] - new_m[r]).exp();
            l[i + r] = correction * l[i + r] + block_sum[r];
            let acc_row = &mut acc[(i + r) * d_head..(i + r + 1) * d_head];
            unsafe { K::scale_inplace(acc_row, correction) };
            m[i + r] = new_m[r];
        }
        i += 4;
    }
    while i < this_br {
        let s_row = &mut scores[i * this_bc..(i + 1) * this_bc];
        let block_max = unsafe { K::max_reduce(s_row) };
        let new_m = m[i].max(block_max);

        let block_sum = unsafe { K::sub_exp_sum_inplace(s_row, new_m) };
        let correction = (m[i] - new_m).exp();

        l[i] = correction * l[i] + block_sum;

        let acc_row = &mut acc[i * d_head..(i + 1) * d_head];
        unsafe { K::scale_inplace(acc_row, correction) };

        m[i] = new_m;
        i += 1;
    }

    // PV accumulation, 4 rows at a time via `pv4` — see `v2.rs`'s identical
    // PV loop / `Kernel::pv4`'s docs for why.
    let mut i = 0;
    while i + 4 <= this_br {
        let mut chunks = acc[i * d_head..(i + 4) * d_head].chunks_exact_mut(d_head);
        let d0 = chunks.next().unwrap();
        let d1 = chunks.next().unwrap();
        let d2 = chunks.next().unwrap();
        let d3 = chunks.next().unwrap();

        let p0 = &scores[i * this_bc..(i + 1) * this_bc];
        let p1 = &scores[(i + 1) * this_bc..(i + 2) * this_bc];
        let p2 = &scores[(i + 2) * this_bc..(i + 3) * this_bc];
        let p3 = &scores[(i + 3) * this_bc..(i + 4) * this_bc];

        unsafe { K::pv4([d0, d1, d2, d3], v_block, [p0, p1, p2, p3]) };
        i += 4;
    }
    while i < this_br {
        let s_row = &scores[i * this_bc..(i + 1) * this_bc];
        let acc_row = &mut acc[i * d_head..(i + 1) * d_head];
        for (v_row, &p) in v_block.chunks_exact(d_head).zip(s_row.iter()) {
            unsafe { K::axpy(acc_row, v_row, p) };
        }
        i += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn run_v3<K: Kernel + Sync>(
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
        // call below slices down to that call's actual `this_br`/
        // `this_bc`.
        || {
            (
                vec![f32::NEG_INFINITY; br],
                vec![0.0f32; br],
                vec![0.0f32; br * d_head],
                [vec![0.0f32; br * bc], vec![0.0f32; br * bc]],
            )
        },
        |(m_buf, l_buf, acc_buf, buf), (qi, out_block)| {
            let q_start = qi * br;
            let this_br = out_block.len() / d_head;
            let q_block = &q[q_start * d_head..(q_start + this_br) * d_head];

            let m = &mut m_buf[..this_br];
            let l = &mut l_buf[..this_br];
            let acc = &mut acc_buf[..this_br * d_head];
            m.fill(f32::NEG_INFINITY);
            l.fill(0.0);
            acc.fill(0.0);

            let num_kv_blocks = seq_len_k.div_ceil(bc);

            // Hoist the causal bound before pipelining anything: this is
            // pure index arithmetic (never depends on a computed score), so
            // it can be resolved once, up front, mirroring v2's own break
            // condition verbatim. This means no fully-future tile's QK^T is
            // ever issued at all (strictly better than v1's compute-then-mask),
            // and it's what makes it safe to pipeline the remaining range.
            let mut last_kj = num_kv_blocks;
            if config.causal {
                for kj in 0..num_kv_blocks {
                    if kj * bc > q_start + this_br - 1 {
                        last_kj = kj;
                        break;
                    }
                }
            }

            if last_kj == 0 {
                // No visible keys at all for this query block (shouldn't
                // happen given seq_len_k >= 1 and causal always including
                // the diagonal tile, but guard the `last_kj - 1` subtraction
                // below defensively).
                for i in 0..this_br {
                    out_block[i * d_head..(i + 1) * d_head].fill(0.0);
                }
                return;
            }

            let tile_len = |kj: usize| bc.min(seq_len_k - kj * bc);

            // Prologue: issue tile 0's QK^T + mask.
            compute_tile::<K>(
                q_block,
                k,
                d_head,
                this_br,
                0,
                tile_len(0),
                scale,
                config.causal,
                q_start,
                &mut buf[0],
            );

            // Steady state: issue tile kj+1's independent score computation
            // before finishing tile kj's softmax+PV — no data dependency
            // between them, so this exposes independent work to the
            // compiler/CPU while `finish_tile`'s serial m/l/acc chain
            // resolves.
            for kj in 0..last_kj - 1 {
                let cur = kj & 1;
                let k_start_next = (kj + 1) * bc;
                let this_bc_next = tile_len(kj + 1);
                let (a, b) = buf.split_at_mut(1);
                let (cur_buf, nxt_buf) = if cur == 0 {
                    (&mut a[0], &mut b[0])
                } else {
                    (&mut b[0], &mut a[0])
                };
                compute_tile::<K>(
                    q_block,
                    k,
                    d_head,
                    this_br,
                    k_start_next,
                    this_bc_next,
                    scale,
                    config.causal,
                    q_start,
                    nxt_buf,
                );

                let this_bc_cur = tile_len(kj);
                let k_start_cur = kj * bc;
                let v_block = &v[k_start_cur * d_head..(k_start_cur + this_bc_cur) * d_head];
                finish_tile::<K>(
                    &mut cur_buf[..this_br * this_bc_cur],
                    v_block,
                    d_head,
                    this_br,
                    this_bc_cur,
                    m,
                    l,
                    acc,
                );
            }

            // Epilogue: finish the last tile (no successor to pipeline).
            let last = last_kj - 1;
            let this_bc_last = tile_len(last);
            let k_start_last = last * bc;
            let v_block = &v[k_start_last * d_head..(k_start_last + this_bc_last) * d_head];
            finish_tile::<K>(
                &mut buf[last & 1][..this_br * this_bc_last],
                v_block,
                d_head,
                this_br,
                this_bc_last,
                m,
                l,
                acc,
            );

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
    // These call `run_v3::<K>` directly for a specific kernel, bypassing
    // runtime feature detection, so both paths get exercised regardless of
    // which one `flash_attention_v3`'s public dispatch would pick on the
    // machine running the tests.
    use super::*;
    use crate::naive::naive_attention;
    use rand::{Rng, SeedableRng};

    fn random_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    fn check_kernel_cfg<K: Kernel + Sync>(
        seq_q: usize,
        seq_k: usize,
        d: usize,
        causal: bool,
        br: usize,
        bc: usize,
    ) {
        let q = random_vec(seq_q * d, 1);
        let k = random_vec(seq_k * d, 2);
        let v = random_vec(seq_k * d, 3);
        let config = FlashAttentionConfig {
            block_size_q: br,
            block_size_kv: bc,
            causal,
        };

        let mut out = vec![0.0f32; seq_q * d];
        run_v3::<K>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out);

        let mut out_naive = vec![0.0f32; seq_q * d];
        naive_attention(&q, &k, &v, seq_q, seq_k, d, causal, &mut out_naive);

        let diff = out
            .iter()
            .zip(out_naive.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-3, "diff {diff} too large (br={br} bc={bc})");
    }

    fn check_kernel<K: Kernel + Sync>(seq_q: usize, seq_k: usize, d: usize, causal: bool) {
        check_kernel_cfg::<K>(seq_q, seq_k, d, causal, 16, 24);
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
                run_v3::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

                let mut out_avx2 = vec![0.0f32; seq_q * d];
                run_v3::<Avx2Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_avx2);

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
        check_kernel::<Sse41Kernel>(53, 71, 40, false);
        check_kernel::<Sse41Kernel>(53, 71, 40, true);
        check_kernel::<Sse41Kernel>(1, 1, 8, true);
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
                run_v3::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

                let mut out_sse41 = vec![0.0f32; seq_q * d];
                run_v3::<Sse41Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_sse41);

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
                run_v3::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

                let mut out_avx512 = vec![0.0f32; seq_q * d];
                run_v3::<Avx512Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_avx512);

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
        run_v3::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

        let mut out_neon = vec![0.0f32; seq_q * d];
        run_v3::<NeonKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_neon);

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
        run_v3::<ScalarKernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_scalar);

        let mut out_simd128 = vec![0.0f32; seq_q * d];
        run_v3::<Simd128Kernel>(&q, &k, &v, seq_q, seq_k, d, &config, &mut out_simd128);

        let diff = out_scalar
            .iter()
            .zip(out_simd128.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff < 1e-3, "scalar/simd128 diff {diff} too large");
    }

    // Boundary cases for the pipeline's steady-state loop iteration count
    // (`last_kj`): 1 (straight prologue -> epilogue, loop runs zero times),
    // 2 (loop runs exactly once), and >=3 (loop runs multiple times).
    #[test]
    fn pipeline_last_kj_is_one() {
        // seq_k=8, bc=64 => a single kv block covers everything.
        check_kernel_cfg::<ScalarKernel>(10, 8, 16, false, 4, 64);
        check_kernel_cfg::<ScalarKernel>(10, 8, 16, true, 4, 64);
    }

    #[test]
    fn pipeline_last_kj_is_two() {
        // seq_k=48, bc=24 => exactly 2 kv blocks.
        check_kernel_cfg::<ScalarKernel>(48, 48, 16, false, 16, 24);
        check_kernel_cfg::<ScalarKernel>(48, 48, 16, true, 16, 24);
    }

    #[test]
    fn pipeline_last_kj_is_three_or_more() {
        // seq_k=71, bc=24 => 71.div_ceil(24) == 3 kv blocks.
        check_kernel::<ScalarKernel>(53, 71, 40, false);
        check_kernel::<ScalarKernel>(53, 71, 40, true);
    }
}
