//! Shared config type and boilerplate reused by all three algorithm
//! variants (`v1`, `v2`, `v3`): shape assertions and the batched multi-head
//! dispatch pattern. Kept here rather than triplicated per-module.

use rayon::prelude::*;

/// Tuning knobs. Defaults keep the working set (`Q` block + `K`/`V` blocks +
/// the `Br x Bc` score tile) around a couple hundred KB, which comfortably
/// targets L2 residency on typical desktop/server cores; tune for your own
/// `d_head` and cache sizes if you need to squeeze further.
///
/// Shared by all three variants (`v1`, `v2`, `v3`) — the fields describe the
/// externally observable contract only; how each variant implements causal
/// masking internally (compute-and-mask vs. skip-ahead) is documented on
/// each variant's entry point instead.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttentionConfig {
    /// Query rows processed per tile (`Br`). Also the unit of work handed
    /// to each Rayon task.
    pub block_size_q: usize,
    /// Key/value rows processed per inner-loop tile (`Bc`).
    pub block_size_kv: usize,
    /// If true, query position `i` may only attend to key positions `<= i`
    /// (standard autoregressive self-attention).
    pub causal: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 128,
            causal: false,
        }
    }
}

/// Shared shape asserts for the single-head entry points. Returns `true` if
/// the caller should return immediately (a zero-sized dimension) without
/// running the tiling loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_shapes(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len_q: usize,
    seq_len_k: usize,
    d_head: usize,
    out: &[f32],
) -> bool {
    assert_eq!(q.len(), seq_len_q * d_head, "q shape mismatch");
    assert_eq!(k.len(), seq_len_k * d_head, "k shape mismatch");
    assert_eq!(v.len(), seq_len_k * d_head, "v shape mismatch");
    assert_eq!(out.len(), seq_len_q * d_head, "out shape mismatch");

    seq_len_q == 0 || seq_len_k == 0 || d_head == 0
}

/// Shared batched multi-head dispatch. Layout: `q` is
/// `[batch, heads, seq_len_q, d_head]`, `k`/`v` are
/// `[batch, heads, seq_len_k, d_head]`, all contiguous row-major. Splits
/// into per-(batch,head) slices with Rayon and calls `per_head` on each.
///
/// Parametrized over a plain `fn` pointer rather than a generic `Fn` type:
/// `flash_attention_v1/_v2/_v3` are free functions with no captured state,
/// an `fn` pointer is unconditionally `Send + Sync`, and this keeps the
/// helper compiled once rather than once per version.
pub(crate) type SingleHeadFn =
    fn(&[f32], &[f32], &[f32], usize, usize, usize, &FlashAttentionConfig, &mut [f32]);

#[allow(clippy::too_many_arguments)]
pub(crate) fn multihead_dispatch(
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
    per_head: SingleHeadFn,
) {
    let per_bh_q = seq_len_q * d_head;
    let per_bh_k = seq_len_k * d_head;
    assert_eq!(q.len(), batch * heads * per_bh_q);
    assert_eq!(k.len(), batch * heads * per_bh_k);
    assert_eq!(v.len(), batch * heads * per_bh_k);
    assert_eq!(out.len(), batch * heads * per_bh_q);

    q.par_chunks(per_bh_q.max(1))
        .zip(k.par_chunks(per_bh_k.max(1)))
        .zip(v.par_chunks(per_bh_k.max(1)))
        .zip(out.par_chunks_mut(per_bh_q.max(1)))
        .for_each(|(((q_bh, k_bh), v_bh), out_bh)| {
            per_head(
                q_bh, k_bh, v_bh, seq_len_q, seq_len_k, d_head, config, out_bh,
            );
        });
}
