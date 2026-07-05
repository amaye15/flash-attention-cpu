//! Textbook scaled-dot-product attention that materializes the full
//! `[seq_len_q, seq_len_k]` score matrix. This is the O(n²)-memory baseline
//! that flash attention's tiling avoids; it exists here as a correctness
//! oracle and a benchmark comparison point, not for production use.

/// `q`: [seq_len_q, d_head], `k`/`v`: [seq_len_k, d_head], row-major.
/// `out`: [seq_len_q, d_head].
#[allow(clippy::too_many_arguments)]
pub fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len_q: usize,
    seq_len_k: usize,
    d_head: usize,
    causal: bool,
    out: &mut [f32],
) {
    assert_eq!(q.len(), seq_len_q * d_head);
    assert_eq!(k.len(), seq_len_k * d_head);
    assert_eq!(v.len(), seq_len_k * d_head);
    assert_eq!(out.len(), seq_len_q * d_head);

    let scale = 1.0 / (d_head as f32).sqrt();
    // The O(seq_len_q * seq_len_k) allocation flash attention is designed to avoid.
    let mut scores = vec![0.0f32; seq_len_q * seq_len_k];

    for i in 0..seq_len_q {
        let qi = &q[i * d_head..(i + 1) * d_head];
        let row = &mut scores[i * seq_len_k..(i + 1) * seq_len_k];
        for j in 0..seq_len_k {
            if causal && j > i {
                row[j] = f32::NEG_INFINITY;
                continue;
            }
            let kj = &k[j * d_head..(j + 1) * d_head];
            row[j] = dot(qi, kj) * scale;
        }
    }

    for i in 0..seq_len_q {
        let row = &mut scores[i * seq_len_k..(i + 1) * seq_len_k];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in row.iter_mut() {
            *s = (*s - m).exp();
            sum += *s;
        }
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for s in row.iter_mut() {
            *s *= inv_sum;
        }
    }

    for i in 0..seq_len_q {
        let row = &scores[i * seq_len_k..(i + 1) * seq_len_k];
        let out_row = &mut out[i * d_head..(i + 1) * d_head];
        out_row.fill(0.0);
        for j in 0..seq_len_k {
            let p = row[j];
            if p == 0.0 {
                continue;
            }
            let vj = &v[j * d_head..(j + 1) * d_head];
            for d in 0..d_head {
                out_row[d] += p * vj[d];
            }
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
