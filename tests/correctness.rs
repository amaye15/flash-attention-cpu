use flash_attention_cpu::{
    flash_attention_multihead_v1, flash_attention_multihead_v2, flash_attention_multihead_v3,
    flash_attention_v1, flash_attention_v2, flash_attention_v3, naive::naive_attention,
    FlashAttentionConfig,
};
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, Debug)]
enum Variant {
    V1,
    V2,
    V3,
}

const ALL_VARIANTS: [Variant; 3] = [Variant::V1, Variant::V2, Variant::V3];

#[allow(clippy::too_many_arguments)]
fn run_flash(
    variant: Variant,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    d: usize,
    config: &FlashAttentionConfig,
    out: &mut [f32],
) {
    match variant {
        Variant::V1 => flash_attention_v1(q, k, v, seq_q, seq_k, d, config, out),
        Variant::V2 => flash_attention_v2(q, k, v, seq_q, seq_k, d, config, out),
        Variant::V3 => flash_attention_v3(q, k, v, seq_q, seq_k, d, config, out),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_multihead(
    variant: Variant,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    d: usize,
    config: &FlashAttentionConfig,
    out: &mut [f32],
) {
    match variant {
        Variant::V1 => {
            flash_attention_multihead_v1(q, k, v, batch, heads, seq_q, seq_k, d, config, out)
        }
        Variant::V2 => {
            flash_attention_multihead_v2(q, k, v, batch, heads, seq_q, seq_k, d, config, out)
        }
        Variant::V3 => {
            flash_attention_multihead_v3(q, k, v, batch, heads, seq_q, seq_k, d, config, out)
        }
    }
}

fn random_vec(n: usize, seed_shift: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE ^ seed_shift);
    (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn run_case(
    variant: Variant,
    seq_q: usize,
    seq_k: usize,
    d: usize,
    br: usize,
    bc: usize,
    causal: bool,
) {
    let q = random_vec(seq_q * d, 1);
    let k = random_vec(seq_k * d, 2);
    let v = random_vec(seq_k * d, 3);

    let mut out_flash = vec![0.0f32; seq_q * d];
    let mut out_naive = vec![0.0f32; seq_q * d];

    let config = FlashAttentionConfig {
        block_size_q: br,
        block_size_kv: bc,
        causal,
    };
    run_flash(
        variant,
        &q,
        &k,
        &v,
        seq_q,
        seq_k,
        d,
        &config,
        &mut out_flash,
    );
    naive_attention(&q, &k, &v, seq_q, seq_k, d, causal, &mut out_naive);

    let diff = max_abs_diff(&out_flash, &out_naive);
    assert!(
        diff < 1e-3,
        "{variant:?} seq_q={seq_q} seq_k={seq_k} d={d} br={br} bc={bc} causal={causal}: max abs diff {diff}"
    );
}

#[test]
fn matches_naive_exact_block_multiples() {
    for variant in ALL_VARIANTS {
        run_case(variant, 64, 64, 64, 32, 32, false);
        run_case(variant, 128, 128, 64, 64, 64, false);
    }
}

#[test]
fn matches_naive_ragged_sizes() {
    // Deliberately not multiples of the block size, to exercise remainder
    // handling in both the tiling loop and the SIMD tail loops.
    for variant in ALL_VARIANTS {
        run_case(variant, 17, 23, 32, 8, 8, false);
        run_case(variant, 1, 1, 16, 8, 8, false);
        run_case(variant, 1, 50, 33, 16, 16, false);
        run_case(variant, 50, 1, 33, 16, 16, false);
        run_case(variant, 97, 131, 17, 32, 40, false);
        run_case(variant, 200, 200, 128, 64, 128, false);
    }
}

#[test]
fn block_size_q_not_a_multiple_of_four() {
    // The register-blocked QK^T/PV loops (see `Kernel::dot4`/`axpy4`)
    // process query rows in groups of 4 with a scalar-row fallback for the
    // remainder. Every `block_size_q` used elsewhere in this file happens
    // to be a multiple of 4, so a full (non-final) query block never hits
    // that remainder path there — these values (3, 5, 6, 7) make *every*
    // full block hit it, exercising 3, 1, 2, and 3 leftover rows
    // respectively (mod 4).
    for variant in ALL_VARIANTS {
        for &br in &[3usize, 5, 6, 7] {
            run_case(variant, 23, 29, 24, br, 12, false);
            run_case(variant, 23, 29, 24, br, 12, true);
        }
    }
}

#[test]
fn matches_naive_causal() {
    for variant in ALL_VARIANTS {
        run_case(variant, 64, 64, 64, 32, 32, true);
        run_case(variant, 97, 97, 33, 16, 24, true);
        run_case(variant, 1, 1, 16, 8, 8, true);
        run_case(variant, 5, 5, 8, 64, 64, true); // block bigger than sequence
    }
}

#[test]
fn matches_naive_causal_asymmetric_seq_lens() {
    // seq_len_q < seq_len_k: e.g. decoding with a KV cache longer than the
    // freshly-computed query chunk. Query absolute position offset by
    // (seq_len_k - seq_len_q) is a common convention, but this crate's
    // causal mask assumes q and k share the same coordinate system
    // (position i can see keys <= i), so keep seq_len_q <= seq_len_k with
    // matching indices here.
    for variant in ALL_VARIANTS {
        run_case(variant, 10, 40, 32, 4, 8, true);
    }
}

#[test]
fn single_block_bigger_than_everything() {
    for variant in ALL_VARIANTS {
        run_case(variant, 10, 10, 8, 1024, 1024, false);
        run_case(variant, 10, 10, 8, 1024, 1024, true);
    }
}

#[test]
fn multihead_matches_per_head_naive() {
    let (batch, heads, seq_q, seq_k, d) = (2usize, 3usize, 37usize, 50usize, 16usize);
    let q = random_vec(batch * heads * seq_q * d, 21);
    let k = random_vec(batch * heads * seq_k * d, 22);
    let v = random_vec(batch * heads * seq_k * d, 23);

    let config = FlashAttentionConfig {
        block_size_q: 8,
        block_size_kv: 12,
        causal: true,
    };

    for variant in ALL_VARIANTS {
        let mut out = vec![0.0f32; batch * heads * seq_q * d];
        run_multihead(
            variant, &q, &k, &v, batch, heads, seq_q, seq_k, d, &config, &mut out,
        );

        let per_bh_q = seq_q * d;
        let per_bh_k = seq_k * d;
        for bh in 0..(batch * heads) {
            let q_bh = &q[bh * per_bh_q..(bh + 1) * per_bh_q];
            let k_bh = &k[bh * per_bh_k..(bh + 1) * per_bh_k];
            let v_bh = &v[bh * per_bh_k..(bh + 1) * per_bh_k];
            let out_bh = &out[bh * per_bh_q..(bh + 1) * per_bh_q];

            let mut want = vec![0.0f32; per_bh_q];
            naive_attention(q_bh, k_bh, v_bh, seq_q, seq_k, d, true, &mut want);

            let diff = max_abs_diff(out_bh, &want);
            assert!(diff < 1e-3, "{variant:?} head {bh}: max abs diff {diff}");
        }
    }
}

#[test]
fn causal_first_query_equals_first_value_row() {
    // Under causal masking, query position 0 can only attend to key 0, so
    // softmax degenerates to a single weight of 1.0 and the output must
    // exactly equal V's row 0 — independent of block size.
    let (seq_len, d) = (256, 64);
    let q = random_vec(seq_len * d, 30);
    let k = random_vec(seq_len * d, 31);
    let v = random_vec(seq_len * d, 32);

    for variant in ALL_VARIANTS {
        for (br, bc) in [(64usize, 128usize), (32, 32), (1024, 1024), (1, 1)] {
            let mut out = vec![0.0f32; seq_len * d];
            let config = FlashAttentionConfig {
                block_size_q: br,
                block_size_kv: bc,
                causal: true,
            };
            run_flash(variant, &q, &k, &v, seq_len, seq_len, d, &config, &mut out);
            let diff = max_abs_diff(&out[..d], &v[..d]);
            assert!(diff < 1e-5, "{variant:?} br={br} bc={bc}: diff {diff}");
        }
    }
}

#[test]
fn output_is_finite() {
    let (seq_q, seq_k, d) = (40, 40, 32);
    let q = random_vec(seq_q * d, 10);
    let k = random_vec(seq_k * d, 11);
    let v = random_vec(seq_k * d, 12);

    for variant in ALL_VARIANTS {
        let mut out = vec![0.0f32; seq_q * d];
        run_flash(
            variant,
            &q,
            &k,
            &v,
            seq_q,
            seq_k,
            d,
            &FlashAttentionConfig {
                causal: true,
                ..Default::default()
            },
            &mut out,
        );
        assert!(
            out.iter().all(|x| x.is_finite()),
            "{variant:?}: non-finite output"
        );
    }
}

/// All three variants implement the same mathematical attention operation
/// (they only differ in loop order / normalization scheduling / causal-skip
/// strategy), so given identical inputs they must agree with each other,
/// not just with the naive oracle independently.
#[test]
fn v1_v2_v3_mutually_agree() {
    for &(seq_q, seq_k, d, br, bc, causal) in &[
        (64usize, 64usize, 64usize, 32usize, 32usize, false),
        (97, 131, 17, 32, 40, true),
        (5, 5, 8, 64, 64, true), // block bigger than sequence
        (200, 200, 128, 64, 128, true),
    ] {
        let q = random_vec(seq_q * d, 1);
        let k = random_vec(seq_k * d, 2);
        let v = random_vec(seq_k * d, 3);
        let config = FlashAttentionConfig {
            block_size_q: br,
            block_size_kv: bc,
            causal,
        };

        let mut out_v1 = vec![0.0f32; seq_q * d];
        let mut out_v2 = vec![0.0f32; seq_q * d];
        let mut out_v3 = vec![0.0f32; seq_q * d];
        run_flash(
            Variant::V1,
            &q,
            &k,
            &v,
            seq_q,
            seq_k,
            d,
            &config,
            &mut out_v1,
        );
        run_flash(
            Variant::V2,
            &q,
            &k,
            &v,
            seq_q,
            seq_k,
            d,
            &config,
            &mut out_v2,
        );
        run_flash(
            Variant::V3,
            &q,
            &k,
            &v,
            seq_q,
            seq_k,
            d,
            &config,
            &mut out_v3,
        );

        let d_v1_v2 = max_abs_diff(&out_v1, &out_v2);
        let d_v1_v3 = max_abs_diff(&out_v1, &out_v3);
        // v2 and v3 perform byte-for-byte identical arithmetic per row (only
        // *other* tiles' independent work moves earlier in program order),
        // so hold them to a much tighter bound than the general
        // naive-oracle tolerance — a real pipeline bug (stale buffer,
        // off-by-one in `last_kj`) should fail loudly here.
        let d_v2_v3 = max_abs_diff(&out_v2, &out_v3);

        assert!(
            d_v1_v2 < 1e-3,
            "seq_q={seq_q} seq_k={seq_k} causal={causal}: v1 vs v2 diff {d_v1_v2}"
        );
        assert!(
            d_v1_v3 < 1e-3,
            "seq_q={seq_q} seq_k={seq_k} causal={causal}: v1 vs v3 diff {d_v1_v3}"
        );
        assert!(
            d_v2_v3 < 1e-5,
            "seq_q={seq_q} seq_k={seq_k} causal={causal}: v2 vs v3 diff {d_v2_v3}"
        );
    }
}
