use flash_attention_cpu::{
    flash_attention, flash_attention_multihead, flash_attention_v1, flash_attention_v2,
    flash_attention_v3, FlashAttentionConfig,
};
use rand::{Rng, SeedableRng};

fn random_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn main() {
    // --- Single-head, non-causal ---
    let (seq_len, d_head) = (256, 64);
    let q = random_vec(seq_len * d_head, 1);
    let k = random_vec(seq_len * d_head, 2);
    let v = random_vec(seq_len * d_head, 3);
    let mut out = vec![0.0f32; seq_len * d_head];

    flash_attention(
        &q,
        &k,
        &v,
        seq_len,
        seq_len,
        d_head,
        &FlashAttentionConfig::default(),
        &mut out,
    );
    println!("single-head output[0][0..4] = {:?}", &out[..4]);

    // --- Causal (autoregressive) self-attention ---
    let config_causal = FlashAttentionConfig {
        causal: true,
        ..Default::default()
    };
    flash_attention(
        &q,
        &k,
        &v,
        seq_len,
        seq_len,
        d_head,
        &config_causal,
        &mut out,
    );
    println!("causal output[0][0..4]      = {:?}", &out[..4]);

    // --- Batched multi-head: [batch, heads, seq, d_head] contiguous ---
    let (batch, heads) = (4, 8);
    let mh_q = random_vec(batch * heads * seq_len * d_head, 4);
    let mh_k = random_vec(batch * heads * seq_len * d_head, 5);
    let mh_v = random_vec(batch * heads * seq_len * d_head, 6);
    let mut mh_out = vec![0.0f32; batch * heads * seq_len * d_head];

    flash_attention_multihead(
        &mh_q,
        &mh_k,
        &mh_v,
        batch,
        heads,
        seq_len,
        seq_len,
        d_head,
        &config_causal,
        &mut mh_out,
    );
    println!(
        "multihead: batch={batch} heads={heads} seq_len={seq_len} d_head={d_head}, output len={}",
        mh_out.len()
    );

    // Custom block sizes: tune for your d_head / cache sizes / causal-skip granularity.
    let config_tuned = FlashAttentionConfig {
        block_size_q: 32,
        block_size_kv: 32,
        causal: true,
    };
    flash_attention(
        &q,
        &k,
        &v,
        seq_len,
        seq_len,
        d_head,
        &config_tuned,
        &mut out,
    );
    println!("tuned-block output[0][0..4] = {:?}", &out[..4]);

    // --- Explicit version selection: flash_attention/_multihead alias v2
    // for backward compatibility, but v1/v2/v3 can be called directly to be
    // explicit about which algorithmic tradeoffs you're getting. ---
    flash_attention_v1(
        &q,
        &k,
        &v,
        seq_len,
        seq_len,
        d_head,
        &config_causal,
        &mut out,
    );
    println!("v1 output[0][0..4]          = {:?}", &out[..4]);
    flash_attention_v2(
        &q,
        &k,
        &v,
        seq_len,
        seq_len,
        d_head,
        &config_causal,
        &mut out,
    );
    println!("v2 output[0][0..4]          = {:?}", &out[..4]);
    flash_attention_v3(
        &q,
        &k,
        &v,
        seq_len,
        seq_len,
        d_head,
        &config_causal,
        &mut out,
    );
    println!("v3 output[0][0..4]          = {:?}", &out[..4]);
}
