use flash_attention_cpu::{
    flash_attention_v1, flash_attention_v2, flash_attention_v3, naive::naive_attention,
    FlashAttentionConfig,
};
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn random_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn time_it<F: FnMut()>(mut f: F, iters: u32) -> f64 {
    // one warm-up call (page faults, cache warm-up) excluded from timing
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn main() {
    println!(
        "{:>7} {:>5} | {:>9} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "seq", "d", "naive_ms", "v1_ms", "v2_ms", "v3_ms", "v1c_ms", "v2c_ms", "v3c_ms"
    );

    for &(seq_len, d_head) in &[(256, 64), (512, 64), (1024, 64), (2048, 64), (1024, 128)] {
        let q = random_vec(seq_len * d_head, 1);
        let k = random_vec(seq_len * d_head, 2);
        let v = random_vec(seq_len * d_head, 3);
        let mut out = vec![0.0f32; seq_len * d_head];

        let config = FlashAttentionConfig::default();
        let config_causal = FlashAttentionConfig {
            causal: true,
            ..Default::default()
        };

        let iters = if seq_len <= 512 { 10 } else { 3 };

        let naive_t = time_it(
            || naive_attention(&q, &k, &v, seq_len, seq_len, d_head, false, &mut out),
            iters,
        );
        let v1_t = time_it(
            || flash_attention_v1(&q, &k, &v, seq_len, seq_len, d_head, &config, &mut out),
            iters,
        );
        let v2_t = time_it(
            || flash_attention_v2(&q, &k, &v, seq_len, seq_len, d_head, &config, &mut out),
            iters,
        );
        let v3_t = time_it(
            || flash_attention_v3(&q, &k, &v, seq_len, seq_len, d_head, &config, &mut out),
            iters,
        );
        let v1c_t = time_it(
            || {
                flash_attention_v1(
                    &q,
                    &k,
                    &v,
                    seq_len,
                    seq_len,
                    d_head,
                    &config_causal,
                    &mut out,
                )
            },
            iters,
        );
        let v2c_t = time_it(
            || {
                flash_attention_v2(
                    &q,
                    &k,
                    &v,
                    seq_len,
                    seq_len,
                    d_head,
                    &config_causal,
                    &mut out,
                )
            },
            iters,
        );
        let v3c_t = time_it(
            || {
                flash_attention_v3(
                    &q,
                    &k,
                    &v,
                    seq_len,
                    seq_len,
                    d_head,
                    &config_causal,
                    &mut out,
                )
            },
            iters,
        );

        println!(
            "{:>7} {:>5} | {:>9.3} | {:>8.3} {:>8.3} {:>8.3} | {:>8.3} {:>8.3} {:>8.3}",
            seq_len,
            d_head,
            naive_t * 1000.0,
            v1_t * 1000.0,
            v2_t * 1000.0,
            v3_t * 1000.0,
            v1c_t * 1000.0,
            v2c_t * 1000.0,
            v3c_t * 1000.0,
        );
    }

    // Peak extra-memory comparison at a size where it actually matters.
    // v1/v2 use one score tile (`block_size_q * block_size_kv` f32s); v3
    // double-buffers it for the pipeline, so its scratch is ~2x — still
    // O(block_size), still independent of seq_len.
    let seq_len = 4096usize;
    let naive_scores_bytes = seq_len * seq_len * 4;
    let tile_bytes = FlashAttentionConfig::default().block_size_q
        * FlashAttentionConfig::default().block_size_kv
        * 4;
    println!(
        "\nAt seq_len={seq_len}: naive score-matrix allocation = {:.1} MB, v1/v2 tile scratch = {:.1} KB, v3 tile scratch = {:.1} KB (all independent of seq_len)",
        naive_scores_bytes as f64 / (1024.0 * 1024.0),
        tile_bytes as f64 / 1024.0,
        (tile_bytes * 2) as f64 / 1024.0,
    );
}
