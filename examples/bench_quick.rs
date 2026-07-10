use flash_attention_cpu::{
    flash_attention_v1, flash_attention_v2, flash_attention_v3, naive::naive_attention,
    FlashAttentionConfig,
};
use rand::{Rng, SeedableRng};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

/// Resolves the commit this run should be tagged as in `--csv` mode: an
/// explicit `BENCH_COMMIT` env var (CI passes `$GITHUB_SHA`) takes
/// priority, falling back to a best-effort `git rev-parse` for zero-config
/// local use, and finally `"unknown"` if neither is available (e.g. running
/// from a source tarball with no `.git` directory).
fn resolve_commit() -> String {
    if let Ok(c) = std::env::var("BENCH_COMMIT") {
        return c;
    }
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    let csv_mode = std::env::args().any(|a| a == "--csv");
    // (variant, causal, seq_len, d_head, time_ms) — collected regardless of
    // output mode so the measurement loop below stays a single source of
    // truth; only the printing at the end differs.
    let mut rows: Vec<(&'static str, bool, usize, usize, f64)> = Vec::new();

    if !csv_mode {
        println!(
            "{:>7} {:>5} | {:>9} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
            "seq", "d", "naive_ms", "v1_ms", "v2_ms", "v3_ms", "v1c_ms", "v2c_ms", "v3c_ms"
        );
    }

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

        if csv_mode {
            rows.push(("naive", false, seq_len, d_head, naive_t * 1000.0));
            rows.push(("v1", false, seq_len, d_head, v1_t * 1000.0));
            rows.push(("v2", false, seq_len, d_head, v2_t * 1000.0));
            rows.push(("v3", false, seq_len, d_head, v3_t * 1000.0));
            rows.push(("v1", true, seq_len, d_head, v1c_t * 1000.0));
            rows.push(("v2", true, seq_len, d_head, v2c_t * 1000.0));
            rows.push(("v3", true, seq_len, d_head, v3c_t * 1000.0));
        } else {
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
    }

    if csv_mode {
        // Deliberately no date/time dependency (chrono/time) for a single
        // integer column in one example — Unix seconds sorts and diffs
        // fine, and this crate's only runtime dependency is `rayon`.
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let commit = resolve_commit();
        let os = std::env::consts::OS;
        let arch = std::env::consts::ARCH;
        let threads = rayon::current_num_threads();

        println!("timestamp,commit,os,arch,threads,variant,causal,seq_len,d_head,time_ms");
        for (variant, causal, seq_len, d_head, time_ms) in rows {
            println!(
                "{timestamp},{commit},{os},{arch},{threads},{variant},{causal},{seq_len},{d_head},{time_ms:.4}"
            );
        }
        return;
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
