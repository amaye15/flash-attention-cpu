use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use flash_attention_cpu::{
    flash_attention, flash_attention_v1, flash_attention_v3, naive::naive_attention,
    FlashAttentionConfig,
};
use rand::{Rng, SeedableRng};

fn random_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn bench_attention(c: &mut Criterion) {
    let d_head = 64;
    let mut group = c.benchmark_group("attention");

    for &seq_len in &[128usize, 512, 2048] {
        let q = random_vec(seq_len * d_head, 1);
        let k = random_vec(seq_len * d_head, 2);
        let v = random_vec(seq_len * d_head, 3);
        let mut out = vec![0.0f32; seq_len * d_head];

        group.throughput(criterion::Throughput::Elements(
            (seq_len * seq_len * d_head) as u64,
        ));

        group.bench_with_input(BenchmarkId::new("naive", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                naive_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    seq_len,
                    seq_len,
                    d_head,
                    false,
                    &mut out,
                )
            })
        });

        let config = FlashAttentionConfig::default();
        let config_causal = FlashAttentionConfig {
            causal: true,
            ..Default::default()
        };

        // `flash` / `flash_causal` remain backed by v2 (the alias
        // `flash_attention`), kept as-is for continuity with any historical
        // baseline. `v1`/`v3` are benched explicitly alongside for the
        // three-way comparison.
        group.bench_with_input(BenchmarkId::new("flash", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                flash_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    seq_len,
                    seq_len,
                    d_head,
                    &config,
                    &mut out,
                )
            })
        });
        group.bench_with_input(
            BenchmarkId::new("flash_causal", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    flash_attention(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        seq_len,
                        seq_len,
                        d_head,
                        &config_causal,
                        &mut out,
                    )
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("v1", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                flash_attention_v1(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    seq_len,
                    seq_len,
                    d_head,
                    &config,
                    &mut out,
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("v1_causal", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                flash_attention_v1(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    seq_len,
                    seq_len,
                    d_head,
                    &config_causal,
                    &mut out,
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("v3", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                flash_attention_v3(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    seq_len,
                    seq_len,
                    d_head,
                    &config,
                    &mut out,
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("v3_causal", seq_len), &seq_len, |b, _| {
            b.iter(|| {
                flash_attention_v3(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    seq_len,
                    seq_len,
                    d_head,
                    &config_causal,
                    &mut out,
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_attention);
criterion_main!(benches);
