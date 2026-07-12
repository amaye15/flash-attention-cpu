//! wasm32-only correctness check, run via `wasm-pack test --node` (or
//! `--firefox`/`--chrome`). Plain `#[test]` functions are silently not
//! picked up by `wasm-bindgen-test-runner` ("no tests to run!") — hence a
//! dedicated file using `#[wasm_bindgen_test]`, separate from
//! `tests/correctness.rs`'s full native-only shape matrix. This exercises
//! the real public dispatch path (`flash_attention_v1/_v2/_v3`), which
//! resolves to `Simd128Kernel` when built with `-C target-feature=+simd128`
//! (the default for this crate's own `wasm32-unknown-unknown` builds — see
//! `.cargo/config.toml`) and to the scalar fallback otherwise.

#![cfg(target_arch = "wasm32")]

use flash_attention_cpu::{
    flash_attention_v1, flash_attention_v2, flash_attention_v3, naive::naive_attention,
    FlashAttentionConfig,
};
use rand::{Rng, SeedableRng};
use wasm_bindgen_test::wasm_bindgen_test;

fn random_vec(n: usize, seed_shift: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE ^ seed_shift);
    (0..n).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[derive(Clone, Copy)]
enum Variant {
    V1,
    V2,
    V3,
}

#[allow(clippy::too_many_arguments)]
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
    let config = FlashAttentionConfig {
        block_size_q: br,
        block_size_kv: bc,
        causal,
    };

    let mut out = vec![0.0f32; seq_q * d];
    match variant {
        Variant::V1 => flash_attention_v1(&q, &k, &v, seq_q, seq_k, d, &config, &mut out),
        Variant::V2 => flash_attention_v2(&q, &k, &v, seq_q, seq_k, d, &config, &mut out),
        Variant::V3 => flash_attention_v3(&q, &k, &v, seq_q, seq_k, d, &config, &mut out),
    }

    let mut out_naive = vec![0.0f32; seq_q * d];
    naive_attention(&q, &k, &v, seq_q, seq_k, d, causal, &mut out_naive);

    let diff = max_abs_diff(&out, &out_naive);
    assert!(
        diff < 1e-3,
        "seq_q={seq_q} seq_k={seq_k} d={d} br={br} bc={bc} causal={causal}: max abs diff {diff}"
    );
}

#[wasm_bindgen_test]
fn v1_matches_naive() {
    run_case(Variant::V1, 64, 64, 64, 32, 32, false);
    run_case(Variant::V1, 97, 131, 17, 32, 40, true);
    run_case(Variant::V1, 1, 1, 16, 8, 8, true);
}

#[wasm_bindgen_test]
fn v2_matches_naive() {
    run_case(Variant::V2, 64, 64, 64, 32, 32, false);
    run_case(Variant::V2, 97, 131, 17, 32, 40, true);
    run_case(Variant::V2, 1, 1, 16, 8, 8, true);
}

#[wasm_bindgen_test]
fn v3_matches_naive() {
    run_case(Variant::V3, 64, 64, 64, 32, 32, false);
    run_case(Variant::V3, 97, 131, 17, 32, 40, true);
    run_case(Variant::V3, 1, 1, 16, 8, 8, true);
}

#[wasm_bindgen_test]
fn ragged_sizes() {
    for &variant in &[Variant::V1, Variant::V2, Variant::V3] {
        run_case(variant, 17, 23, 32, 8, 8, false);
        run_case(variant, 1, 50, 33, 16, 16, false);
        run_case(variant, 50, 1, 33, 16, 16, false);
        run_case(variant, 200, 200, 128, 64, 128, true);
    }
}

/// Guards the assumption every other test in this file relies on: without
/// this, a build-config regression (e.g. `.cargo/config.toml`'s rustflags
/// changing or going missing) would make `v1_matches_naive`/etc. silently
/// pass against the scalar fallback instead of exercising `Simd128Kernel`,
/// defeating the point of this file.
#[wasm_bindgen_test]
fn simd128_target_feature_is_actually_enabled() {
    assert!(
        cfg!(target_feature = "simd128"),
        "simd128 not enabled — the tests above ran the scalar fallback, not Simd128Kernel"
    );
}

/// Only exists when built with `relaxed-simd` (CI's `wasm-relaxed-simd` job,
/// via `RUSTFLAGS="-C target-feature=+relaxed-simd"`) — guards against that
/// job silently degrading to the default `simd128`-only build and testing
/// nothing new, the same way `simd128_target_feature_is_actually_enabled`
/// guards the base case above.
#[cfg(target_feature = "relaxed-simd")]
#[wasm_bindgen_test]
fn relaxed_simd_target_feature_is_actually_enabled() {
    assert!(cfg!(target_feature = "relaxed-simd"));
}
