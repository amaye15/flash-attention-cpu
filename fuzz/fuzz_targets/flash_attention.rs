//! Differential fuzzing: constructs small, always-shape-valid `q`/`k`/`v`
//! inputs (so the crate's documented shape-mismatch `assert!`s never fire —
//! those are intentional, not what this harness is hunting for) from
//! arbitrary bytes, at a randomized-but-internally-consistent magnitude
//! scale per case (well beyond the fixed uniform(-1, 1) unit test inputs),
//! and checks that `flash_attention_v1`/`_v2`/`_v3` agree with *each other*.
//! This targets two classes of bug the fixed-shape unit tests can't:
//! out-of-bounds/UB in the `unsafe` SIMD kernels under unusual (but
//! in-bounds) shapes, and algorithmic divergence between the three variants
//! under unusual magnitudes.
//!
//! Deliberately compared against each other, **not** against
//! [`naive_attention`]: an earlier version of this harness did that, and
//! immediately "found" that v1/v2/v3 (identically) disagree with naive at
//! moderately large magnitudes — root-caused to summation-order sensitivity
//! in the `Q.K` dot product: naive sums sequentially in plain scalar code,
//! the crate's `Kernel::dot` sums via a 2-way unrolled SIMD accumulator, and
//! floating-point addition isn't associative, so near-cancelling terms can
//! shift the result well past any fixed tolerance. That's inherent to
//! comparing *any* differently-ordered floating-point reduction against a
//! scalar reference, not a bug in this crate's tiling logic — v1/v2/v3 all
//! route through the identical `Kernel::dot` for a given build, so
//! comparing them to each other isolates genuine algorithmic divergence
//! (broken causal masking, a botched pipeline, wrong normalization timing)
//! without that false-positive source.
//!
//! See `tests/correctness.rs`'s `v1_v2_v3_mutually_agree` for the
//! fixed-shape version of this same idea; this extends it with a broad,
//! automatically-explored shape/magnitude space.

#![no_main]

use arbitrary::Arbitrary;
use flash_attention_cpu::{
    flash_attention_v1, flash_attention_v2, flash_attention_v3, FlashAttentionConfig,
};
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Case {
    seq_len_q: u8,
    seq_len_k: u8,
    d_head: u8,
    block_size_q: u8,
    block_size_kv: u8,
    causal: bool,
    magnitude_exp: u8,
    q_seed: u64,
    k_seed: u64,
    v_seed: u64,
}

/// Deterministic xorshift64 stream mapped into `[-magnitude, magnitude]` —
/// always finite by construction (built from bounded integer arithmetic,
/// never a raw float bit-pattern reinterpretation), so shape-mismatch and
/// NaN/Inf-from-input are never what a failure here is about.
fn scaled_vec(seed: u64, magnitude: f32, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mantissa = (state >> 40) & 0x00ff_ffff; // 24 bits, always finite
            let unit = mantissa as f32 / 0x00ff_ffff as f32; // [0, 1]
            (unit * 2.0 - 1.0) * magnitude // [-magnitude, magnitude]
        })
        .collect()
}

fn assert_agrees(name_a: &str, a: &[f32], name_b: &str, b: &[f32]) {
    for (x, y) in a.iter().zip(b.iter()) {
        match (x.is_finite(), y.is_finite()) {
            (true, true) => {
                let tol = 1e-1 * (x.abs().max(y.abs()) + 1.0);
                assert!(
                    (x - y).abs() < tol,
                    "{name_a} vs {name_b} diverged: {x} vs {y} (tol {tol})"
                );
            }
            (x_finite, y_finite) => assert_eq!(
                x_finite, y_finite,
                "{name_a} vs {name_b} finiteness mismatch: {x} vs {y}"
            ),
        }
    }
}

fuzz_target!(|case: Case| {
    // Small, always >=1 in every dimension: breadth of shapes/values is the
    // point here, not runtime, and 0-length inputs are already covered by
    // the crate's own unit/integration tests.
    let seq_len_q = (case.seq_len_q % 20) as usize + 1;
    let seq_len_k = (case.seq_len_k % 20) as usize + 1;
    let d_head = (case.d_head % 16) as usize + 1;
    let block_size_q = (case.block_size_q % 8) as usize + 1;
    let block_size_kv = (case.block_size_kv % 8) as usize + 1;

    // One random-but-shared scale per case (1 to 10^6), applied to q/k/v
    // alike — still exercises small-value and large-value regimes across
    // different fuzz iterations, without the pathological same-vector
    // cross-magnitude mixing described above.
    let magnitude = 10f32.powi((case.magnitude_exp % 7) as i32);

    let q = scaled_vec(case.q_seed, magnitude, seq_len_q * d_head);
    let k = scaled_vec(case.k_seed, magnitude, seq_len_k * d_head);
    let v = scaled_vec(case.v_seed, magnitude, seq_len_k * d_head);

    let config = FlashAttentionConfig {
        block_size_q,
        block_size_kv,
        causal: case.causal,
    };

    let mut out1 = vec![0.0f32; seq_len_q * d_head];
    flash_attention_v1(&q, &k, &v, seq_len_q, seq_len_k, d_head, &config, &mut out1);

    let mut out2 = vec![0.0f32; seq_len_q * d_head];
    flash_attention_v2(&q, &k, &v, seq_len_q, seq_len_k, d_head, &config, &mut out2);

    let mut out3 = vec![0.0f32; seq_len_q * d_head];
    flash_attention_v3(&q, &k, &v, seq_len_q, seq_len_k, d_head, &config, &mut out3);

    assert_agrees("v1", &out1, "v2", &out2);
    assert_agrees("v2", &out2, "v3", &out3);
});
