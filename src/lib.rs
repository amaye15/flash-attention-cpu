//! Pure-Rust, CPU-optimized Flash Attention.
//!
//! Implements three explicit tiled / online-softmax algorithm variants,
//! each adapted for CPU execution (cache hierarchy, SIMD, threads) rather
//! than GPU SRAM/tensor cores:
//!
//! - [`v1`]: FlashAttention-1 (Dao et al., 2022) — normalizes the output
//!   accumulator on every KV-block step; no causal tile-skip.
//! - [`v2`]: FlashAttention-2 (Dao, 2023) — outer loop over query blocks,
//!   a single deferred normalization after the whole KV sweep, and a
//!   causal early-exit that skips fully-future KV tiles outright.
//! - [`v3`]: this crate's same-core, instruction-level-parallelism analog
//!   of FlashAttention-3's compute/softmax overlap — v2's algorithm with
//!   the next KV tile's score computation software-pipelined ahead of the
//!   current tile's softmax+PV finish. This is **not** a port of FA3's
//!   actual mechanism (Hopper-specific warp specialization with
//!   asynchronous tensor-core pipelines, plus FP8 numerics) — that has no
//!   same-core CPU equivalent. See the README's Design section for the
//!   full comparison and honest caveats.
//!
//! All three share:
//! - **Tiling + online softmax**: `O(block_size)` extra memory instead of
//!   materializing the full `[seq_len_q, seq_len_k]` score matrix.
//! - **SIMD** for the dot products, softmax, and weighted-sum inner loops,
//!   including a hand-vectorized `exp`, tiered per target: on x86_64,
//!   AVX-512F (16 lanes) ahead of AVX2+FMA (8 lanes) ahead of scalar, all
//!   selected via runtime feature detection (`is_x86_feature_detected!`);
//!   on aarch64, NEON (4 lanes) selected unconditionally — it's part of the
//!   mandatory AArch64 baseline, so no runtime check is needed; on wasm32,
//!   SIMD128 (4 lanes) selected at compile time when built with
//!   `-C target-feature=+simd128` (there's no runtime feature-detection
//!   mechanism for WASM), scalar otherwise.
//! - **Rayon** data parallelism across query blocks and, for the batched
//!   entry points, across batch/heads.
//!
//! `flash_attention`/`flash_attention_multihead` are kept as aliases to the
//! v2 functions for backward compatibility; prefer the explicit
//! `flash_attention_v1`/`_v2`/`_v3` names to be clear about which
//! algorithmic tradeoffs you're getting.
//!
//! ```
//! use flash_attention_cpu::{flash_attention, FlashAttentionConfig};
//!
//! let (seq_len, d_head) = (128, 64);
//! let q = vec![0.0f32; seq_len * d_head];
//! let k = vec![0.0f32; seq_len * d_head];
//! let v = vec![0.0f32; seq_len * d_head];
//! let mut out = vec![0.0f32; seq_len * d_head];
//!
//! flash_attention(&q, &k, &v, seq_len, seq_len, d_head, &FlashAttentionConfig::default(), &mut out);
//! ```

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod common;
mod kernel;
pub mod naive;
#[cfg(target_arch = "aarch64")]
mod neon;
mod scalar;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd128;
pub mod v1;
pub mod v2;
pub mod v3;

pub use common::FlashAttentionConfig;
pub use v1::{flash_attention_multihead_v1, flash_attention_v1};
pub use v2::{flash_attention_multihead_v2, flash_attention_v2};
pub use v3::{flash_attention_multihead_v3, flash_attention_v3};

// Backward-compatible aliases: prior to v1/v2/v3, the crate exposed a
// single unversioned algorithm — the one now called v2 (deferred
// normalization + causal tile-skip).
pub use v2::flash_attention_multihead_v2 as flash_attention_multihead;
pub use v2::flash_attention_v2 as flash_attention;
