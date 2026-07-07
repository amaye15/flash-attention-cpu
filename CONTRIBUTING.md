# Contributing

Thanks for considering a contribution. This project is small enough that
there's no formal process — open an issue or PR and it'll get reviewed.

## Building and testing

```bash
cargo test                                      # native — whatever SIMD tier your host has
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

This crate has SIMD kernels for four targets (`src/avx2.rs`, `src/avx512.rs`,
`src/neon.rs`, `src/simd128.rs`), so `cargo test` on your machine only
exercises whichever one matches your host architecture — the others are
`#[cfg(target_arch = "...")]`-gated out entirely. Before relying on a change
to one of the non-native kernels, cross-check it too:

```bash
# x86_64 (AVX2/AVX-512F) — type-checks + clippy even from a non-x86_64 host;
# only actually *executes* if your host is x86_64.
rustup target add x86_64-apple-darwin  # or x86_64-unknown-linux-gnu
cargo clippy --target x86_64-apple-darwin --all-targets --all-features -- -D warnings
cargo test --target x86_64-apple-darwin --release   # only runs on an x86_64 host

# wasm32 (SIMD128) — real execution via Node.js, not just a type-check
cargo install wasm-pack
rustup target add wasm32-unknown-unknown
wasm-pack test --node
```

`.cargo/config.toml` enables `target-feature=+simd128` for this repo's own
wasm32 builds/tests, so the `wasm-pack test` command above actually
exercises `Simd128Kernel`, not the scalar fallback — `tests/wasm_simd.rs`
has a guard test (`simd128_target_feature_is_actually_enabled`) that fails
loudly if that ever silently stops being true.

## Adding a new SIMD kernel

Each kernel implements the `Kernel` trait in `src/kernel.rs` — see that
file's doc comments for the full, current contract (it's grown across three
rounds of performance work, so trust the trait definition over any summary
here). Three families of primitives, in increasing implementation effort:

- **Scalar**: `dot`, `axpy`, `scale_inplace`, `max_reduce` — one element at
  a time, no SIMD-specific logic.
- **One-sided register-blocked** (`dot4`, `sub_exp_sum_inplace`): 4
  query-rows-at-once / a fused subtract+exp+sum pass, respectively.
- **Packed/row-blocked** (`dot4x4`, `pv4`, `max_reduce4`,
  `sub_exp_sum_inplace4`): two-sided register tiling for QK^T, a
  register-resident accumulator for PV, and 4-row-interleaved reductions for
  the softmax bookkeeping loop — see `src/kernel.rs`'s doc comments on each
  for why (in short: sharing operand loads across output rows/columns, and
  giving the CPU enough independent accumulator chains to hide FMA/reduction
  latency).

See any of the five existing kernels (`scalar.rs`/`avx2.rs`/`avx512.rs`/
`neon.rs`/`simd128.rs`) for the pattern — they're structurally identical
modulo lane width and FMA-argument-order conventions — and `src/avx2.rs`'s
module docs for the vectorized-`exp` algorithm all of them share
(range-reduction + degree-5 minimax polynomial + direct IEEE-754
exponent-bit reconstruction). Wire the new kernel into `v1.rs`/`v2.rs`/
`v3.rs`'s dispatch chains (all three, identically) and `src/lib.rs`'s module
declarations, matching the existing `#[cfg(...)]` pattern for whichever
target it's for.

## Algorithmic changes

`src/v1.rs`/`v2.rs`/`v3.rs`'s module docs explain what's supposed to differ
between the three variants (loop order, normalization timing, causal-skip
strategy) — read those before changing the tiling logic, since the whole
point of having three variants is that each one faithfully reflects a real
algorithmic distinction from the published papers, not an arbitrary
implementation choice. `tests/correctness.rs`'s `v1_v2_v3_mutually_agree`
test (and the `fuzz/` target below) exist specifically to catch the three
variants drifting apart when they shouldn't.

## Fuzzing

```bash
cargo install cargo-fuzz
cargo +nightly fuzz run flash_attention -- -max_total_time=60
```

`fuzz/fuzz_targets/flash_attention.rs` checks that `flash_attention_v1`/
`_v2`/`_v3` agree with *each other* (not the naive oracle — see that file's
module docs for why: comparing against a differently-summation-ordered
scalar reference at extreme magnitudes is a well-known source of
floating-point false positives, not a useful bug signal) across randomly
generated shapes and magnitudes. If you add a new code path to the tiling
logic, running this for a minute or two locally is a good sanity check
beyond the fixed-shape unit tests.

## MSRV

The minimum supported Rust version (currently 1.89, see `rust-version` in
`Cargo.toml`) is set by whichever dependency or language feature actually
needs it — right now that's the AVX-512F intrinsics in `src/avx512.rs`, not
this crate's own code. If you bump it, update `rust-version` in `Cargo.toml`
and the `msrv` job's toolchain pin in `.github/workflows/ci.yml` together,
and verify with:

```bash
rustup toolchain install <version>
rustup run <version> cargo check --lib
```

(`--lib` deliberately excludes `tests`/`benches`/`examples`: dev-dependencies
like `criterion` drift their own MSRV independently of this crate's actual
published surface.)

## Licenses and dependencies

New dependencies are checked by `cargo deny check` (config in `deny.toml`)
for license compatibility, security advisories, and duplicate/banned
crates — CI runs this on every PR via `EmbarkStudios/cargo-deny-action`.
