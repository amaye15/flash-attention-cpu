# Security Policy

This crate is a CPU numeric kernel (tiled attention + hand-written SIMD) with
one dependency (`rayon`) — the realistic attack surface is mostly memory
safety in the `unsafe` SIMD code (`src/avx2.rs`, `src/avx512.rs`,
`src/neon.rs`, `src/simd128.rs`) and dependency supply-chain issues, not
classic web/network vulnerability classes.

## Reporting a vulnerability

Please **do not** open a public issue for a suspected security
vulnerability. Instead, use
[GitHub's private vulnerability reporting](https://github.com/amaye15/flash-attention-cpu/security/advisories/new)
for this repository. If that's not available, contact the maintainer
directly through their GitHub profile.

Include what you'd include in any good bug report: affected version,
reproduction steps or a minimal example, and the impact you'd expect
(crash, memory corruption, incorrect output used in a security-relevant
context, etc.).

## Scope

- Memory safety of the `unsafe` SIMD kernels and the FFI-free intrinsics
  they call.
- Panics/undefined behavior triggerable from safe, documented public API
  usage (i.e. not violating a documented `# Panics`/shape contract).
- Dependency vulnerabilities — tracked via `cargo deny check` against the
  [RustSec Advisory Database](https://rustsec.org/) in CI on every push, and
  via Dependabot for proactive update PRs.

Numerically incorrect output from a documented misuse of the API (wrong
shapes, etc. — which panic per the `# Panics` sections in the docs) is a
correctness bug, not a security issue — please file those as a normal
GitHub issue instead.

## Supported versions

This crate is pre-1.0 (`0.x`); only the latest published version on
crates.io is supported. There is no long-term-support branch.
