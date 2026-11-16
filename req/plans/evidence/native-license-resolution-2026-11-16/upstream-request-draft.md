# Draft provenance request to MLX maintainers

Historical draft prepared 2026-11-16 for owner review. The owner submitted an adapted version
as [MLX issue #4902](https://github.com/ml-explore/mlx/issues/4902) at 16:54:53 UTC that day.
This file preserves the preparation draft, not the posted issue's exact wording.

Suggested title: Licensing provenance for error-function and CPU SIMD adaptations

We are preparing jmlx's first distribution of a macOS ARM64 native artifact containing the
`mlx-metal` 0.32.4 wheel and mlx-c. Our review uses MLX commit
`36969c094354086e15d2a327a9c50feb20b9cf11` and wheel SHA-256
`514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936`.
Could you point us to the applicable permission/provenance records for two adaptations?

1. [`mlx/backend/metal/kernels/erf.h`](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/kernels/erf.h)
   cites [Juffa's error-function answer](https://stackoverflow.com/a/35148199).
   The inverse-function coefficients also match
   [his inverse-error-function answer](https://stackoverflow.com/a/49743348), including
   the CPU version in
   [`mlx/backend/cpu/simd/math.h`](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/cpu/simd/math.h).
   The answer records currently declare CC BY-SA 4.0. We found Juffa's BSD-licensed
   NVIDIA forum samples, but those have different coefficients/evaluations. Is there an
   alternative grant covering the exact versions incorporated in MLX, or guidance on
   attribution and licensing of these adaptations in distributed dylibs and Metal kernels?
   We are not assuming that `expm1f.h`'s BSD notice covers the other functions.

2. The same CPU SIMD header traces its exponential and sine/cosine approximations through
   fastermath and NEON math to Cephes. We retain the immediate BSD/zlib notices and the
   [original Cephes permission statement](https://www.netlib.org/cephes/readme).
   Can you identify the licensed source distribution or permission record covering this
   particular provenance chain, including any additional attribution required downstream?

We are seeking clarification to package accurate notices and terms, not asserting a
license violation. If another rights holder must answer part of this, a pointer to the
appropriate provenance/contact route would help. We are holding our native publication
while these two questions remain open.
