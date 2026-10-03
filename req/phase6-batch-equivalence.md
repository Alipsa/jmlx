# Batch-step numerical equivalence

PR #30's `BatchStepEquivalenceTest` compares three left-padded rows with unequal
prompt lengths against independent unpadded rows, followed by a cached decode.
The initial absolute tolerance of `1e-4` failed for all six families.

Investigation used the staged pinned MLX 0.31.2 / mlx-c fba4470 runtime on an
Apple M5 Max, macOS ARM64. Temporary diagnostics measured every logit in every
row, for both prefill and decode (not just the first failing element).

| Family | Default prefill maximum | Default decode maximum |
| --- | ---: | ---: |
| Llama | 3.949e-4 | 5.180e-4 |
| Qwen2 | 4.322e-4 | 6.078e-4 |
| Mistral | 4.225e-4 | 4.285e-4 |
| Gemma | 9.063e-4 | 1.0965e-3 |
| Phi-3 | 3.539e-4 | 5.025e-4 |
| Mixtral | 3.536e-4 | 4.919e-4 |

With `MLX_ENABLE_TF32=0`, the maximum across every family, row and step fell to
`2.3842e-7`. This controlled comparison attributes the discrepancy to native
reduced precision: the Java graph, weights, masks, positions and cache logic
were unchanged. MLX documents reduced-precision float32 matrix operations on
supported hardware and the full-float32 opt-out in its
[numerical precision guide](https://ml-explore.github.io/mlx/build/html/usage/precision.html).
The pinned version's M5 behavior is also reported in
[upstream issue #3534](https://github.com/ml-explore/mlx/issues/3534).

The test uses an absolute `2e-3` tolerance for the default precision mode, with
headroom above the observed worst case, and preserves the original `1e-4` bound
when TF32 is explicitly disabled. It also checks greedy token identity for each
row and step. These bounds apply to these small committed float32 checkpoints;
they are not a general accuracy guarantee for arbitrary models or hardware.

Gradle now enforces both modes in separate test JVMs. The ordinary `test` task
sets `MLX_ENABLE_TF32=1`; `float32GoldenTest` sets it to `0`. Both run this suite
and are included in `check`/`build`:

```sh
./gradlew :jmlx-models:test --tests '*BatchStepEquivalenceTest'
./gradlew :jmlx-models:float32GoldenTest --tests '*BatchStepEquivalenceTest'
```
