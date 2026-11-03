# Qwen3.5 references (Phase 7.6)

Opt-in CPython 3.12 tools for the reviewed
[Phase 7.6 plan](../../req/plans/phase7-6-plan.md). HF and MLX have independent hash-locked
macOS arm64 environments. They do not change the existing HF/image/tokenizer reference tools.
Ordinary Gradle checks verify committed digests; they never install or execute Python.

The current evidence covers executable reference probes and Java DeltaNet component fixtures.
Full Java model loading, generation, vision composition and real-artifact acceptance are pending.
Smoke probes, normalization diagnostics and benchmarks are not acceptance goldens.

```sh
./tools/qwen35-reference/install.sh hf
./tools/qwen35-reference/install.sh mlx
./gradlew verifyQwen35ReferenceGoldens
```

`hf/requirements.in` and `mlx/requirements.in` list direct pins; their locks include transitive
hashes. The selected wheel names, platform tags and hashes, installed distributions and source
hashes are in `evidence/{hf,mlx}-environment.json`. These records describe the executed environments.
The direct compute pins include Transformers 5.22.1, Torch 2.6.0 (HF), MLX/Metal 0.32.4,
mlx-vlm 0.6.8 and mlx-lm 0.32.2 (MLX). No remote Python model code is executed.

## Reference inputs and fixtures

The real-artifact inventories pin all files and tensor headers at these Hub revisions:

- `Qwen/Qwen3.5-0.8B`: `2fc06364715b967f1860aea9cf38778875588b17`.
- `mlx-community/Qwen3.8-27B-4bit`: `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.

Download them with `hf download --revision ... --local-dir .tier-b/phase7-6/artifacts/...`.
Keep large artifacts, captures, wheels and benchmark inputs in ignored `.tier-b/phase7-6/`.
The first session used temporary storage, which disappeared between sessions. Both large 27B
captures were then recreated from the archived executed probe in persistent ignored storage;
their raw hashes and complete JSON records reproduce the originals exactly. The binaries remain
external; paired attribution summaries and hashes are committed.

`probe_hf.py --output <json> --export-checkpoint <directory>` exports the seed-20261102 tiny HF
fixture and records text/image prefill and cached/fresh checks. Optional repeated
`--artifact-config <config.json>` arguments exercise the actual legacy configs. The fixture has
an untied 128-token vocabulary and a plain BPE tokenizer, with pad/BOS/EOS 0/1/2 and vision
start/end/image/video 118/119/120/121. Video execution remains unsupported.

`run_converters.py --source fixtures/hf --output-root <empty-directory> --record <json>` runs
both packages' actual converter entry points, saving float and affine outputs. Use the MLX venv
and `MLX_ENABLE_TF32=0`. The committed saved outputs are under `fixtures/`; saved-output inventories
and converter arguments/source hashes are in `evidence/converter-provenance.json`.
`verify_converted_weights.py --fixtures <fixtures-directory> --output <json>` checks the saved
float tensors against the source: exact selected norm offsets, convolution/patch axes and vision
inventory. It compares files; it does not create synthetic converter layouts.

`generate_delta.py --output <safetensors>` and `generate_net.py --output <safetensors>` use the HF
venv to generate independent CPU FLOAT32 component goldens under `goldens/`. Java comparisons use
the existing `full-float32` task with the unchanged `1e-4` bound. Regeneration is a manual reviewed
operation, never a way to accommodate a Java discrepancy.

## Probes and attribution

- `probe_config.py` records HF legacy/default behavior, output-gate metadata equivalence and
  partial rotary vectors. Java's planned validation is deliberately stricter than HF defaulting.
- `probe_delta.py` records actual device/use-kernel dispatch and native-op boundary cases.
  The custom mlx-vlm kernel and MLX's built-in op remain separate implementations.
- `prepare_real_inputs.py` records rendered prompts/IDs and slow-PIL patches, the 4096-row cap,
  image digest and a fast/slow comparison. All model comparisons consume the same saved patches.
- `reference_real_hf.py` records the 0.8B CPU FLOAT32 prefill/cached/fresh results. Its own
  cached/fresh text discrepancy reaches `3.19e-4`; failed `1e-4` measurements remain explicit.
  This is not a relaxation of strict tiny goldens or a declared Java real-artifact bound.
- `probe_reload.py --family vlm|lm --model <directory> --inputs <json> --output <json>` reloads
  saved models and records per-phase delta, convolution and gated-norm paths with source hashes.
  `--float32` upcasts only floating parameters, preserving packed integer weights.
- For 27B stock histories, add `--greedy-history --decode-steps 8 --capture-dir <external-directory>`.
  Diagnostic runs use the same inputs and `--teacher-history-from <stock.json>` in a fresh process.

`--normalization-diagnostic` is the plan's sole math-patch exception. It changes only the two
mlx-vlm Q/K RMSNorm epsilon arguments to `1e-6 / Dk`; the diff and source hashes are recorded.
Its outputs are separately labeled and **must never produce acceptance results or goldens**.
`compare_normalization.py` verifies paired inputs/capture hashes and reports Q/K, recurrent
output/state and logit contributions, greedy choices and margins. Java/reference remainder still
requires measurement after model implementation.

`executed-sources/` preserves the exact older reload probe used for recorded 27B results. Later
observation-hook changes do not silently rewrite the source provenance of those runs.

## Benchmarks

`benchmark_delta.py --inputs-dir <external-directory> --output <json>` evaluates chunked ops,
token recurrence, mlx-vlm's custom Metal kernel and MLX's built-in op at the planned shapes and
lengths. Failing candidates are ineligible; bounds are not loosened. `--layers 48 --shapes 27b`
measures aggregate layer cost. Inputs are saved and hashed for identical Java/reference data.
The first sample is the first candidate call after preparing inputs/reference qualification;
it is not evidence of an untouched system's first inference.

```sh
JMLX_DELTA_BENCHMARK_INPUT_DIR="$PWD/.tier-b/phase7-6/benchmark-inputs" \
  ./gradlew :jmlx-core:deltaBenchmark
```

Java records first calls per shape within its JVM separately from qualified warmed samples.
These are not untouched-system first-inference measurements.

The Java task supports `JMLX_DELTA_BENCHMARK_{SHAPES,LENGTHS,SAMPLES,LAYERS,OUTPUT}`. Shapes and
lengths are comma-separated. Its default output is `build/benchmarks/qwen35-delta-java.json`.
Add `-Pqwen35CountBindings=true` for a **separate** counting run; its timings are invalid. The
agent instruments generated downcall entry points only at runtime. Committed generated bindings
and arithmetic remain unchanged. Native API downcall counts are not Metal dispatch counts.

Samples include timestamped CPU/process/VM observations and evaluated allocator peaks. They do
not establish GPU utilization or exclude all competing GPU work. Initial exploratory results
are not substitutes for the planned committed-source benchmark and default review gate.

`provenance.py --record` explicitly records local source/fixture/evidence hashes after review;
without that flag it verifies them. The ordinary Gradle guard performs its own digest checks
without Python. This provenance inventory is not a claim that Phase 7.6 is complete.
