# Phase 7.6 — Reference findings

## 2026-11-02 planning pass: metadata and source inspection

This is a source/metadata inspection, not an executed neural reference. No model weights were
downloaded, no converter was run and no logits were generated. The detailed
[implementation plan](phase7-6-plan.md) requires those checks in WP0 before production code.

### Artifact revisions

| Artifact | Hub revision | Index facts |
| --- | --- | --- |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 488 tensor keys, one shard, index `total_size` 1,746,882,752 B |
| `mlx-community/Qwen3.8-27B-4bit` | `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | 2180 tensor keys, three shards, index `total_size` 16,054,262,240 B |

These counts and sizes come from the downloaded index JSON, not verified shard headers or
memory measurements. Both model cards declare Apache-2.0. Pinned configuration sources:
[0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json),
[27B](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/blob/10c35caafbb80f7dc6a7a432cdd11af10a6d4818/config.json).

| Metadata file | 0.8B SHA-256 | 27B SHA-256 |
| --- | --- | --- |
| `config.json` | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` | `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85` |
| `preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `model.safetensors.index.json` | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` | `13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8` |

The configs identify `qwen3_5` / `Qwen3_5ForConditionalGeneration`. The 0.8B has 24 text
layers, 16 key/value linear heads, tied output and a 12-layer vision tower. The 27B has
64 text layers, 16 key / 48 value linear heads, untied output and a 27-layer tower. Both
declare kernel 4, full attention every fourth layer, 256-wide full-attention heads, partial
rotary factor 0.25 and sections `[11,11,10]`. Their processors declare patch 16, temporal size 2,
merge 2, mean/std 0.5, shortest edge 65536 and longest edge 16777216. Actual geometry/pixel
behavior under the selected reference version remains to be probed.

The 27B config declares affine 4-bit/group-64 quantization. Its
[model card](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/blob/10c35caafbb80f7dc6a7a432cdd11af10a6d4818/README.md)
names mlx-vlm 0.6.8 as converter. That recorded version motivates the reference pin; the
newest converter is not automatically the reference for this artifact.

### Version-pinned implementation sources

The following are hashes of raw source files downloaded during this pass. WP0 must also
record installed-package source hashes; a tag inspection does not prove a wheel executes it.

| Upstream source | SHA-256 |
| --- | --- |
| [Transformers 5.22.1 modeling](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | `af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c` |
| [Transformers 5.22.1 configuration](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/qwen3_5/configuration_qwen3_5.py) | `332ea31ed31cc81ddb72da28737e9868a55d295f9f1daa3e0ad0d57d56d4e657` |
| [mlx-vlm 0.6.8 model/sanitize](https://github.com/Blaizzy/mlx-vlm/blob/v0.6.8/mlx_vlm/models/qwen3_5/qwen3_5.py) | `4b9fb8e1950ade4a8375ae797bb836f0a7c321a6b8b459f079ffee7f1d9d58e7` |
| [mlx-vlm 0.6.8 language](https://github.com/Blaizzy/mlx-vlm/blob/v0.6.8/mlx_vlm/models/qwen3_5/language.py) | `f88705ca614896235071378ca1c5ca21f91062b6e13edbfa1d20c3db973ddb89` |
| [mlx-lm 0.32.2 model/sanitize](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/qwen3_5.py) | `851449fd9989d6bf4b3c4fed92ddc3be5f1e781eb2246f89c54d7eff2518d9c6` |

Source observations for WP0 to confirm by execution:

- HF separates full-attention query/gate within each head and uses a sigmoid gate. Partial
  rotary concatenates an unrotated tail. `output_gate_type` is not read in the inspected model
  or config source; absent/`swish` equivalence still requires a config-and-output probe.
- HF uses a 64-token chunk fallback and a token recurrence, normalizes Q/K with epsilon 1e-6,
  and stores its recurrent matrix in key-by-value orientation. Its gated output RMSNorm applies
  SiLU(z) after ordinary learned normalization. Other text norms are zero-centered.
- mlx-vlm 0.6.8 computes its norm-shift predicate **before** dropping MTP keys, including both
  MTP presence and unconverted conv axes in that predicate. It also tests the original prefix
  per norm key. This differs from the parent plan's 2026-10-29 mutable-main observation that
  MTP was dropped first. The distinction must be preserved in converter-produced fixtures.
- mlx-lm 0.32.2 computes its shift predicate from conv axes, drops MTP and adds 1 to selected
  text norms only when the conv predicate is true. The converted prefix by itself is not
  sufficient evidence about a partially sanitized or manually assembled checkpoint.
- The inspected mlx-lm wrapper remaps HF text keys to `language_model.model.*` and drops
  vision keys. Thus mlx-lm and mlx-vlm can share the text namespace; the absence of a complete
  vision inventory determines TEXT-only capability. WP0 must inspect each converter's saved
  result, not assume that mlx-lm exports the unwrapped `model.*` namespace.

### Dependency-resolution check

Both proposed environments resolved with `uv pip compile --python-version 3.12 --generate-hashes`
on the macOS arm64 host, into temporary files. HF direct pins: Torch 2.6.0, torchvision 0.21.0,
Transformers 5.22.1, Pillow 12.3.0, NumPy 2.5.3, safetensors 0.8.0, tokenizers 0.23.2 and
Jinja2 3.1.6. MLX direct pins: mlx-vlm 0.6.8, mlx-lm 0.32.2, mlx 0.32.4, mlx-metal 0.32.4,
Transformers 5.22.1, Pillow 12.3.0, NumPy 2.5.3, safetensors 0.8.0 and tokenizers 0.23.2;
Jinja2 also resolved to 3.1.6. The MLX lock selected mlx-audio 0.6.4 transitively.

This only establishes dependency resolution. No venv was installed and no import or model
execution was tested. WP0 must commit reproducible locks and verify the installed environments;
the existing reference locks remain unchanged.

These inspections support the proposed design; they are not parity, speed or memory results.
The detailed plan owns the choices and the executable stop conditions.

## 2026-11-02 plan-review amendments

- The pinned [mlx-vlm custom delta kernel](https://github.com/Blaizzy/mlx-vlm/blob/v0.6.8/mlx_vlm/models/qwen3_5/gated_delta.py)
  divides the key width by 32 in its per-thread loops. The tiny fixture now proposes key width
  32 / value width 64; width 16 is unsuitable for its default GPU reference path. This is a
  source-derived constraint, not a completed fixture run or a verified constraint on MLX's
  separate built-in `mx.fast.gated_delta_update` / `mlx_fast_gated_delta_update`.
- mlx-vlm's default GPU dispatcher uses its own `gated_delta_kernel`, built with
  `mx.fast.metal_kernel`, rather than the built-in native update. Off GPU or with
  `use_kernel=False`, it selects `gated_delta_chunked` for multi-token scalar gating, otherwise
  `gated_delta_ops`. WP0 must record device, `use_kernel` and the actual path per prefill/decode
  reference run, and separately establish the built-in op's shape limits. Those limits remain
  unverified; the custom kernel's width rule cannot stand in for them.
- `Qwen3VlVisionTower` currently rejects empty DeepStack indices in its constructor. Planned
  reuse therefore requires allowing empty indices in the shared tower while retaining the
  Qwen3-VL descriptor guard and enforcing the same requirement at its model loader. The new
  Qwen3.5 descriptor requires empty indices. The existing empty-array rejection assertion in
  `Qwen3VlVisionTowerValidationTest.constructorRejectsInvalidDeepstackIndexes` moves to
  descriptor/loader tests; the shared-tower case becomes a positive empty-levels test, with its
  other invalid-index cases retained. Both real learned grids are 2304 positions; only the
  synthetic fixture uses 64.
- The inspected config/index JSON does not identify a converter. Shard `__metadata__` remains
  uninspected. Loader classification is specified from the complete prefix/conv-axis/MTP
  inventory; reference-tool converter provenance is not a required runtime input.
- RoPE acceptance is explicit: nested factor required, matching top-level duplicate only,
  `rope_type: default` and `mrope_interleaved: true`. Missing nested factors are rejected rather
  than inferred from the model's 1.0 fallback or the config's 0.25 backward-compatibility default.
  `language_model_only` is accepted only absent/false, with capability determined from weights.
- The tiny vocabulary gets explicit in-range special IDs, config/tokenizer/golden pins, and
  bounds checks. Neither artifact's real vision IDs can be copied into a 128-token fixture.
- The inspected HF normal path uses `solve_triangular`; its export fallback iterates 63 rows
  for chunk size 64, with the chunk dimension batched. A Java ops implementation that does this
  per chunk can incur many dispatches. No cost was measured: WP1 now requires comparison of
  chunked ops, token recurrence, mlx-vlm's custom GPU kernel and MLX's built-in native update
  before selecting a default, with unsupported native shapes explicitly recorded.

### Normalization and per-step path clarification

The inspected HF formula is `x / sqrt(sum(x*x) + 1e-6)` before query scaling. The pinned
[mlx-lm normalization helper](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/gated_delta.py)
passes `eps * inv_scale**2` to RMSNorm, preserving that sum-space epsilon. By contrast, the
inspected mlx-vlm 0.6.8 `language.py` passes an unscaled `1e-6`. Algebraically, its key is
`k / sqrt(sum(k*k) + Dk*1e-6)`; its query has the same denominator plus the usual
`1/sqrt(Dk)` multiplier. Thus its effective epsilon is `32e-6` for the revised tiny fixture
and `128e-6` for both real artifacts. This is a source-derived convention difference, not a
measured error or a claim that BF16 makes it irrelevant.

WP0 must quantify it with paired stock mlx-vlm and a separately labeled normalization-only
diagnostic using `1e-6 / Dk` in RMSNorm. Record identical-weight/history per-step Q/K, state and
logit differences, remaining Java/reference error and greedy margins. Java keeps the HF formula;
stock MLX remains the acceptance reference. Diagnostic source changes and outputs are separate
evidence, never silently substituted goldens or a reason to loosen strict HF bounds.

The inspected mlx-vlm convolution dispatcher uses `_qwen3_5_decode_depthwise_conv` with
float32 accumulation for eligible single-token BF16/FP16-weight decode; other cases, including
prefill, use `nn.Conv1d`. The gated RMSNorm's `_precise_swiglu` computes its SiLU gate and
product in float32 before casting back. WP0 records convolution, gated norm and delta callables
and dtypes for each phase, including when FLOAT32 weights disable the decode helper. Ragged
left-padded attention and speculative target-verify paths are outside the batch-one greedy
reference matrix, not additional required comparison modes.

## 2026-11-03 executable reference and component checkpoint

The isolated environments have now installed and executed. Full file/header inventories,
selected wheel hashes/tags, installed source hashes, saved converter outputs and probe records
are under `tools/qwen35-reference/`. Its provenance guard verifies committed files without
executing Python. This is reference/component evidence; Java full-model acceptance is pending.

- Both real legacy configs parse in Transformers 5.22.1. Tiny CPU FLOAT32 text/image prefill
  and cached decode execute with eager attention and the Torch convolution/delta fallbacks.
  Image forwards require explicit `mm_token_type_ids`. Missing/duplicate rotary-factor and
  gate-metadata behavior is recorded in `evidence/config-probes.json`; Java's planned strict
  acceptance policy remains distinct from HF's permissive/defaulting behavior.
- Both actual converter entry points produced saved float and affine fixtures, then reloaded.
  Saved float tensors match the source bit-for-bit after the specified text norm shifts and
  convolution/vision patch transposes. mlx-lm removes vision and uses the same converted text
  prefix as mlx-vlm. `linear_attn.norm.weight` remains unshifted. Tiny fixtures use an in-range
  plain BPE tokenizer; the initial exploratory WordLevel tokenizer was replaced before this
  checkpoint because jmlx does not support that tokenizer model.
- Complete real inventories contain 488 tensors (0.8B) and 2180 tensors (27B). Each 27B shard's
  metadata is only `{"format":"mlx"}`: it does not supply a converter identity/version.
- Recorded dispatcher probes confirm mlx-vlm's GPU custom kernel and off-GPU/disabled-kernel
  chunked/ops routes. Corrected mlx-lm observations identify its own delta helper/kernel and
  `Qwen3NextRMSNormGated`/`precise_swiglu`, with `nn.Conv1d` in both recorded phases. The archived
  earlier reload probe preserves the exact source hash of the 27B attribution runs.
- The built-in native op accepts the tested key widths 1/16/31/32/33/64/128, value widths
  32/64/128 and grouped heads on CPU/GPU in the recorded boundary matrix. It rejects the tested
  nondivisible head grouping. These are accepted API cases, not a claim that all use an
  accelerated internal kernel. The custom kernel's width constraint does not transfer to it.
  Native and mlx-vlm state orientation is `[B,Hv,Dv,Dk]`, transposed against Java/HF. Masked
  output rows differ, while active rows and final state agree within the probe's recorded
  bound. The separate strict benchmark makes the built-in op ineligible on real-width cases
  where its output/state error exceeds `1e-4`; no bound was relaxed to admit it.
- Paired stock and explicitly labeled normalization-only runs quantify the discrepancy. Tiny
  stock mlx-vlm prefill errors against HF are about 0.01; the diagnostic reduces them below
  `6e-5`. Correct mlx-vlm decode calls the language-model entry point, preserving prefill MRoPE
  delta rather than re-entering the multimodal wrapper. The 27B paired text/image records use
  FLOAT32 floating parameters, identical packed weights/pixels and eight shared teacher-forced
  stock histories. Greedy choices agree, while prefill logit contributions reach about 0.10.
  Diagnostic outputs are attribution only, never acceptance goldens. Java/reference remainder
  remains unmeasured. Token histories deliberately continue past EOS for state comparison;
  this is not an end-to-end public-generation stopping-policy test.
- Real input records pin slow-PIL patches and compare the declared fast backend on the selected
  RGB input, with the explicit 4096-row cap. This particular no-resize sample differs by about
  `5.9e-8` in normalized pixels. It does not establish all resize/geometry equivalences; vision
  integration still owns the differential processor/tower cases.
- The independent 0.8B HF record includes prefill, eight cached steps and fresh full forwards.
  HF's own text cached/fresh discrepancy reaches `3.19e-4`; failed `1e-4` measurements remain
  explicit. Strict tiny bounds are unchanged. Real Java acceptance bounds must be declared
  separately before those runs, including stability/margins and normalization attribution.

Locally, both Java delta-rule prototypes and the composed DeltaNet pass independent strict HF
FLOAT32 goldens at the planned chunk/convolution boundary lengths, with unequal widths, grouped
heads, nonzero state and split prefills. Cache ownership, failure poisoning, repeated-request
cleanup, a 4,128-position memory case and exact parameter-scope handle stability pass. The full
`:jmlx-core:check` and reference provenance guard pass. These tests are added to CI's required
native execution assertion; hosted CI has not run this branch yet.

The four-path reference benchmark and exploratory Java timings show material dispatch overhead:
the Java 27B/1,280-position chunked prototype was about 54 ms per layer versus 413 ms for the
per-token prototype on the recorded exploratory run. This is not the Python reference's roughly
7 ms chunked result. Prepared input files are shared and hashed across languages. Final
committed-source/48-layer Java timings and separate measured native API downcall counts are
still pending. API downcalls are not Metal dispatches. No production prefill default has been
selected, and the WP1 review gate remains open.
