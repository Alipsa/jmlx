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
