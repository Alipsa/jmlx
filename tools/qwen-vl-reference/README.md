# Qwen3-VL MLX-affine reference (opt-in)

Phase 7.3b WP1 item 5 evidence tool: an independent MLX-affine runtime for the
quantized 8B Qwen3-VL candidate. This directory is **opt-in and never an
ordinary-build dependency**: Gradle never installs the venv, never runs the
probe, and no module depends on it.

## What this is

A hash-locked CPython 3.12 environment around `mlx-vlm`, used to run the
`mlx-community/Qwen3-VL-8B-Instruct-4bit` checkpoint (an MLX-affine 4-bit,
group-64 conversion of `Qwen/Qwen3-VL-8B-Instruct`, per its README: converted
with mlx-vlm 0.3.4) on an Apple Silicon GPU. It feeds the model the **same
explicit slow-compensated preprocessing patches** that the pinned HF CPU
float32 reference consumed (the lossless exports of
`tools/image-oracle/fast-comparison/runner.py`, SHA-256 re-verified against
that runner's manifest), records compute/cache dtypes, the
quantized-vs.-float kernel breakdown, prefill/decode logits and greedy ids,
and cross-checks the sequence-expansion facts (prompt length, image span,
grids, MRoPE deltas) against the 2B HF reference probe
(`tools/hf-reference/goldens/qwen3-vl-2b-wp1-probe.json`).

**No parity claim.** This is a different runtime (Metal, 4-bit quantized)
and a different checkpoint (8B) than the pinned HF reference (CPU float32,
2B). The recorded facts are compatibility and provenance evidence only.

## Pin selection

`requirements.lock` pins `mlx-vlm==0.3.9` with `mlx==0.31.2` /
`mlx-metal==0.31.2` (matching the current native and MLX-oracle pins),
`transformers==4.57.6` (matching the HF reference), Pillow 12.3.0 and NumPy
2.5.3 (matching the other image/HF locks; the root
`verifyImageOracleDependencyPins` guard compares these pins across all four
locks: image-oracle, fast-comparison, this lock and hf-reference).
mlx-vlm 0.3.9 is the newest release in the transformers-4.57 /
mlx-0.31.2 generation (0.3.4 added Qwen3-VL; 0.3.6/0.3.8/0.3.9 fixed
attention, position ids, mrope and image-feature/token matching). Releases
0.3.10+ require transformers 5 and a newer MLX and were rejected for this
probe. The lock was resolved host-natively for macOS arm64
(`uv pip compile --generate-hashes --python-version 3.12`); the venv is
Apple-Silicon-only.

## Install

```sh
tools/qwen-vl-reference/install.sh   # needs python3.12; creates tools/qwen-vl-reference/.venv
```

The `.venv/` is git-ignored; it is a derived artifact, not evidence.

## Run the probe

The probe needs the 8B checkpoint on disk (6.8 GB; not a Tier-B artifact and
not downloaded by any build task) and the fast-comparison runner's output
directory:

```sh
# 1. produce (or reuse) the fast-comparison patches, e.g. via
#    ./gradlew runImageOracleFastComparison (output in build/fast-comparison/)
# 2. download the pinned artifact snapshot (revision defcdea7cc7a4b0858fea563cbbce171d328e457)
#    from https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit
# 3. run the probe
tools/qwen-vl-reference/.venv/bin/python tools/qwen-vl-reference/run_qwen3vl_mlx_probe.py \
    --directory <8B-snapshot-dir> \
    --patches-dir <fast-comparison-output-dir> \
    --out tools/qwen-vl-reference/goldens/qwen3-vl-8b-wp1-mlx-probe.json
```

The probe exits non-zero when any pin drifts, when the rendered prompt or
the sequence-expansion facts do not match the 2B HF reference, when the
patch exports no longer match the fast-comparison manifest, or when the hub
LFS digests disagree with the local files.

## Committed evidence

- `requirements.lock` — hash-locked environment (SHA-256 pinned below).
- `run_qwen3vl_mlx_probe.py` — the runner (schema
  `jmlx-phase7-3b-wp1-qwen3vl-mlx8b-v1`).
- `goldens/qwen3-vl-8b-wp1-mlx-probe.json` — the committed probe output
  (artifact inventory, tensor layout, kernel/dtype facts, prefill/decode
  evidence for the two model images).
- `provenance.json` — SHA-256 pins for the golden, the lock, the
  fast-comparison manifest/lock, and every file of the 8B artifact snapshot.

To reverify the committed golden's inputs without the 6.8 GB download, check
`provenance.json`'s digests against a fresh snapshot download.
