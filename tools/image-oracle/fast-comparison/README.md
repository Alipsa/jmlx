# Qwen2-VL slow-vs-fast image processor comparison (opt-in)

Phase 7.3b WP1 measures the slow-vs-fast preprocessing divergence **once**, on
pinned representative inputs, in this separate comparison venv. The artifact
declares `Qwen2VLImageProcessorFast`; the documented jmlx preprocessing
reference is the slow PIL processor (a 2026-10-08 decision), so this tool
compares three explicit configurations side by side. It produces processed
patches only — it never computes model logits. The model-side comparison
(patches fed through the same HF-reference checkpoint) lives in
`tools/hf-reference/qwen3vl_slowfast_logits.py`.

This environment is **opt-in**: Gradle never installs or imports it, and root
`check` never needs it. The only thing the ordinary build reads from this
directory is `requirements.lock`, through the root
`verifyImageOracleDependencyPins` task, which requires its Pillow/NumPy/
Transformers pins (versions **and** environment markers) to agree with both
`../requirements.lock` and `tools/hf-reference/requirements.lock`.

The lock carries reviewed hash-pinned Torch 2.6.0 (2.6.0+cpu off-Darwin) and
torchvision 0.21.0 in addition to the shared Transformers 4.57.6 / Pillow
12.3.0 / NumPy 2.5.3 pins. torchvision is present so the comparison venv
resolves a real `BaseVideoProcessor` (the slow-reference HF venv deliberately
stays torchvision-free); the runner itself only needs the image processors.

## Install

CPython 3.12 is required:

```sh
./tools/image-oracle/fast-comparison/install.sh
```

The lock was compiled with `--extra-index-url
https://download.pytorch.org/whl/cpu` (the non-Darwin Torch `+cpu` wheel only
exists there); `install.sh` passes the same index.

## Run

```sh
tools/image-oracle/fast-comparison/.venv/bin/python \
    tools/image-oracle/fast-comparison/runner.py --out <output-dir>
```

The runner verifies every `fixtures/` file against `provenance.json` before
use (an unpinned or drifted fixture stops the run), then writes to
`<output-dir>`:

* `manifest.json` — schema
  `jmlx-phase7-3b-wp1-qwen3vl-fastcomparison-v1`: environment versions
  (including the exact Torch version), the lock SHA-256, fixture SHA-256s,
  the three processor classes with their resolved `min_pixels`/`max_pixels`
  and `size`, and per (configuration, image) source size, resized size,
  grid, dtype, shape, the normalized patch-buffer SHA-256, and the
  pre-rescale uint8 patch SHA-256 (splitting resize divergence from
  normalization divergence), plus the slow-vs-fast identity flags.
* `fast__<image>.npy` / `slow_compensated__<image>.npy` — lossless float32
  patch exports for the two model fixtures only. The geometry probe exceeds
  the 16777216-pixel budget (a ~400 MB export) and is fingerprinted by
  grid/shape/SHA-256 only.

Feed the export directory to the model-side comparison:

```sh
tools/hf-reference/.venv/bin/python tools/hf-reference/qwen3vl_slowfast_logits.py \
    --directory .tier-b/qwen3-vl-2b-instruct --patches-dir <output-dir> \
    --out tools/hf-reference/goldens/qwen3-vl-2b-wp1-slow-fast-logits.json
```
