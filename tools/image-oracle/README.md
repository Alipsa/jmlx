# Image oracle

This directory pins the slow PIL image processor (`Idefics3ImageProcessor`,
transformers 4.57.6 over Pillow 12.3.0 and numpy 2.5.3) that is the reference for
`jmlx-vision`'s decode, resize and preprocessing ports. It loads only committed local
image files plus the hash-pinned `preprocessor_config.json`; generation and verification
never download images or models from the Hub.

The three fixture sets, each a `*.input.json` spec with its generated `*.expected.json`:

- `decode.input.json` — per-mode decode conversions (RGB/gray/RGBA/LA/palette/tRNS/ICC/gAMA/
  EXIF-rotated/PNG/JPEG) with the exact reference RGB bytes, plus the explicit rejection
  fixtures (16-bit PNG, CMYK/YCCK JPEG, truncated files, foreign formats).
- `resize.input.json` — single-resize uint8 outputs (identity, up/down/odd samples, edge
  impulse, noise) for all three kernels.
- `chain.input.json` — the full two-stage resize + split + rescale/normalize chain in the
  96/32 test-variant config (the hash-pinned SmolVLM config with only
  `size`/`max_image_size` overridden), stage-by-stage uint8 plus per-tile normalized float32
  and masks.

Install with `./tools/image-oracle/install.sh`. Gradle's `verifyImageOracle` checks the
interpreter, platform, and package pins; `generateImageOracleFixtures` is the only supported
way to rewrite the expected JSON (manual, reviewed — review the diff before committing), and
`verifyImageOracleFixtures` is read-only. The root `check` verifies every committed fixture's
SHA-256 against `provenance.json` in pure Java; the Python-executing tasks are explicit and
run in the pure-Java CI configuration.

## Known reference divergences (documented, deliberate)

- The reference `preprocess` crashes on 2-D (grayscale `L`) and index (`P`) inputs
  (`ValueError: Unsupported number of image dimensions: 2`); chain/resize fixtures therefore
  use opaque RGB only. The Java port always presents 3-channel input and is a deliberate
  superset.
- The reference resizes palette images in **index space** (PIL `P`-mode resize), which
  interpolates palette indices rather than colors; the Java port decodes palette to RGB
  first. Palette images are therefore excluded from the full-chain strict comparison.
- The reference composites alpha **after** resize (its `convert_to_rgb` runs per tile);
  the Java decoder composites over white at decode time. RGBA/LA inputs are excluded from
  the full-chain strict comparison; decode-level fixtures verify the composite math itself.
- The reference accepts `resample` values 4 (BOX) and 5 (HAMMING); the Java port rejects
  every value other than 1/2/3 rather than mapping to a nearby filter.
