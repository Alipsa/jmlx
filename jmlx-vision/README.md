# jmlx-vision

Pure-Java image preprocessing for `jmlx-models`: PNG/JPEG decoding, RGB conversion,
Pillow-exact resampling, rescale/normalize, the SmolVLM/Idefics3 image processor (tile
splitting, padding masks, grid/order metadata), the Qwen3-VL image preprocessing port
(smart_resize, BICUBIC resize, rescale/normalize, merge-block patch flatten) and the
Phi-3.5-Vision (LLaVA-style) image preprocessing port (HD_transform, white padding,
torch-exact bicubic global image, sub-crop slots). No native
dependency; Java 21+.

## Public surface

| Type | Role |
| --- | --- |
| `RgbImage` | Immutable interleaved unsigned-RGB, row-major, `w*h*3` bytes. Content equality/hashCode; defensive copies. |
| `ImageDecoder` | Decodes a `Path` or `InputStream` to an `RgbImage`. The no-argument overloads apply `ImageDecodeLimits(16_777_216, 16_384)`. The policy overloads take an `AlphaPolicy` and/or an `ExifOrientationPolicy`. |
| `ImageDecodeLimits` | Positive `maxPixels` / `maxDimension` bound, checked from decoded metadata *before* any pixel allocation. |
| `ImageTensor` | Immutable `float[]` with shape/layout (`HWC` or `NHWC`), defensive copies, bounded `toString`. |
| `ImageTransforms` | `resize(RgbImage, w, h, Resampling)` and `normalize(RgbImage, scale, mean, std)`. |
| `Resampling` | `BILINEAR`, `BICUBIC`, `LANCZOS` — ports of the pinned Pillow kernels. |
| `AlphaPolicy` | `COMPOSITE_OVER_WHITE` (every overload's default) or `DROP_ALPHA` (keeps the source color channels, discards the alpha band — the Qwen3-VL reference's `convert("RGB")` semantics). `tRNS` is stripped before decode under both. |
| `ExifOrientationPolicy` | `IGNORE` (default: stored pixels) or `APPLY` (Pillow's `exif_transpose` before the RGB conversion). `APPLY` is the Qwen3-VL file path's policy (Phase 7.3b WP9); the SmolVLM path keeps `IGNORE`. |
| `SmolVlmProcessorConfig` | Parsed, validated `preprocessor_config.json`. |
| `SmolVlmImageProcessor` | Full processor: two-stage resize chain, tile splitting, global thumbnail, masks, per-tile normalized tensors. |
| `SmolVlmGeometryPlan` | Pure per-image geometry (stage-1/stage-2 sizes, split grid, tile count) computed from dimensions alone, with no pixel work. |
| `Qwen3VlProcessorConfig` | Parsed, validated `preprocessor_config.json` for the slow-processor Qwen3-VL path; the declared max-pixel bound is capped at the raw-row budget's derived hard cap (1,048,576 at the default 4,096-row budget, decision (b)), and the 4,096 raw-patch-row budget (decision (c)) is a builder option only. |
| `Qwen3VlGeometryPlan` | Pure per-image geometry (resized size, `grid_thw`, raw/merged row counts) from dimensions alone, with the aspect and raw-row-budget checks, shared with the zero-token path. |
| `Qwen3VlImageProcessor` | `process(RgbImage)` runs the ported chain; `processFile(Path, ImageDecodeLimits)` reads a PNG/JPEG from disk and `processEncoded(byte[], ImageDecodeLimits)` decodes in-memory bytes — both are the file-to-processor entry points applying the Qwen file policy (decode `DROP_ALPHA` + `APPLY`, then process). |
| `Qwen3VlImageProcessorResult` | The flattened merge-block patch tensor (`grid_h*grid_w` rows of 1,536 floats at the pinned geometry — not an image layout) with grid/order metadata; defensive buffer accessors. |
| `Phi3VProcessorConfig` | Parsed, validated bundled-processor config of the pinned `microsoft/Phi-3.5-vision-instruct` artifact (`image_mean`/`image_std`/`num_crops`); `do_convert_rgb: false` is rejected because the ported decode path always delivers RGB. |
| `Phi3VGeometryPlan` | Pure per-image geometry (HD resize and 336-multiple padding targets, crop grid, slot count, per-image image-token count) computed from dimensions alone, with no pixel work. |
| `Phi3VImageProcessor` | The LLaVA-style preprocessing chain (`process(RgbImage)`, `processFile(Path, ImageDecodeLimits)`, `processEncoded(byte[], ImageDecodeLimits)`), including the torch-exact global-image bicubic resize. |
| `Phi3VImageProcessorResult` | The reference's one `pixel_values` batch row: `slotCount()` slots of `3*336*336` floats, slot-major then CHW, with the geometry plan; defensive buffer accessors. |

`SmolVlmImageProcessor.geometry(height, width)` returns the `SmolVlmGeometryPlan` for an image's
dimensions; the processor's pixel chain uses that same plan for its resize targets and split
grid, so prompt-side tile layout computed from dimensions cannot drift from the tiles actually
produced.

## Decode policy

- **Supported:** 8-bit RGB, grayscale, RGBA and grayscale-with-alpha PNG, 1-, 2-, 4- and
  8-bit palette PNG (Pillow's own default for a small palette is 4-bit) with optional `tRNS`,
  and 8-bit RGB and grayscale JPEG, including scans with restart markers (`RST0`–`RST7`) and an
  optional DRI segment. A `tRNS` chunk is ignored for every color type and stripped from the bytes
  before decode: the pinned processor never applies it (a palette image is rebuilt from
  `getpalette()`, RGB only, and the RGB/L conversion drops the chunk), so tRNS files decode
  identically to their opaque twins. A `PLTE` chunk on a truecolor or RGBA image is only a
   *suggested* palette (RFC 2083 §11.2) and is ignored — as it is by the built-in reader and
   by the reference; only palette images decode through the palette. Decoding uses the JDK's
   built-in
  `javax.imageio` PNG/JPEG readers; decoded bytes are addressed through the raster's
  `SampleModel` (the readers store `B,G,R`/`A,B,G,R`, not channel order) and the band-to-channel
  mapping is probed once through the decoded `ColorModel`, so the band order is never assumed.
- **Rejected with format-specific `IOException`s:** 16-bit PNG, CMYK/YCCK JPEG, and any input
  above the configured limits.
- **EXIF orientation:** ignored by default — every overload that does not take an
  `ExifOrientationPolicy` decodes an EXIF-rotated JPEG to its stored (unrotated) pixels. That
  matches the pinned SmolVLM/Idefics3 image-oracle reference path (`PIL.Image.open` → `load()`
  with no `exif_transpose`), and the committed decode fixtures stay pinned under it. The
  overloads that take an `ExifOrientationPolicy` add an opt-in `APPLY` that applies the stored
  orientation before the RGB conversion exactly as Pillow's `ImageOps.exif_transpose` does — a
  pure pixel permutation, no resampling; both JPEG `APP1` `Exif` segments and PNG `eXIf` chunks
  are honored. `APPLY` is the decode policy of the Qwen3-VL file path (the Phase 7.3b WP9
  decision: the SmolVLM path keeps the default). Under either policy a malformed or truncated
  EXIF block is treated as absent — it never breaks a decode.
- **ICC and gAMA color management are ignored.** Embedded color profiles and gamma chunks are
  not applied to samples, matching the reference path, which performs no `ImageCms` conversion.
  Do not expect display-referred colors; callers doing color management must do it themselves.
- Default limits are conservative: the no-argument overloads reject typical 24 MP and 48 MP
  phone photos. Use the explicit `ImageDecodeLimits` overloads to accept such inputs when your
  memory budget allows.

## Precision

`ImageTransforms.resize` targets **bit-exact uint8 output** against the pinned Pillow 12.3.0
resampler (`src/libImaging/Resample.c`, ported coefficients, fixed-point intermediate
rounding and clip8 behavior) for all three kernels, up and down, at arbitrary sizes. The
resampling port and its license are recorded in `NOTICE` and in the source-file header.

The committed oracle fixtures are compared **byte-exactly**: every accepted decode (including
the color and grayscale JPEG fixtures) matches the pinned Pillow output with maximum channel
error `0`, and the full two-stage resize/split/normalize chain matches byte-for-byte on uint8
pixels (normalized float tiles within the documented `1e-6` float32 tolerance). The Qwen3-VL
`qwen3vl` family (21 cases, including the shared processed-patch bins under
`qwen3vl/patches/`) is byte-verified the same way by `ImageOracleQwen3VlTest` (see
"Qwen3-VL preprocessing" below). See `req/plans/phase7-3a-reference-findings.md` §8 for the
decode-path findings behind the byte-exactness (raster band order, packed sub-8-bit palettes,
JPEG truncation detection).

## Qwen3-VL preprocessing

Phase 7.3b WP2 port of the pinned transformers 4.57.6 slow `Qwen2VLImageProcessor` image path:
`smart_resize` (Python's ties-to-even rounding, binary64 operation order preserved),
Pillow-exact BICUBIC resize, rescale/normalize in float32, temporal duplication and the
merge-block patch permutation/flatten. The config is parsed from the pinned
`preprocessor_config.json` (committed as the `qwen3vl` fixture, byte-identical to the 2B
artifact's file): the declared `max_pixels` is capped at the raw-row budget's derived hard cap
`rawRowBudget * patchSize * patchSize` — 1,048,576 at the default 4,096-row budget (the
planned-effective bound, decision (b); the pre-cap value stays available as
`declaredMaxPixels()` for provenance) — and the 4,096 raw-patch-row per-image budget
(decision (c)) is a builder option only, so an untrusted config cannot lift it.

`Qwen3VlImageProcessor.processFile(Path, ImageDecodeLimits)` reads the image bytes from disk and
`processEncoded(byte[], ImageDecodeLimits)` decodes them from memory; both are the
file-to-processor entry points and apply the Qwen file policy: `ImageDecoder` with
`AlphaPolicy.DROP_ALPHA` + `ExifOrientationPolicy.APPLY` — the
reference's `exif_transpose` + `convert("RGB")`, which drops alpha, never composites over white
and never applies a palette `tRNS` — then `process`. This is a deliberate contrast with the
module-default decode (`COMPOSITE_OVER_WHITE` + `IGNORE`) used by the SmolVLM path, which is
unchanged.

Byte-exactness evidence: the `qwen3vl` family of the `tools/image-oracle` fixtures (21 cases,
including the rotated eXIf fixtures with their stored-decode controls) plus the shared
processed-patch bins under `tools/image-oracle/fixtures/qwen3vl/patches/`, byte-verified by
`ImageOracleQwen3VlTest`; the reference findings behind them are §4 of
`req/plans/phase7-3b-reference-findings.md` (§4.8 for the part-2 fixture set). End-to-end
model-side acceptance of this preprocessing — the composed Qwen3-VL prefill and cached decode
against a fresh, full, uncached CPU-HF float32 reference — is the Phase 7.3b WP8 golden suites
(same findings, §11.8).

## Phi-3.5-Vision preprocessing

Phase 7.3c WP2 port of the bundled trust-remote-code `Phi3VImageProcessor.preprocess` of the
pinned `microsoft/Phi-3.5-vision-instruct` artifact (findings §6 of
`req/plans/phase7-3c-reference-findings.md`): `HD_transform` (portrait transposition, the
double-precision scale-selection loop, BILINEAR resize, 336-multiple white padding),
`ToTensor()` (`float32(pixel / 255)`) and `Normalize(mean, std)` in float32, the global-image
step, the h-major sub-crop split (pure copies, no resampling) and the slot assembly
(`[global] + crops`, zero-padded to `num_crops + 1` slots). The output of
`Phi3VImageProcessor.process` is one reference `pixel_values` batch row of
`slotCount() * 3 * 336 * 336` float32 values, slot-major then CHW.

The file/bytes entry points decode with the Phi-3.5-Vision policy: `ImageDecoder` with
`AlphaPolicy.DROP_ALPHA` + `ExifOrientationPolicy.IGNORE` — the reference's `convert("RGB")`
on its string load path, no `exif_transpose` — then `process`.

**The global-image resize is torch-exact, not Pillow-exact.** The reference runs
`F.interpolate(x, size=(336, 336), mode='bicubic')` on the *normalized* float32 tensor — a
PyTorch CPU path, not a Pillow kernel. `TorchBicubic` (package-private) is a port of the
pinned PyTorch **v2.6.0** CPU upsampling source (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`
and `aten/src/ATen/native/UpSample.h` — URLs, SHA-256s and the BSD 3-Clause license text are
recorded in `NOTICE` and in the file header). The pinned wheels are compiled with
floating-point contraction enabled, so every C subexpression `a * b + c` in the kernel (the
real-source-index computation, both cubic convolution polynomials, and the separable
height-outer/width-inner accumulations, where the first term is a bare product) executes as a
single-rounded hardware fused-multiply-add. The port reproduces that compiled arithmetic
exactly with `fma32`, a bit-exact IEEE 754 binary32 fused multiply-add implemented in pure
Java (Java's `Math.fma` is double-only, and a plain double-emulation would double-round):
a plain `*`/`+` transcription of the staged source deviates from the pinned wheels by up to
~881 float ulps, while the fused form is bit-identical on the fixtures. This is a
*different algorithm* from the module's `PillowResample` BICUBIC kernel (Pillow's 8-bit
fixed-point `a = -0.5` pipeline, ported for the SmolVLM and Qwen3-VL uint8 paths) and is not
used by them.

Byte-exactness evidence: the `phi3v` family of the `tools/image-oracle` fixtures (5 cases
plus the shared processed-slot bins under `phi3v/patches/`), byte-verified by
`ImageOraclePhi3VTest`; the `fma32` core was stress-verified against libm's `fmaf`
(~650,000 cases, zero mismatches) before the family was generated.

## Divergences from the reference's operation order


The reference pipeline (the pinned `transformers` 4.57.6 `Idefics3ImageProcessor.preprocess`)
resizes **and splits before** `convert_to_rgb`; this port decodes straight to RGB (compositing
alpha over white, expanding palettes) and resizes in RGB space. Decoding to RGB first is a
*deliberate extension* of the reference, and the two orderings agree only for fully opaque
**RGB** input:

- The reference's default `preprocess` (channels-first output) accepts only opaque RGB:
  grayscale (`L`) and palette (`P`) inputs raise
  `ValueError: Unsupported number of image dimensions: 2`, and `LA` input raises
  `ValueError: Unable to infer channel dimension format` (verified on the committed decode
  fixtures against the pinned venv). The port accepts all of them.
- With `input_data_format="channels_last"`, the reference accepts `L` and `P`, but it resizes
  the single channel as a PIL image in mode `P`, and the pinned Pillow 12.3.0 `Image.resize`
  forces `Resampling.NEAREST` for modes `1`/`P` regardless of the configured kernel — so even
  there the reference does not run the configured kernel on the single channel, while the port
  resizes the expanded/composited RGB with it. Measured max absolute difference on the
  normalized float32 `pixel_values` (pinned SmolVLM-256M config): `1.6313726` on
  `gray-60x40.png` and `1.5843138` on `palette8-32x32.png`, exactly `0.0` on the opaque RGB
  fixture, where the two orderings agree.

The byte-exact full-chain fixtures therefore use opaque RGB only (the oracle runner enforces
`image.mode == "RGB"`); decode itself is verified byte-exactly for every accepted family
(decode fixtures). The reference pipeline also keeps EXIF orientation unapplied and ignores
ICC/gAMA, which the port matches (see "Decode policy" above).

## License

MIT (root `LICENSE`). The resampling kernels port Pillow source under its MIT-CMU license;
the torch-exact bicubic resize (`TorchBicubic`) ports PyTorch source under its BSD 3-Clause
license; see `NOTICE` for the full texts and the pinned source hashes.
