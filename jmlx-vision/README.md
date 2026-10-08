# jmlx-vision

Pure-Java image preprocessing for `jmlx-models`: PNG/JPEG decoding, RGB conversion,
Pillow-exact resampling, rescale/normalize, and the SmolVLM/Idefics3 image processor (tile
splitting, padding masks, grid/order metadata). No native dependency; Java 21+.

## Public surface

| Type | Role |
| --- | --- |
| `RgbImage` | Immutable interleaved unsigned-RGB, row-major, `w*h*3` bytes. Content equality/hashCode; defensive copies. |
| `ImageDecoder` | Decodes a `Path` or `InputStream` to an `RgbImage`. The no-argument overloads apply `ImageDecodeLimits(16_777_216, 16_384)`. |
| `ImageDecodeLimits` | Positive `maxPixels` / `maxDimension` bound, checked from decoded metadata *before* any pixel allocation. |
| `ImageTensor` | Immutable `float[]` with shape/layout (`HWC` or `NHWC`), defensive copies, bounded `toString`. |
| `ImageTransforms` | `resize(RgbImage, w, h, Resampling)` and `normalize(RgbImage, scale, mean, std)`. |
| `Resampling` | `BILINEAR`, `BICUBIC`, `LANCZOS` — ports of the pinned Pillow kernels. |
| `SmolVlmProcessorConfig` | Parsed, validated `preprocessor_config.json`. |
| `SmolVlmImageProcessor` | Full processor: two-stage resize chain, tile splitting, global thumbnail, masks, per-tile normalized tensors. |

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
- **EXIF orientation is ignored.** An EXIF-rotated JPEG decodes to its stored (unrotated)
  pixels, matching the pinned reference path (`PIL.Image.open` → `load()` with no
  `exif_transpose`). When you need the rotated view — in particular for phone photos — rotate
  the image yourself before decoding. No orientation accessor is exposed in this milestone.
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
pixels (normalized float tiles within the documented `1e-6` float32 tolerance). See
`req/plans/phase7-3a-reference-findings.md` §8 for the decode-path findings behind the
byte-exactness (raster band order, packed sub-8-bit palettes, JPEG truncation detection).

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

MIT (root `LICENSE`). The resampling kernels port Pillow source under its MIT-CMU license; see
`NOTICE` for the full text and the pinned source hashes.
