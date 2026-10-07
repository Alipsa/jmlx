#!/usr/bin/env python3
"""Deterministic generator for the committed image-oracle fixture images.

Run once with the pinned reference environment:

    tools/hf-reference/.venv/bin/python tools/image-oracle/generate_images.py

Every committed fixture byte is SHA-256-pinned in provenance.json; this script is the
record of how the fixtures were constructed (all content is derived from fixed
arithmetic, no unseeded randomness). Re-running it must reproduce identical bytes;
any diff after a Pillow bump is a provenance/golden review, not a silent change.

Layout (all paths relative to tools/image-oracle/fixtures/):
  decode/  accepted-mode decode fixtures (expected RGB bytes recorded by runner.py)
  reject/  explicit rejection fixtures (Java ImageDecoder header pre-scan messages)
  resize/  opaque RGB fixtures for single-resize and full-processor-chain cases
"""
import struct
import zlib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent / "fixtures"
JPEG_QUALITY = 90


def save(im: Image.Image, rel: str, **kwargs) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, **kwargs)
    print(f"wrote {rel} ({path.stat().st_size} bytes)")


def gradient_rgb(w: int, h: int) -> Image.Image:
    """Three independent channel gradients so every channel pattern is distinct."""
    px = bytearray()
    for y in range(h):
        for x in range(w):
            px += bytes(((x * 4 + y * 2) % 256, (x + y * 5) % 256, (x * y) % 256))
    return Image.frombytes("RGB", (w, h), bytes(px))


def lcg_noise(w: int, h: int, seed: int) -> Image.Image:
    """Deterministic LCG noise (fixed seed): worst case for the float-scale bound."""
    state = seed
    px = bytearray()
    for _ in range(w * h):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        px.append(state >> 24)
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        px.append(state >> 24)
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        px.append(state >> 24)
    return Image.frombytes("RGB", (w, h), bytes(px))


def checker(w: int, h: int, cell: int = 2) -> Image.Image:
    """High-frequency checkerboard: stresses kernel averaging and LANCZOS ringing."""
    px = bytearray()
    for y in range(h):
        for x in range(w):
            v = 255 if ((x // cell + y // cell) % 2) == 0 else 0
            px += bytes(((v + x) % 256, (v + y) % 256, v))
    return Image.frombytes("RGB", (w, h), bytes(px))


def impulse(w: int, h: int) -> Image.Image:
    """Black field with white impulse pixels at corners, edge midpoints and inside."""
    im = Image.new("RGB", (w, h), (0, 0, 0))
    points = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (0, h // 2), (w - 1, h // 2), (w // 2, h - 1),
        (w // 2, h // 2), (w // 2 + 1, h // 2 - 1), (w // 3, h // 3),
    ]
    for x, y in points:
        im.putpixel((x, y), (255, 255, 255))
    return im


# ------------------------------------------------------------------ hand-made PNGs

def png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_png(rel: str, width: int, height: int, bit_depth: int, color_type: int,
              samples_per_pixel: int, row_bytes: list[bytes]) -> None:
    """Minimal valid PNG writer for the bit depths PIL cannot emit (4-bit gray,
    16-bit truecolor). Filter type 0 on every row."""
    raw = b"".join(b"\x00" + row for row in row_bytes)
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    )
    data += png_chunk(b"IDAT", zlib.compress(raw, 9))
    data += png_chunk(b"IEND", b"")
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"wrote {rel} ({path.stat().st_size} bytes)")


# ----------------------------------------------------------------------- main

def main() -> None:
    # decode: accepted modes
    save(gradient_rgb(64, 48), "decode/rgb-64x48.png")

    gray = Image.new("L", (60, 40))
    gray.putdata([(x * 4 + y * 5) % 256 for y in range(40) for x in range(60)])
    save(gray, "decode/gray-60x40.png")

    rgba = Image.new("RGBA", (32, 32))
    rgba.putdata([
        ((x * 3 + y) % 256, (x + y * 7) % 256, (x * y) % 256, (x * 8) % 256)
        for y in range(32) for x in range(32)
    ])
    save(rgba, "decode/rgba-32x32.png")

    la = Image.new("LA", (32, 24))
    la.putdata([((x * 8) % 256, (y * 16) % 256) for y in range(24) for x in range(32)])
    save(la, "decode/la-32x24.png")

    palette16 = [(i * 16, 255 - i * 16, (i * 40) % 256) for i in range(16)]
    for rel, trns in (
        ("decode/palette-32x32.png", None),
        ("decode/palette-trns-single-32x32.png",
         bytes([255] * 5 + [0] + [255] * 10)),  # single transparent index 5
        ("decode/palette-trns-perentry-32x32.png",
         bytes([255, 0, 128, 255, 64, 255, 200, 0, 128, 255, 32, 255, 255, 64, 255, 255])),
        ("decode/palette-trns-short-32x32.png",
         bytes([255, 0, 128, 255, 64, 255, 200, 0])),  # 8 entries; indices 8..15 stay 255
    ):
        p = Image.new("P", (32, 32))
        p.putpalette([c for rgb in palette16 for c in rgb])
        p.putdata([(x + y * 2) % 16 for y in range(32) for x in range(32)])
        if trns is not None:
            p.info["transparency"] = trns
        save(p, rel)

    save(gradient_rgb(96, 64), "decode/jpeg-color-96x64.jpg", quality=JPEG_QUALITY)

    gray_j = Image.new("L", (64, 48))
    gray_j.putdata([(x * 4 + y * 5) % 256 for y in range(48) for x in range(64)])
    save(gray_j, "decode/jpeg-gray-64x48.jpg", quality=JPEG_QUALITY)

    # EXIF orientation 6 (stored 100x60 landscape; viewers would rotate to 60x100).
    # The Java decoder must return the stored, unrotated 100x60 pixels.
    exif_img = gradient_rgb(100, 60)
    exif = exif_img.getexif()
    exif[0x0112] = 6  # EXIF Orientation tag
    save(exif_img, "decode/exif-orient6-100x60.jpg", quality=JPEG_QUALITY,
         exif=exif.tobytes())

    # ICC/gAMA color-management fixtures: the decoder must ignore both. The P3 profile
    # comes from the host's system color sync profiles (PIL cannot synthesize one); the
    # generated PNG is hash-pinned in provenance.json, so this one-time host input does not
    # affect fixture reproducibility. Override with ICC_PROFILE_PATH if needed.
    import os

    profile_path = os.environ.get(
        "ICC_PROFILE_PATH", "/System/Library/ColorSync/Profiles/Display P3.icc"
    )
    with open(profile_path, "rb") as f:
        p3_bytes = f.read()
    if len(p3_bytes) < 128 or p3_bytes[36:40] != b"acsp":
        raise SystemExit(f"not an ICC profile: {profile_path}")
    save(gradient_rgb(32, 32), "decode/icc-p3-32x32.png", icc_profile=p3_bytes)
    save(gradient_rgb(32, 32), "decode/gama-32x32.png", gamma=1.0 / 4.5)
    # The unprofiled twin of the ICC fixture: decode of both must be byte-identical.
    save(gradient_rgb(32, 32), "decode/icc-p3-twin-32x32.png")

    # Limits fixtures: solid colors keep the PNG files tiny.
    save(Image.new("RGB", (5760, 4200), (200, 100, 50)), "decode/limits-5760x4200.png")
    save(Image.new("RGB", (8192, 5760), (90, 210, 170)), "decode/limits-8192x5760.png")
    save(Image.new("RGB", (20000, 300), (10, 20, 30)), "decode/limits-wide-20000x300.png")

    # reject: 16-bit PNG (gray via PIL "I;16", truecolor hand-made)
    im16 = Image.new("I;16", (8, 8))
    im16.putdata([i * 6553 for i in range(64)])
    save(im16, "reject/png-16bit-gray.png")

    rows16 = []
    for y in range(8):
        row = bytearray()
        for x in range(8):
            row += struct.pack(">HHH", (x * 32510) % 65536, (y * 32510) % 65536,
                               ((x + y) * 16383) % 65536)
        rows16.append(bytes(row))
    write_png("reject/png-16bit-rgb.png", 8, 8, 16, 2, 3, rows16)

    # reject: sub-8-bit PNGs (hand-made)
    rows4 = []
    for y in range(8):
        row = bytearray()
        for x in range(8):
            row.append(((x * 7 + y * 3) % 16) << 4 | ((x + y) % 16))
        rows4.append(bytes(row))
    write_png("reject/png-4bit-gray.png", 8, 8, 4, 0, 1, rows4)

    onebit = Image.new("1", (16, 16))
    onebit.putdata([((x + y * 3) % 4 == 0) * 255 for y in range(16) for x in range(16)])
    save(onebit, "reject/png-1bit-palette.png")

    # reject: CMYK/YCCK JPEGs (byte surgery on a valid RGB JPEG; the Java decoder
    # rejects from the header pre-scan before any pixel decode).
    base = (ROOT / "decode/jpeg-color-96x64.jpg").read_bytes()

    # (a) four-component SOF: patch the component count byte of the first SOFn.
    cmyk = bytearray(base)
    pos = 2
    while cmyk[pos] != 0xFF or cmyk[pos + 1] < 0xC0 or cmyk[pos + 1] > 0xCF:
        pos += 2 + int.from_bytes(cmyk[pos + 2:pos + 4], "big")
    cmyk[pos + 9] = 4
    path = ROOT / "reject/jpeg-cmyk-4comp.jpg"
    path.write_bytes(bytes(cmyk))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    # (b) Adobe APP14 transform code 2 (the CMYK marker used by Adobe encoders).
    app14 = (
        b"\xff\xee"
        + struct.pack(">H", 2 + 5 + 2 + 2 + 1 + 4)
        + b"Adobe"
        + bytes([1, 0, 0, 0, 2])
        + b"\x00\x00\x00\x00"
    )
    adobe = base[:2] + app14 + base[2:]
    path = ROOT / "reject/jpeg-adobe-transform2.jpg"
    path.write_bytes(adobe)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    # reject: truncated / non-image inputs
    png = (ROOT / "decode/rgb-64x48.png").read_bytes()
    idat = png.index(b"IDAT")
    path = ROOT / "reject/png-truncated.png"
    path.write_bytes(png[: idat + 8 + 10])
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    jpg = base
    path = ROOT / "reject/jpeg-truncated.jpg"
    path.write_bytes(jpg[: len(jpg) * 3 // 5])
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")

    bmp = Image.new("RGB", (32, 32), (0, 0, 0))
    bmp.putpixel((5, 6), (255, 0, 0))
    save(bmp, "reject/bmp-32x32.bmp")

    # resize: opaque RGB fixtures (shared by single-resize and full-chain cases)
    save(gradient_rgb(100, 80), "resize/gradient-100x80.png")
    save(gradient_rgb(80, 100), "resize/gradient-80x100.png")
    save(gradient_rgb(200, 40), "resize/wide-200x40.png")
    save(gradient_rgb(96, 96), "resize/square-96x96.png")
    save(checker(120, 80), "resize/checker-120x80.png")
    save(impulse(64, 64), "resize/impulse-64x64.png")
    save(gradient_rgb(321, 199), "resize/odd-321x199.png")
    save(gradient_rgb(96, 64), "resize/small-96x64.png")
    save(checker(48, 48, cell=1), "resize/identity-48x48.png")
    save(Image.new("RGB", (64, 64), (137, 200, 64)), "resize/solid-64x64.png")
    save(lcg_noise(96, 96, seed=12345), "resize/noisy-96x96.png")


if __name__ == "__main__":
    main()
