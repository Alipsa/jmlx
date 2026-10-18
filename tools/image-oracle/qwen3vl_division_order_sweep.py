#!/usr/bin/env python3
"""Phase 7.3b WP2 part 2 division-order sweep for the Qwen3-VL smart_resize port.

Reproducible evidence for the division-order waiver recorded in
req/plans/phase7-3b-plan.md section 4 (the 2026-10-18 review waiver note)
and req/plans/phase7-3b-reference-findings.md section 4.5 (the "Division-order
waiver record"). It is never invoked by Gradle; run it with the image-oracle
venv:

    ./tools/image-oracle/install.sh     # one-time, hash-locked CPython 3.12
    tools/image-oracle/.venv/bin/python tools/image-oracle/qwen3vl_division_order_sweep.py
    tools/image-oracle/.venv/bin/python tools/image-oracle/qwen3vl_division_order_sweep.py --control

It sweeps the full jmlx-vision ImageDecodeLimits.DEFAULTS range (sides <=
16,384, area <= 16,777,216) at the pinned Qwen3-VL processor numbers (factor
32, max_pixels 1,048,576), counting the cells whose rounded h_bar * w_bar
exceeds max_pixels -- the cells reaching smart_resize's max-pixels branch,
with the <= 200 aspect guard -- and asking whether any of them gets a
different final geometry under a different division order:

  default:   floor(h / beta / factor)   vs   floor(h / (beta * factor))
             the port's two sequential divisions (the reference's order) vs
             the folded form
  --control: floor(h / beta / factor)   vs   floor(h * (1 / beta) / factor)
             the port's order vs the reciprocal-multiply form; the control
             must flag cells, or the sweep proves nothing

Only numpy is required. Expected output as of 2026-10-18 (the script exits
non-zero if either count drifts from these):

  default:   max-branch cells 56380384 division-order-sensitive 0
  --control: max-branch cells 56380384 division-order-sensitive 1334

The trailing examples list the first sensitive (h, w) cells, at most three
per height; it is empty for the default comparison. The venv-verified
geometry cases in jmlx-vision's Qwen3VlGeometryPlanTest (including
1920x1080 -> 1344x768 and the 160x6560 max-branch case) re-run under the
ordinary :jmlx-vision:test task.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

MAXP = 1048576.0
F = 32.0
MAXSIDE = 16384
MAXAREA = 16777216

EXPECTED = {
    "default": (56_380_384, 0),
    "control": (56_380_384, 1334),
}


def sweep(control: bool) -> tuple[int, int, list[tuple[int, int]]]:
    hits: list[tuple[int, int]] = []
    cells = 0
    sensitive = 0
    for h in range(1, MAXSIDE + 1):
        w = np.arange(1, min(MAXSIDE, MAXAREA // h) + 1, dtype=np.float64)
        hf = float(h)
        ok = np.maximum(hf, w) / np.minimum(hf, w) <= 200
        hb = np.round(hf / F) * F
        wb = np.round(w / F) * F
        m = ok & (hb * wb > MAXP)
        if not m.any():
            continue
        w = w[m]
        cells += w.size
        beta = np.sqrt(hf * w / MAXP)
        if control:
            dh = np.floor(hf / beta / F) != np.floor(hf * (1.0 / beta) / F)
            dw = np.floor(w / beta / F) != np.floor(w * (1.0 / beta) / F)
        else:
            dh = np.floor(hf / beta / F) != np.floor(hf / (beta * F))
            dw = np.floor(w / beta / F) != np.floor(w / (beta * F))
        d = dh | dw
        if d.any():
            sensitive += int(d.sum())
            for ww in w[d][:3]:
                hits.append((h, int(ww)))
    return cells, sensitive, hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep the DEFAULTS range for smart_resize division-order sensitivity."
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="compare against floor(x * (1 / beta) / factor) instead of the folded form",
    )
    args = parser.parse_args()
    name = "control" if args.control else "default"
    cells, sensitive, hits = sweep(args.control)
    print(f"max-branch cells {cells} division-order-sensitive {sensitive} examples {hits[:10]}")
    expected_cells, expected_sensitive = EXPECTED[name]
    if (cells, sensitive) != (expected_cells, expected_sensitive):
        print(
            "DRIFT: expected max-branch cells "
            f"{expected_cells} division-order-sensitive {expected_sensitive} "
            "(as of 2026-10-18); investigate before citing this sweep",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
