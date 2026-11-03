#!/usr/bin/env python3
"""Run each pinned package's real converter entry point; never hand-assemble saved weights."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from mlx_vlm.convert import convert as vlm_convert
from mlx_lm.convert import convert as lm_convert
from inventory import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for family, converter in (("vlm", vlm_convert), ("lm", lm_convert)):
        for affine in (False, True):
            output = args.output_root / (family + ("-affine" if affine else "-float"))
            if output.exists():
                raise ValueError(f"Choose an empty converter output directory: {output}")
            kwargs = dict(hf_path=str(args.source), mlx_path=str(output), dtype="float32",
                          trust_remote_code=False, quantize=affine)
            if affine:
                kwargs.update(q_group_size=64, q_bits=4, q_mode="affine")
            converter(**kwargs)
            records.append(dict(family=family, affine=affine, arguments=kwargs,
                                entry_point=converter.__module__ + "." + converter.__name__,
                                converter_source_sha256=hashlib.sha256(Path(inspect.getfile(converter)).read_bytes()).hexdigest(),
                                source=inventory(args.source), output=inventory(output)))
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(dict(
        kind="actual converter saved-output provenance; reload evidence is separate",
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        conversions=records), indent=2)+"\n")
    print(args.record)


if __name__ == "__main__":
    main()
