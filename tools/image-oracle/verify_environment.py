#!/usr/bin/env python3
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main() -> None:
    provenance = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    actual_platform = {"system": platform.system(), "machine": platform.machine()}
    if actual_platform not in provenance["platforms"]:
        raise SystemExit(f"unsupported image oracle platform: {actual_platform}")
    if list(sys.version_info[:2]) != provenance["python"]:
        raise SystemExit(
            f"image oracle requires Python {provenance['python']}, "
            f"found {list(sys.version_info[:2])}"
        )
    versions = {
        "pillow": importlib.metadata.version("Pillow"),
        "numpy": importlib.metadata.version("numpy"),
        "transformers": importlib.metadata.version("transformers"),
    }
    for name, expected in (
        ("pillow", provenance["pillow"]),
        ("numpy", provenance["numpy"]),
        ("transformers", provenance["transformers"]),
    ):
        actual = versions[name]
        if actual != expected:
            raise SystemExit(f"image oracle requires {name} {expected}, found {actual}")

    from PIL import Image  # noqa: F401  -- raises ImportError if unimportable
    import numpy  # noqa: F401  -- raises ImportError if unimportable
    from transformers.models.idefics3.image_processing_idefics3 import (  # noqa: F401
        Idefics3ImageProcessor,
    )

    print(
        f"Image oracle verified: Python {sys.version_info.major}."
        f"{sys.version_info.minor}, Pillow {versions['pillow']}, "
        f"numpy {versions['numpy']}, transformers {versions['transformers']}, "
        f"{actual_platform['system']}/{actual_platform['machine']}"
    )


if __name__ == "__main__":
    main()
