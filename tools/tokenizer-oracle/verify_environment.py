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
        raise SystemExit(f"unsupported tokenizer oracle platform: {actual_platform}")
    if list(sys.version_info[:2]) != provenance["python"]:
        raise SystemExit(
            f"tokenizer oracle requires Python {provenance['python']}, "
            f"found {list(sys.version_info[:2])}"
        )
    version = importlib.metadata.version("tokenizers")
    if version != provenance["tokenizers"]:
        raise SystemExit(
            f"tokenizer oracle requires tokenizers {provenance['tokenizers']}, found {version}"
        )
    from tokenizers import Tokenizer  # noqa: F401  -- raises ImportError if unimportable

    print(
        f"Tokenizer oracle verified: Python {sys.version_info.major}."
        f"{sys.version_info.minor}, tokenizers {version}, "
        f"{actual_platform['system']}/{actual_platform['machine']}"
    )


if __name__ == "__main__":
    main()
