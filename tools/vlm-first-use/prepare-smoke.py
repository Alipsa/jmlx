#!/usr/bin/env python3
"""Prepare one-pair WP4 smoke inputs without creating the session output directory."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from model_cache import clean_model_dir


WORKLOADS = {
    "qwen3vl": {
        "name": "qwen3vl-2b",
        "manifest": "tools/tier-b/qwen3vl-2b-instruct.json",
        "stop_token_ids": [151643, 151645],
    },
    "phi3v": {
        "name": "phi3v-4b",
        "manifest": "tools/tier-b/phi3v-4b-instruct.json",
        "stop_token_ids": [32000],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", choices=["both", *WORKLOADS], required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--old-tree", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    old_tree = args.old_tree.resolve()
    run_root = args.run_root.resolve()
    model_root = args.model_root.resolve()
    if run_root.is_relative_to(repo) or run_root.is_relative_to(old_tree):
        parser.error("run root must be outside both trees")
    if run_root.exists():
        parser.error(f"run root already exists: {run_root}")

    selected = list(WORKLOADS) if args.workloads == "both" else [args.workloads]
    workloads = []
    for key in selected:
        details = WORKLOADS[key]
        manifest = repo / details["manifest"]
        model_dir = model_root / key
        clean_model_dir(model_dir, manifest)
        workloads.append(
            {
                "name": details["name"],
                "manifest": details["manifest"],
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "model_dir": str(model_dir),
                "stop_token_ids": details["stop_token_ids"],
            }
        )

    def tree(path: Path) -> dict:
        sha = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        return {
            "path": str(path),
            "expected_tree_sha": sha,
            "class_dir": str(path / "jmlx-benchmarks/build/install/jmlx-benchmarks/lib"),
            "native_dir": str(path / "native/install/lib"),
        }

    spec = {
        "schema": "jmlx-vlm-first-use-spec-1",
        "session": f"wp4-native-smoke-{run_root.name}",
        "out_root": str(run_root / "session"),
        "repo_root": str(repo),
        "hostrec_script": str(repo / "tools/gpu-burn/hostrec.sh"),
        "trees": {"old": tree(old_tree), "new": tree(repo)},
        "workloads": workloads,
        "pairs": 1,
        "warmups": 1,
        "measured": 2,
        "worker_timeout_s": 900,
        "session_time_budget_s": 3600,
        "precision_env": "MLX_ENABLE_TF32",
        "precision_value": "0",
    }
    run_root.mkdir(parents=True)
    (run_root / "spec.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {run_root / 'spec.json'} for {', '.join(selected)}")


if __name__ == "__main__":
    main()
