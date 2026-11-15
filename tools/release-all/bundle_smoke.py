#!/usr/bin/env python3
"""Run an exact-bundle consumer and record its environment and output digests."""

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path

from release_inputs import check_bundle, sha256


def output(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def environment():
    env = {"system": platform.system(), "architecture": platform.machine(), "kernel": platform.release(),
           "launcherJava": output("java", "-version")}
    if env["system"] == "Darwin":
        env.update(osVersion=output("sw_vers", "-productVersion"), swVers=output("sw_vers"),
                   chip=output("sysctl", "-n", "machdep.cpu.brand_string"),
                   virtualized=output("sysctl", "-n", "kern.hv_vmm_present") != "0")
    else:
        env["osRelease"] = Path("/etc/os-release").read_text()
        env["containerImage"] = os.environ.get("JMLX_LINUX_CONTAINER_IMAGE_ID", "")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--preliminary", type=Path, required=True)
    parser.add_argument("--preliminary-sha256", required=True)
    parser.add_argument("--kind", choices=("macos-full", "macos-pure", "linux-pure", "minimum-host"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        raise ValueError("report is immutable; use a new output path")
    record = check_bundle(args.bundle, args.preliminary, args.preliminary_sha256,
                          root / "tools/release-smoke/released-8.4.json", root)
    env = environment()
    if args.kind == "minimum-host":
        if (env["system"] != "Darwin" or env["architecture"] != "arm64" or env["virtualized"]
                or list(map(int, env["osVersion"].split(".")[:2])) != record["native"]["minimum"][:2]):
            raise ValueError("physical minimum-host OS/architecture requirements not met")
        if not os.environ.get("GITHUB_RUN_ID"):
            raise ValueError("minimum-host workflow run identity missing")
    elif args.kind == "linux-pure":
        if env["system"] != "Linux" or env["architecture"] != "aarch64" or not env["containerImage"]:
            raise ValueError("local Ubuntu ARM64 container identity required")
    elif env["system"] != "Darwin":
        raise ValueError("macOS report requested on another OS")
    profile = "pure-java" if args.kind.endswith("pure") else "all"
    command = ["bash", str(root / "tools/release-smoke/run.sh"), "candidate", "--profile", profile,
               "--bundle", str(args.bundle.resolve()), "--bundle-record", str(args.preliminary.resolve()),
               "--bundle-record-sha256", args.preliminary_sha256]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix(".log")
    with log.open("x") as stream:
        subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
    report = {"schema": "jmlx-bundle-smoke-1", "kind": args.kind, "passed": True,
              "bundleSha256": record["bundleSha256"], "preliminarySha256": args.preliminary_sha256,
              "candidateSha": record["tag"]["sha"], "environment": env, "command": command,
              "consumerJavaFeature": 21 if profile == "pure-java" else 25,
              "runId": os.environ.get("GITHUB_RUN_ID"), "outputs": {log.name: sha256(log)}}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(sha256(args.output))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"bundle smoke failed: {exc}") from exc
