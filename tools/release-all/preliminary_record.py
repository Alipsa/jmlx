#!/usr/bin/env python3
"""Bind a completed bundle-producing run to its candidate tag and native/test evidence."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from release_inputs import native_jar_report, native_report, read_json, sha256
from verify_native_tests import verify, verify_events


def create(root, tag, events_path, invocation):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if not re.fullmatch(r"jmlx-phase8\.4-rc\.[1-9][0-9]*", tag):
        raise ValueError("invalid candidate tag namespace")
    ref = "refs/tags/" + tag
    if git("cat-file", "-t", ref) != "tag" or git("status", "--porcelain"):
        raise ValueError("candidate must be annotated with a clean checkout")
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", ref + "^{commit}"):
        raise ValueError("candidate tag does not identify HEAD")
    build = root / "build/releaseAll"
    bundle = build / "jmlx-mega-bundle.zip"
    plugin_receipt = build / "bundle-receipt.json"
    receipt = read_json(plugin_receipt)
    if receipt["bundleHash"] != sha256(bundle) or receipt["manifestHash"] != sha256(build / "manifest.json"):
        raise ValueError("plugin bundle receipt mismatch")
    events = read_json(events_path)
    verify_events(events, invocation)
    bootstrap = root / "scripts/bootstrap-native.sh"
    staged = native_report(root / "native/install/lib", bootstrap, [str(Path.home()), str(root)])
    import tempfile
    import zipfile
    from release_inputs import manifest
    native = manifest(root / "tools/release-smoke/released-8.4.json")["jmlx-native-macos-arm64"]
    native_path = f"se/alipsa/jmlx-native-macos-arm64/{native['version']}/jmlx-native-macos-arm64-{native['version']}.jar"
    with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(bundle) as archive:
        jar = Path(temp) / "native.jar"
        jar.write_bytes(archive.read(native_path))
        packaged = native_jar_report(jar, bootstrap, [str(Path.home()), str(root)])
    if packaged != staged:
        raise ValueError("bundle native bytes differ from staged runtime")
    toolchain = {}
    for name, command in {
        "cc": ["cc", "--version"], "cmake": ["cmake", "--version"],
        "sdk": ["xcrun", "--show-sdk-version"], "xcode": ["xcodebuild", "-version"],
    }.items():
        toolchain[name] = subprocess.check_output(command, text=True).strip()
    return {
        "schema": "jmlx-preliminary-bundle-1",
        "tag": {"name": tag, "object": git("rev-parse", ref), "sha": head},
        "bundleSha256": sha256(bundle), "bundleBytes": bundle.stat().st_size,
        "pluginReceiptSha256": sha256(plugin_receipt),
        "pluginReceipt": receipt,
        "releaseManifestSha256": sha256(root / "tools/release-smoke/released-8.4.json"),
        "native": staged, "toolchain": toolchain, "invocationId": invocation,
        "taskEvidence": {"path": str(events_path.resolve()), "sha256": sha256(events_path)},
        "junit": verify(root),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--invocation", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = create(args.root.resolve(), args.tag, args.events, args.invocation)
        # A preliminary record is immutable. A new bundle requires a new record path.
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(sha256(args.output))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")
