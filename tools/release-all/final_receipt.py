#!/usr/bin/env python3
"""Create/revalidate final bundle evidence; only the final schema may authorize upload."""

import argparse
import json
import subprocess
from pathlib import Path

from release_inputs import check_bundle, native_report, read_json, sha256
from verify_native_tests import verify_events

REQUIRED = {"macos-full", "macos-pure", "linux-pure", "minimum-host"}


def verify(root, receipt_path):
    receipt = read_json(receipt_path)
    if receipt.get("schema") != "jmlx-verified-bundle-1":
        raise ValueError("only a final verified-bundle receipt is accepted")
    preliminary = Path(receipt["preliminary"]["path"])
    bundle = Path(receipt["bundle"])
    record = check_bundle(bundle, preliminary, receipt["preliminary"]["sha256"],
                          root / "tools/release-smoke/released-8.4.json", root)
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise ValueError("verified upload requires a clean checkout")
    plugin_receipt = root / "build/releaseAll/bundle-receipt.json"
    if sha256(plugin_receipt) != record["pluginReceiptSha256"]:
        raise ValueError("plugin receipt changed")
    if record["pluginReceipt"]["manifestHash"] != sha256(root / "build/releaseAll/manifest.json"):
        raise ValueError("plugin plan changed")
    actual = native_report(root / "native/install/lib", root / "scripts/bootstrap-native.sh",
                           [str(Path.home()), str(root)])
    if json.loads(json.dumps(actual)) != record["native"]:
        raise ValueError("staged native runtime changed")
    evidence = record["taskEvidence"]
    if sha256(evidence["path"]) != evidence["sha256"]:
        raise ValueError("task evidence changed")
    events = read_json(evidence["path"])
    verify_events(events, record["invocationId"])
    for path, digest in record["junit"].items():
        if sha256(root / path) != digest:
            raise ValueError(f"JUnit report changed: {path}")
    if set(receipt["reports"]) != REQUIRED:
        raise ValueError("required exact-bundle report set differs")
    for kind, report_file in receipt["reports"].items():
        if sha256(report_file["path"]) != report_file["sha256"]:
            raise ValueError(f"report digest changed: {kind}")
        report = read_json(report_file["path"])
        if (report.get("schema") != "jmlx-bundle-smoke-1" or report.get("kind") != kind
                or report.get("passed") is not True or report.get("bundleSha256") != record["bundleSha256"]
                or report.get("preliminarySha256") != receipt["preliminary"]["sha256"]
                or report.get("candidateSha") != record["tag"]["sha"]):
            raise ValueError(f"report identity/status mismatch: {kind}")
        env = report.get("environment", {})
        if report.get("consumerJavaFeature") != (21 if kind.endswith("pure") else 25):
            raise ValueError("consumer Java feature evidence differs")
        if not report.get("outputs"):
            raise ValueError("smoke result log digests missing")
        for name, digest in report["outputs"].items():
            if Path(name).name != name or sha256(Path(report_file["path"]).parent / name) != digest:
                raise ValueError(f"smoke output changed: {kind}/{name}")
        if kind == "minimum-host":
            target = ".".join(map(str, record["native"]["minimum"][:2]))
            if (env.get("system") != "Darwin" or env.get("osVersion", "").split(".")[:2] != target.split(".")
                    or env.get("architecture") != "arm64" or env.get("virtualized") is not False
                    or not report.get("runId") or not env.get("chip") or not env.get("swVers")):
                raise ValueError("minimum-host evidence is incomplete or incompatible")
        elif kind == "linux-pure" and (env.get("system") != "Linux" or env.get("architecture") != "aarch64"
                                       or not env.get("containerImage") or not env.get("osRelease")):
            raise ValueError("expected local Linux ARM64 evidence")
        elif kind.startswith("macos-") and env.get("system") != "Darwin":
            raise ValueError("expected macOS evidence")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--preliminary", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            verify(args.root.resolve(), args.verify)
        else:
            if not args.preliminary or not args.bundle or not args.output:
                raise ValueError("creation requires preliminary, bundle and output")
            reports = {}
            for pair in args.report:
                kind, path = pair.split("=", 1)
                if kind in reports:
                    raise ValueError("duplicate report kind")
                reports[kind] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
            receipt = {"schema": "jmlx-verified-bundle-1", "bundle": str(args.bundle.resolve()),
                       "preliminary": {"path": str(args.preliminary.resolve()), "sha256": sha256(args.preliminary)},
                       "reports": reports}
            # Validate a temporary representation before writing the immutable final record.
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as temp:
                json.dump(receipt, temp)
                temp.flush()
                verify(args.root.resolve(), Path(temp.name))
            with args.output.open("x") as stream:
                json.dump(receipt, stream, indent=2)
                stream.write("\n")
            print(sha256(args.output))
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")
