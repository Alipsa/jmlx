#!/usr/bin/env python3
"""Create/revalidate final bundle evidence; only the final schema may authorize upload."""

import argparse
import json
import subprocess
from pathlib import Path

import owner_acceptance

from release_inputs import (check_bundle, native_report, read_json, read_runtime, sha256,
                            verify_smoke_environment, verify_minimum_host_workflow)
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
    plugin_bundle = root / "build/releaseAll/jmlx-mega-bundle.zip"
    if bundle.resolve() != plugin_bundle.resolve() or sha256(plugin_bundle) != record["bundleSha256"]:
        raise ValueError("verified bundle must be the unchanged ZIP configured for plugin upload")
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
    argv_file = record["gradleArgvFile"]
    if sha256(argv_file["path"]) != argv_file["sha256"] or read_json(argv_file["path"]) != record["gradleArgv"]:
        raise ValueError("bundle-producing argument vector changed")
    for key, feature in (("releaseJdk", "25"), ("pureJavaJdk", "21")):
        jdk = record.get(key, {})
        if jdk.get("java.feature") != feature or not all(jdk.get(name) for name in
                ("java.version", "java.runtime.version", "java.vendor", "java.home", "java.vm.name")):
            raise ValueError("bundle-producing JDK identity missing")
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
        runtime = report.get("consumerRuntime", {})
        if (runtime.get("java.feature") != str(21 if kind.endswith("pure") else 25)
                or not all(runtime.get(key) for key in ("java.version", "java.vendor", "java.home"))):
            raise ValueError("consumer Java feature evidence differs")
        if not report.get("outputs"):
            raise ValueError("smoke result log digests missing")
        for name, digest in report["outputs"].items():
            if Path(name).name != name or sha256(Path(report_file["path"]).parent / name) != digest:
                raise ValueError(f"smoke output changed: {kind}/{name}")
        runtime_file = report.get("consumerRuntimeFile")
        if (not runtime_file or runtime_file not in report["outputs"]
                or read_runtime(Path(report_file["path"]).parent / runtime_file) != runtime):
            raise ValueError("consumer runtime measurement differs from its evidence file")
        verify_smoke_environment(kind, env, record["native"]["supportedMinimum"], report.get("runId"))
        if kind == "minimum-host":
            verify_minimum_host_workflow(report.get("workflow"))
    return receipt, record


def plugin_digests(root, receipt_path):
    _, record = verify(root, receipt_path)
    return record["bundleSha256"], record["pluginReceipt"]["manifestHash"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--plugin-digests", action="store_true",
                        help="with --verify, print reviewed ZIP and manifest hashes separated by a tab")
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--acceptance-sha256")
    parser.add_argument("--preliminary", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            if args.plugin_digests:
                _, record = verify(args.root.resolve(), args.verify)
                owner_acceptance.verify(args.root.resolve(), args.verify, record,
                                        args.acceptance, args.acceptance_sha256)
                print("\t".join((record["bundleSha256"], record["pluginReceipt"]["manifestHash"])))
            else:
                verify(args.root.resolve(), args.verify)
        else:
            if args.plugin_digests:
                raise ValueError("--plugin-digests requires --verify")
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
                       "identityAssurance": "Local runtime measurements with operator-attested workflow/container identity; not authenticated remote provenance",
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
