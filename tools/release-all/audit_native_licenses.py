#!/usr/bin/env python3
"""Recheck pinned native license sources and packaged notices; no build or upload."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from release_inputs import native_notice_report, read_json, sha256


def normalized(text):
    """Ignore line-end whitespace when matching a license excerpt, preserving its words."""
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def source_bytes(entry, cache, wheel):
    if "archiveMember" in entry:
        with zipfile.ZipFile(wheel) as archive:
            data = archive.read(entry["archiveMember"])
    else:
        cached = cache / entry["sha256"]
        if cached.is_file():
            data = cached.read_bytes()
        else:
            url = entry["source"].replace("https://github.com/", "https://raw.githubusercontent.com/")
            url = url.replace("/blob/", "/")
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            if entry.get("jsonBody"):
                data = json.loads(data)["items"][0]["body"].encode()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"source digest differs: {entry['source']}")
        cache.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"source digest differs: {entry['source']}")
    return data


def verify(root, wheel, cache, jars=()):
    inventory = read_json(root / "req/native-license-inventory.json")
    audit = read_json(root / "req/phase8-4-native-license-audit.json")
    if sha256(wheel) != inventory["wheelSha256"]:
        raise ValueError("wheel digest differs from reviewed inventory")
    legal = root / "jmlx-native-macos-arm64"
    texts = {}
    for name, digest in inventory["packagedTexts"].items():
        if sha256(legal / name) != digest:
            raise ValueError(f"packaged notice source differs: {name}")
        texts[name] = (legal / name).read_text()
    checked = []
    for entry in inventory["components"]:
        text = source_bytes(entry, cache, wheel).decode()
        target = texts[entry.get("packagedAs", "THIRD-PARTY-NOTICES")]
        if "noticeFragments" in entry:
            fragments = entry["noticeFragments"]
        elif "noticeLines" in entry:
            start, end = entry["noticeLines"]
            fragments = ["\n".join(text.splitlines()[start - 1:end])]
        else:
            fragments = [text]
        if not fragments or any(not normalized(fragment) or normalized(fragment) not in normalized(text)
                                or normalized(fragment) not in normalized(target) for fragment in fragments):
            raise ValueError(f"upstream attribution missing from packaged text: {entry['component']}")
        checked.append({"component": entry["component"], "sha256": entry["sha256"]})
    for entry in audit["sourceEvidence"]:
        source_bytes(entry, cache, wheel)
    for jar in jars:
        native_notice_report(jar, inventory)
    return {"schema": 1, "inventorySha256": sha256(root / "req/native-license-inventory.json"),
            "components": checked, "sourceEvidenceCount": len(audit["sourceEvidence"]),
            "packagedTexts": inventory["packagedTexts"], "checkedJars": [jar.name for jar in jars],
            "releaseReady": not audit["unresolvedFindings"],
            "unresolvedFindings": [entry["id"] for entry in audit["unresolvedFindings"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--jar", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-clear", action="store_true")
    args = parser.parse_args()
    result = verify(args.root, args.wheel, args.cache, args.jar)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.require_clear and not result["releaseReady"]:
        raise SystemExit("native license release gate remains open: " + ", ".join(result["unresolvedFindings"]))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f"native license audit: {error}") from error
