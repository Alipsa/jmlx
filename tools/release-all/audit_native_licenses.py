#!/usr/bin/env python3
"""Recheck pinned native license sources and packaged notices; no build or upload."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from release_inputs import (authored_policy, check_authored_attribution, native_notice_report,
                            read_json, sha256, validate_native_legal_inventory)

NOTICE_SEPARATOR = "======== JMLX NATIVE NOTICE SECTION ========"


def normalized(text):
    """Ignore line-end whitespace when matching a license excerpt, preserving its words."""
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def cache_path(entry, cache):
    return cache / (hashlib.sha256(entry["source"].encode("utf-8")).hexdigest()
                    + "-" + entry["sha256"])


def source_bytes(entry, cache, wheel):
    fetched = False
    if "archiveMember" in entry:
        with zipfile.ZipFile(wheel) as archive:
            data = archive.read(entry["archiveMember"])
    else:
        cached = cache_path(entry, cache)
        if cached.is_file():
            data = cached.read_bytes()
        else:
            url = entry["source"].replace("https://github.com/", "https://raw.githubusercontent.com/")
            url = url.replace("/blob/", "/", 1)
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            if entry.get("jsonBody"):
                try:
                    document = json.loads(data)
                except ValueError as error:
                    raise ValueError(f"unexpected Stack Exchange response: {entry['source']}") from error
                items = document.get("items") if isinstance(document, dict) else None
                if (not isinstance(items, list) or len(items) != 1
                        or not isinstance(items[0], dict) or not isinstance(items[0].get("body"), str)
                        or "error_id" in document):
                    raise ValueError(f"unexpected Stack Exchange response: {entry['source']}")
                item = items[0]
                if entry.get("revisionGuid") and (
                        item.get("revision_guid") != entry["revisionGuid"]
                        or item.get("user", {}).get("user_id") != entry["ownerUserId"]
                        or item.get("content_license") != entry["contentLicense"]):
                    raise ValueError("authored upstream revision identity differs")
                data = item["body"].encode("utf-8")
            fetched = True
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"source digest differs: {entry['source']}")
    if fetched:
        cache.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            os.replace(temporary, cached)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return data


def notice_sections(text):
    sections = {}
    for section in re.split(r"^" + re.escape(NOTICE_SEPARATOR) + r"[ \t]*$", text, flags=re.M):
        lines = section.strip().splitlines()
        if lines:
            title = lines[0]
            if title in sections:
                raise ValueError(f"duplicate native notice section: {title}")
            sections[title] = "\n".join(lines[1:])
    return sections


def check_attribution(entry, text, texts):
    name = entry.get("packagedAs", "THIRD-PARTY-NOTICES")
    if not isinstance(name, str) or name not in texts:
        raise ValueError(f"unknown packaged text for {entry['component']}: {name}")
    target = texts[name]
    if name == "THIRD-PARTY-NOTICES":
        sections = notice_sections(target)
        if entry["component"] not in sections:
            raise ValueError(f"missing native notice section: {entry['component']}")
        target = sections[entry["component"]]
    if "noticeFragments" in entry:
        fragments = entry["noticeFragments"]
        if not isinstance(fragments, list) or not all(isinstance(item, str) for item in fragments):
            raise ValueError(f"invalid notice fragments: {entry['component']}")
    elif "noticeLines" in entry:
        bounds = entry["noticeLines"]
        lines = text.splitlines()
        if (not isinstance(bounds, list) or len(bounds) != 2
                or not all(type(value) is int for value in bounds)
                or not 1 <= bounds[0] <= bounds[1] <= len(lines)):
            raise ValueError(f"invalid notice line range: {entry['component']}")
        fragments = ["\n".join(lines[bounds[0] - 1:bounds[1]])]
    else:
        fragments = [text]
    if not fragments or any(not normalized(fragment) or normalized(fragment) not in normalized(text)
                            or normalized(fragment) not in normalized(target) for fragment in fragments):
        raise ValueError(f"upstream attribution missing from packaged text: {entry['component']}")


def verify(root, wheel, cache, jars=()):
    inventory = read_json(root / "req/native-license-inventory.json")
    validate_native_legal_inventory(inventory)
    audit = read_json(root / "req/phase8-4-native-license-audit.json")
    if "scopeWordingImplementation" in audit:
        final_review = audit["scopeWordingImplementation"]
        if not isinstance(final_review, dict):
            raise ValueError("final owner acceptance record pin missing or invalid")
        acceptance = final_review.get("finalReviewAcceptance")
        expected = final_review.get("finalReviewAcceptanceSha256")
        if (not isinstance(acceptance, str) or not acceptance
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("final owner acceptance record pin missing or invalid")
        path = root / acceptance
        if (not path.resolve().is_relative_to(root.resolve()) or not path.is_file()
                or sha256(path) != expected):
            raise ValueError("final owner acceptance record missing or digest differs")
    if sha256(wheel) != inventory["wheelSha256"]:
        raise ValueError("wheel digest differs from reviewed inventory")
    legal = root / "jmlx-native-macos-arm64"
    texts = {}
    for name, digest in inventory["packagedTexts"].items():
        if sha256(legal / name) != digest:
            raise ValueError(f"packaged notice source differs: {name}")
        texts[name] = (legal / name).read_text(encoding="utf-8")
    checked = []
    for entry in inventory["components"]:
        text = source_bytes(entry, cache, wheel).decode("utf-8")
        check_attribution(entry, text, texts)
        checked.append({"component": entry["component"], "sha256": entry["sha256"]})
    for entry in audit["sourceEvidence"]:
        source_bytes(entry, cache, wheel)
    authored = check_authored_attribution((legal / "THIRD-PARTY-NOTICES").read_bytes(), root)
    policy = authored_policy(root)
    for section in policy["sections"]:
        for source in section["upstreamSources"]:
            source_bytes(source, cache, wheel)
    for jar in jars:
        native_notice_report(jar, inventory, root=root)
    return {"schema": 1, "inventorySha256": sha256(root / "req/native-license-inventory.json"),
            "components": checked, "authoredAttributions": authored,
            "authoredSourceEvidenceCount": sum(len(entry["upstreamSources"])
                                              for entry in policy["sections"]),
            "sourceEvidenceCount": len(audit["sourceEvidence"]),
            "sourceUrlDigestPairs": len({(entry["source"], entry["sha256"])
                                         for entry in inventory["components"] + audit["sourceEvidence"]
                                         if "archiveMember" not in entry}),
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
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.require_clear and not result["releaseReady"]:
        raise SystemExit("native license release gate remains open: " + ", ".join(result["unresolvedFindings"]))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        raise SystemExit(f"native license audit: {error}") from error
