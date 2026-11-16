#!/usr/bin/env python3
"""Compare flat publication directories, preserving artifact content and metadata changes."""

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


def digest(data, algorithm="sha256"):
    return hashlib.new(algorithm, data).hexdigest()


def archive_content(data):
    """Canonical ZIP: sorted names, fixed timestamps, stored uncompressed entry bytes."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entry")
        entries = {name: archive.read(name) for name in names}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(entries):
            archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), entries[name])
    return output.getvalue(), entries


def module_content(data, directory):
    """Recompute only artifact sizes/checksums, after validating the original references."""
    metadata = json.loads(data)
    replacements = []
    for variant in metadata["variants"]:
        for artifact in variant.get("files", []):
            name = artifact["url"]
            if Path(name).name != name or not name.endswith(".jar"):
                raise ValueError(f"expected a sibling JAR reference: {name}")
            original = (directory / name).read_bytes()
            if artifact["size"] != len(original):
                raise ValueError(f"artifact size mismatch: {name}")
            normalized, _ = archive_content(original)
            changes = {"size": len(normalized)}
            for algorithm in ("sha512", "sha256", "sha1", "md5"):
                if algorithm in artifact:
                    if artifact[algorithm] != digest(original, algorithm):
                        raise ValueError(f"artifact {algorithm} mismatch: {name}")
                    changes[algorithm] = digest(normalized, algorithm)
            replacements.append((artifact, changes))

    # Preserve the original JSON formatting and every byte outside the validated fields.
    # Gradle file records are flat objects; reject an unsupported shape rather than silently
    # rewriting unrelated metadata with a parse/serialize round trip.
    def replace_record(match):
        record = match.group()
        if '"url"' not in record:
            return record
        original = json.loads(record)
        for index, (artifact, changes) in enumerate(replacements):
            if original == artifact:
                replacements.pop(index)
                for key, value in changes.items():
                    record, count = re.subn(
                        rf'("{key}"\s*:\s*)("[^"\\]*"|[0-9]+)',
                        lambda token: token[1] + json.dumps(value), record)
                    if count != 1:
                        raise ValueError(f"unsupported artifact field: {key}")
                return record
        raise ValueError("unexpected URL record in module metadata")

    result = re.sub(r'\{[^{}]*\}', replace_record, data.decode("utf-8"))
    if replacements:
        raise ValueError("unsupported artifact record in module metadata")
    return result.encode("utf-8")


def compare(central, candidate):
    """Compare every JAR, POM and module file in both directories; missing files fail."""
    def files(directory):
        return {path.name for path in directory.iterdir()
                if path.suffix in (".jar", ".pom", ".module")}

    names = files(central)
    if not names or names != files(candidate):
        raise ValueError("publication file sets differ or are empty")
    results = {}
    for name in sorted(names):
        left = (central / name).read_bytes()
        right = (candidate / name).read_bytes()
        before = {"central": digest(left), "candidate": digest(right)}
        changed_entries = None
        if name.endswith(".jar"):
            left, left_entries = archive_content(left)
            right, right_entries = archive_content(right)
            changed_entries = [entry for entry in sorted(left_entries.keys() | right_entries.keys())
                               if left_entries.get(entry) != right_entries.get(entry)]
        elif name.endswith(".module"):
            left = module_content(left, central)
            right = module_content(right, candidate)
        result = {"rawSha256": before,
                  "normalizedSha256": {"central": digest(left), "candidate": digest(right)},
                  "equal": left == right}
        if changed_entries is not None:
            result["changedEntries"] = changed_entries
        results[name] = result
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.central, args.candidate), indent=2) + "\n")
