#!/usr/bin/env python3
"""Compare flat publication directories, preserving artifact content and metadata changes."""

import argparse
import copy
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


def digest(data, algorithm="sha256"):
    return hashlib.new(algorithm, data).hexdigest()


def archive_content(data):
    """Use sorted, timestamp-fixed stored bytes; compare original ZIP metadata separately."""
    with zipfile.ZipFile(io.BytesIO(data)) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entry")
        entries = {}
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as target:
            target.comment = source.comment
            for name in sorted(names):
                info = copy.copy(source.getinfo(name))
                info.date_time = (1980, 1, 1, 0, 0, 0)
                content = source.read(name)
                entries[name] = (content, info.compress_type, info.external_attr,
                                 info.internal_attr, info.extra, info.comment,
                                 info.create_system, info.create_version, info.extract_version,
                                 info.flag_bits, info.reserved, info.volume)
                info.compress_type = zipfile.ZIP_STORED
                target.writestr(info, content)
    return output.getvalue(), entries


def module_content(data, directory):
    """Recompute only artifact sizes/checksums, after validating the original references."""
    metadata = json.loads(data)
    replacements = []
    if not isinstance(metadata, dict) or not isinstance(metadata.get("variants"), list):
        raise ValueError("module metadata requires a variants list")
    for variant in metadata["variants"]:
        for artifact in variant.get("files", []):
            if not isinstance(artifact, dict) or not isinstance(artifact.get("url"), str):
                raise ValueError("artifact record requires a string url")
            name = artifact["url"]
            if type(artifact.get("size")) is not int or artifact["size"] < 0:
                raise ValueError(f"artifact record requires a nonnegative integer size: {name}")
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


def compare(central, candidate, publications=None):
    """Compare every JAR, POM and module file in both directories; missing files fail."""
    def files(directory):
        return {path.name for path in directory.iterdir()
                if path.suffix in (".jar", ".pom", ".module")}

    names = files(central)
    if not names or names != files(candidate):
        raise ValueError("publication file sets differ or are empty")
    if publications is not None:
        if (not isinstance(publications, dict) or not publications
                or any(not isinstance(module, str) or not module
                       or not isinstance(version, str) or not version
                       for module, version in publications.items())):
            raise ValueError("publications must be a nonempty object mapping module names to version strings")
        expected = {f"{module}-{version}{suffix}"
                    for module, version in publications.items()
                    for suffix in (".jar", "-sources.jar", "-javadoc.jar", ".pom", ".module")}
        if names != expected:
            raise ValueError(f"unexpected publication file set; missing={sorted(expected - names)}, "
                             f"extra={sorted(names - expected)}")
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
                  "equal": left == right and not changed_entries}
        if changed_entries is not None:
            result["changedEntries"] = changed_entries
        results[name] = result
    return {"allEqual": all(result["equal"] for result in results.values()), "artifacts": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--publications", type=Path,
                        help="JSON object mapping expected module names to versions (five files each)")
    parser.add_argument("--require-equal", action="store_true",
                        help="exit 1 after writing the report if any artifact differs")
    args = parser.parse_args()
    publications = json.loads(args.publications.read_text()) if args.publications else None
    result = compare(args.central, args.candidate, publications)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(1 if args.require_equal and not result["allEqual"] else 0)
