#!/usr/bin/env python3
"""Strict, shared release selection and evidence primitives (Python standard library only)."""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

MODULE_KEYS = {
    "jmlx-jinja": "jmlxJinjaVersion",
    "jmlx-tokenizer": "jmlxTokenizerVersion",
    "jmlx-vision": "jmlxVisionVersion",
    "jmlx-native-macos-arm64": "jmlxNativeVersion",
    "jmlx-ffi": "jmlxFfiVersion",
    "jmlx-core": "jmlxCoreVersion",
    "jmlx-models": "jmlxModelsVersion",
    "jmlx-spring-ai": "jmlxSpringAiVersion",
    "jmlx-langchain4j": "jmlxLangchain4jVersion",
}
PURE = set(list(MODULE_KEYS)[:3])
NATIVE_FILES = ("libmlxc.dylib", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib",
                "native-pin.properties")
UPSTREAM_PATH = (b"/Users/runner/work/mlx/mlx/build/temp.macosx-11.0-arm64-cpython-310/"
                 b"mlx.core/mlx/backend/metal/kernels//mlx.metallib")
UPSTREAM_MLX_HASH = "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    raw = Path(path).read_bytes()
    if b"\r" in raw:
        raise ValueError(f"{path}: LF line endings required")
    return json.loads(raw, object_pairs_hook=unique_object)


def manifest(path, profile="all"):
    doc = read_json(path)
    if set(doc) != {"schema", "baselineSha", "audit", "modules"} or doc["schema"] != 1:
        raise ValueError("release manifest schema/keys differ")
    if not re.fullmatch(r"[0-9a-f]{40}", doc["baselineSha"]):
        raise ValueError("invalid baseline SHA")
    if not isinstance(doc["audit"], str) or not doc["audit"].startswith("req/"):
        raise ValueError("audit must name a repository document")
    if not isinstance(doc["modules"], list):
        raise ValueError("modules must be a list")
    entries = {}
    for entry in doc["modules"]:
        if not isinstance(entry, dict) or set(entry) != {"module", "version", "source"}:
            raise ValueError("module entry keys differ")
        name, version, source = entry["module"], entry["version"], entry["source"]
        if name not in MODULE_KEYS or name in entries:
            raise ValueError(f"unknown/duplicate module: {name}")
        if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
            raise ValueError(f"invalid release version for {name}: {version}")
        if source not in ("central", "local"):
            raise ValueError(f"invalid source for {name}: {source}")
        entries[name] = entry
    if entries.keys() != MODULE_KEYS.keys():
        raise ValueError("manifest must contain exactly nine modules")
    if profile not in ("all", "pure-java"):
        raise ValueError(f"unknown profile: {profile}")
    return {name: entry for name, entry in entries.items() if profile == "all" or name in PURE}


def macho_build(data):
    """Decode the ARM64 Mach-O LC_BUILD_VERSION without depending on host tools."""
    if len(data) < 32 or struct.unpack_from("<II", data) != (0xFEEDFACF, 0x100000C):
        raise ValueError("expected thin ARM64 Mach-O dylib")
    count, size = struct.unpack_from("<II", data, 16)
    end, offset, found = 32 + size, 32, []
    if end > len(data):
        raise ValueError("truncated Mach-O commands")
    for _ in range(count):
        if offset + 8 > end:
            raise ValueError("truncated Mach-O command")
        cmd, length = struct.unpack_from("<II", data, offset)
        if length < 8 or offset + length > end:
            raise ValueError("invalid Mach-O command length")
        if cmd == 0x32:
            if length < 24:
                raise ValueError("truncated LC_BUILD_VERSION")
            platform, minimum, sdk = struct.unpack_from("<III", data, offset + 8)
            if platform != 1:
                raise ValueError("expected macOS platform")
            decode = lambda n: (n >> 16, (n >> 8) & 255, n & 255)
            found.append({"minos": decode(minimum), "sdk": decode(sdk)})
        offset += length
    if len(found) != 1:
        raise ValueError("expected exactly one LC_BUILD_VERSION")
    return found[0]


def native_report(directory, bootstrap, forbidden):
    directory = Path(directory)
    script = Path(bootstrap).read_text()
    expected = {}
    for key, field in (("MLX_METAL_VERSION", "mlxMetalVersion"), ("MLX_C_COMMIT", "mlxcCommit")):
        match = re.search(r'^' + key + r'="([^"\n]+)"$', script, re.M)
        if not match:
            raise ValueError(f"bootstrap pin missing: {key}")
        expected[field] = match[1]
    pin_lines = (directory / "native-pin.properties").read_text().splitlines()
    pin = dict(line.split("=", 1) for line in pin_lines if line and not line.startswith("#"))
    if pin != expected:
        raise ValueError("staged native pins differ from bootstrap")
    target_match = re.search(r'^MACOS_DEPLOYMENT_TARGET="([0-9.]+)"$', script, re.M)
    if not target_match:
        raise ValueError("bootstrap deployment target missing")
    target = tuple(map(int, target_match[1].split(".")))
    target = (target + (0, 0, 0))[:3]
    result = {"pins": pin, "minimum": target, "files": {}}
    for name in NATIVE_FILES:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing/empty native file: {name}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        entry = {"sha256": digest, "size": len(data)}
        if name.endswith(".dylib"):
            build = macho_build(data)
            if build["minos"] > target:
                raise ValueError(f"{name}: minos {build['minos']} exceeds {target}")
            entry.update(build)
        if name != "native-pin.properties":
            for prefix in forbidden:
                if prefix and prefix.encode() in data:
                    raise ValueError(f"{name}: release-machine path leaked")
            users = re.findall(rb"/Users/[^\x00\n\r]+", data)
            if name == "libmlx.dylib" and digest == UPSTREAM_MLX_HASH:
                users = [s for s in users if s != UPSTREAM_PATH]
                entry["upstreamPathAllowlist"] = [UPSTREAM_PATH.decode()]
            if users:
                raise ValueError(f"{name}: unreviewed /Users/ path")
        result["files"][name] = entry
    return result


def native_jar_report(jar, bootstrap, forbidden):
    with tempfile.TemporaryDirectory(prefix="jmlx-native-verify-") as temp:
        with zipfile.ZipFile(jar) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate jar entries")
            for name in NATIVE_FILES:
                data = archive.read("se/alipsa/jmlx/native/macos-aarch64/" + name)
                (Path(temp) / name).write_bytes(data)
            for name in ("META-INF/LICENSE", "META-INF/NOTICE", "META-INF/LICENSE-APACHE-2.0"):
                if not archive.read(name).strip():
                    raise ValueError(f"missing native legal text: {name}")
            inventory = read_json(Path(bootstrap).resolve().parents[1] / "req/native-license-inventory.json")
            for name, digest in inventory["packagedTexts"].items():
                if hashlib.sha256(archive.read("META-INF/" + name)).hexdigest() != digest:
                    raise ValueError(f"native legal text differs from reviewed inventory: {name}")
        report = native_report(temp, bootstrap, forbidden)
        if inventory.get("nativePins") != report["pins"]:
            raise ValueError("native license inventory targets different upstream pins")
        return report


def central_exists(url):
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise ValueError(f"ambiguous repository status: {response.status}")
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise ValueError(f"repository status {exc.code}: {url}") from exc


def preflight(entries, root, versions=None, exists=central_exists):
    for name, entry in entries.items():
        if versions is None:
            text = (Path(root) / name / "build.gradle").read_text()
            matches = re.findall(r"^version = '([^']+)'$", text, re.M)
            if len(matches) != 1:
                raise ValueError(f"{name}: expected own version declaration")
            producer = matches[0]
        else:
            producer = versions.get(name)
        if producer != entry["version"]:
            raise ValueError(f"producer {name} version {producer} != manifest {entry['version']}")
        suffix = name.removeprefix("jmlx-").replace("-", "_").upper()
        env_key = "JMLX_" + ("NATIVE" if suffix == "NATIVE_MACOS_ARM64" else suffix) + "_VERSION"
        override = os.environ.get(env_key)
        if override is not None and override != producer:
            raise ValueError(f"{env_key} must repeat manifest version {producer}")
    # Complete all local validation before network probes. These probes perform no writes.
    for name, entry in entries.items():
        base = f"https://repo.maven.apache.org/maven2/se/alipsa/{name}/{entry['version']}/{name}-{entry['version']}"
        for extension in (".pom", ".module", ".jar", "-sources.jar", "-javadoc.jar"):
            present = exists(base + extension)
            if present != (entry["source"] == "central"):
                raise ValueError(f"{name}{extension}: Central availability disagrees with {entry['source']}")


def check_bundle(bundle, record_path, record_hash, manifest_path, root):
    if Path(bundle).stat().st_size > 1_000_000_000:
        raise ValueError("bundle exceeds conservative 1 GB Portal archive limit")
    if sha256(record_path) != record_hash:
        raise ValueError("preliminary record digest mismatch")
    record = read_json(record_path)
    if record.get("schema") != "jmlx-preliminary-bundle-1":
        raise ValueError("expected preliminary bundle record")
    if record.get("bundleSha256") != sha256(bundle):
        raise ValueError("bundle digest mismatch")
    if record.get("releaseManifestSha256") != sha256(manifest_path):
        raise ValueError("release manifest digest mismatch")
    tag = record["tag"]
    if not re.fullmatch(r"jmlx-phase8\.4-rc\.[1-9][0-9]*", tag["name"]):
        raise ValueError("invalid candidate tag namespace")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    ref = "refs/tags/" + tag["name"]
    if (git("cat-file", "-t", ref) != "tag" or git("rev-parse", ref) != tag["object"]
            or git("rev-parse", ref + "^{commit}") != tag["sha"]
            or git("rev-parse", "HEAD") != tag["sha"]):
        raise ValueError("candidate tag/object/HEAD mismatch")
    entries = manifest(manifest_path)
    allowed = {f"se/alipsa/{name}/{entry['version']}" for name, entry in entries.items()
               if entry["source"] == "local"}
    with zipfile.ZipFile(bundle) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError("unsafe bundle path")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("bundle symlinks are forbidden")
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate bundle entries")
        actual = set()
        for name in names:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("unsafe bundle path")
            parent = path.parent.as_posix()
            if parent not in allowed:
                raise ValueError(f"unexpected bundle coordinate: {parent}")
            actual.add(parent)
        if actual != allowed:
            raise ValueError("missing bundle coordinate")
        for name, entry in entries.items():
            if entry["source"] != "local":
                continue
            prefix = f"se/alipsa/{name}/{entry['version']}/{name}-{entry['version']}"
            for suffix in (".pom", ".module", ".jar", "-sources.jar", "-javadoc.jar"):
                if prefix + suffix not in names or prefix + suffix + ".asc" not in names:
                    raise ValueError(f"bundle missing artifact/signature: {prefix + suffix}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("manifest", type=Path)
    select.add_argument("--profile", default="all", choices=("all", "pure-java"))
    check = sub.add_parser("preflight")
    check.add_argument("manifest", type=Path)
    check.add_argument("--profile", default="all", choices=("all", "pure-java"))
    check.add_argument("--root", type=Path, required=True)
    native = sub.add_parser("native")
    native.add_argument("directory", type=Path)
    native.add_argument("--bootstrap", type=Path, required=True)
    native.add_argument("--forbid", action="append", default=[])
    bundle = sub.add_parser("bundle")
    bundle.add_argument("bundle", type=Path)
    bundle.add_argument("--record", type=Path, required=True)
    bundle.add_argument("--record-sha256", required=True)
    bundle.add_argument("--manifest", type=Path, required=True)
    bundle.add_argument("--root", type=Path, required=True)
    bundle.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.command == "select":
        for name, entry in manifest(args.manifest, args.profile).items():
            print("\t".join((name, entry["version"], entry["source"], MODULE_KEYS[name])))
    elif args.command == "preflight":
        preflight(manifest(args.manifest, args.profile), args.root)
    elif args.command == "bundle":
        check_bundle(args.bundle, args.record, args.record_sha256, args.manifest, args.root)
        if args.extract:
            if args.extract.exists():
                raise ValueError("bundle extraction directory must not exist")
            with zipfile.ZipFile(args.bundle) as archive:
                archive.extractall(args.extract)
    else:
        verify = native_jar_report if args.directory.suffix == ".jar" else native_report
        print(json.dumps(verify(args.directory, args.bootstrap, args.forbid), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
