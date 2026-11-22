#!/usr/bin/env python3
"""Strict, shared release selection and evidence primitives (Python standard library only)."""

import argparse
import hashlib
import io
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
from xml.etree import ElementTree

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required for release verification")

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
REQUIRED_NATIVE_LEGAL_TEXTS = {"LICENSE", "NOTICE", "LICENSE-APACHE-2.0", "THIRD-PARTY-NOTICES",
                               "LICENSE-CC-BY-SA-4.0"}
UPSTREAM_PATH = (b"/Users/runner/work/mlx/mlx/build/temp.macosx-11.0-arm64-cpython-310/"
                 b"mlx.core/mlx/backend/metal/kernels//mlx.metallib")
UPSTREAM_MLX_HASH = "d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6"

ROOT = Path(__file__).resolve().parents[2]
NOTICE_SEPARATOR = b"======== JMLX NATIVE NOTICE SECTION ========"
REQUIRED_NATIVE_AUTHORED_SECTIONS = frozenset({'Norbert Juffa erf / erfinv adaptations (source records: CC BY-SA 4.0)'})
REQUIRED_FINAL_ACCEPTANCE = (
    "req/plans/evidence/native-distribution-review-2026-11-22/owner-acceptance.json")
REQUIRED_AUTHORED_FIELDS = {"creator", "profile", "revisions", "license", "modifications", "scope"}
APPROVED_NOTICE_PERMALINK = 'https://github.com/Alipsa/jmlx/blob/551eebeeb66a80ef3af467a6c1b4995c2a7f7034/jmlx-native-macos-arm64/THIRD-PARTY-NOTICES'
REQUIRED_NATIVE_POM_LICENSES = (('MIT License',
  'https://opensource.org/license/mit',
  'Applies to the jmlx native artifact code, mlx-c, Apple/MLX contributions covered by their '
  'existing MIT grants, and the other MIT components listed in META-INF/THIRD-PARTY-NOTICES, '
  'including nlohmann/json, fmt, gguf-tools, FP16 and supervised-lda. MIT notices are also '
  'retained there for the opt_einsum algorithm attribution and VkFFT algorithm reference. The '
  'reviewed Juffa expression has separately identified CC BY-SA 4.0 terms.'),
 ('Apache License, Version 2.0',
  'https://www.apache.org/licenses/LICENSE-2.0.txt',
  'Applies to Apache-2.0 components identified in META-INF/THIRD-PARTY-NOTICES, including vendored '
  'metal-cpp compiled into libmlx.dylib and the NVIDIA/Filipe RNC Maia cexpf and TensorFlow '
  'adaptations. An Apache-2.0 notice is also retained there for the JAX Threefry algorithm '
  'reference.'),
 ('Creative Commons Attribution-ShareAlike 4.0 International',
  'https://creativecommons.org/licenses/by-sa/4.0/',
  "Applies to Norbert Juffa's covered expression in the reviewed Metal erf and Metal/CPU SIMD "
  'erfinv material, including corresponding compiled forms in libmlx.dylib and mlx.metallib. '
  "Apple/MLX contributions rely separately on existing MIT grants. No CC adapter's license is "
  'identified. See '
  f'<{APPROVED_NOTICE_PERMALINK}> '
  '(scope and redistribution basis), also packaged in META-INF/THIRD-PARTY-NOTICES. The complete '
  'CC license is packaged in META-INF/LICENSE-CC-BY-SA-4.0.'),
 ('BSD 2-Clause License',
  'https://opensource.org/license/bsd-2-clause',
  'Applies to BSD-2-Clause material identified in META-INF/THIRD-PARTY-NOTICES, including Norbert '
  'Juffa expm1f.'),
 ('BSD 3-Clause License',
  'https://opensource.org/license/bsd-3-clause',
  'Applies to BSD-3-Clause components identified in META-INF/THIRD-PARTY-NOTICES, including '
  'pocketfft, V8, NumPy, fastermath and PyTorch (including Metal hypot). BSD-3-Clause notices are '
  'also retained there for the Random123 Threefry attribution and CUB merge-sort algorithm '
  'reference.'),
 ('zlib License',
  'https://opensource.org/license/zlib',
  'Applies to zlib components identified in META-INF/THIRD-PARTY-NOTICES, including ThreadPool and '
  'the Julien Pommier NEON math adaptation.'))


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


def read_runtime(path):
    properties = ElementTree.parse(path).getroot()
    pairs = [(entry.attrib["key"], entry.text or "") for entry in properties.findall("entry")]
    result = unique_object(pairs)
    fields = {"java.feature", "java.version", "java.runtime.version", "java.vendor", "java.home", "java.vm.name"}
    if properties.tag != "properties" or set(result) != fields or not all(result.values()):
        raise ValueError("incomplete measured Java runtime evidence")
    if not result["java.feature"].isdigit():
        raise ValueError("invalid measured Java feature version")
    return result


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
    minimum_match = re.search(r'^MACOS_SUPPORTED_MINIMUM="([0-9]+\.[0-9]+)"$', script, re.M)
    if not minimum_match:
        raise ValueError('bootstrap supported minimum missing: expected top-level '
                         'MACOS_SUPPORTED_MINIMUM="X.Y"')
    minimum = tuple(map(int, minimum_match[1].split("."))) + (0,)
    if target > minimum:
        raise ValueError("deployment target exceeds supported minimum")
    result = {"pins": pin, "supportedMinimum": minimum, "deploymentTarget": target, "files": {}}
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
            scanned = data
            if name == "libmlx.dylib" and digest == UPSTREAM_MLX_HASH:
                scanned = scanned.replace(UPSTREAM_PATH, b"")
                entry["upstreamPathAllowlist"] = [UPSTREAM_PATH.decode()]
            for prefix in forbidden:
                if prefix and prefix.encode() in scanned:
                    raise ValueError(f"{name}: release-machine path leaked")
            users = re.findall(rb"/Users/[^\x00\n\r]+", scanned)
            if users:
                raise ValueError(f"{name}: unreviewed /Users/ path")
        result["files"][name] = entry
    return result


def verify_minimum_host_workflow(identity):
    """Validate the reported manual workflow identity; review the actual run independently."""
    if (not isinstance(identity, dict) or identity.get("ref") != "refs/heads/main"
            or identity.get("event") != "workflow_dispatch"
            or identity.get("workflowRef") !=
            "Alipsa/jmlx/.github/workflows/release-candidate.yml@refs/heads/main"
            or not isinstance(identity.get("sha"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", identity["sha"])):
        raise ValueError("minimum-host workflow must identify the manual main-branch release workflow")


def verify_smoke_environment(kind, env, supported_minimum, run_id=None):
    """Require supported macOS acceptance hosts and physical minimum-series execution."""
    if kind == "linux-pure":
        if (env.get("system") != "Linux" or env.get("architecture") != "aarch64"
                or not env.get("containerImage") or not env.get("osRelease")):
            raise ValueError("expected local Linux ARM64 evidence")
        return
    if kind not in ("minimum-host", "macos-full", "macos-pure"):
        raise ValueError(f"unknown smoke kind: {kind}")
    error = ("minimum-host evidence is incomplete or incompatible" if kind == "minimum-host"
             else "expected supported macOS evidence")
    version = env.get("osVersion")
    if (env.get("system") != "Darwin" or env.get("architecture") != "arm64"
            or not isinstance(version, str)
            or not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", version)):
        raise ValueError(error)
    actual = tuple(map(int, version.split(".")[:2]))
    minimum = tuple(supported_minimum[:2])
    if kind == "minimum-host":
        if (actual != minimum or env.get("virtualized") is not False or not run_id
                or not env.get("chip") or not env.get("swVers")):
            raise ValueError(error)
    elif actual < minimum:
        raise ValueError(error)


def raw_notice_sections(data):
    """Extract untouched sections using only complete LF/CRLF separator lines."""
    boundaries = []
    offset = 0
    physical = data.split(b"\n")
    for number, part in enumerate(physical, 1):
        terminated = number < len(physical)
        line = part[:-1] if part.endswith(b"\r") else part
        if line.rstrip(b" \t") == NOTICE_SEPARATOR:
            if line != NOTICE_SEPARATOR:
                raise ValueError(f"malformed-separator at line {number}")
            if terminated:
                boundaries.append((offset, offset + len(part) + 1))
        offset += len(part) + (1 if terminated else 0)
    result = {}
    for index, (_, start) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(data)
        section = data[start:end]
        title = None
        for part in section.split(b"\n"):
            line = part[:-1] if part.endswith(b"\r") else part
            text = line.decode("utf-8", errors="strict")
            if not text.strip(" \t"):
                continue
            first = text.lstrip(" \t")
            if first[0].isspace():
                raise ValueError("unsupported-leading-whitespace before native notice title")
            title = text
            break
        if title is None:
            raise ValueError("empty native notice section")
        if title in result:
            raise ValueError(f"duplicate native notice section: {title}")
        result[title] = section
    return result


def authored_policy(root=ROOT):
    """Validate authored policy coverage and its independently pinned approval evidence."""
    root = Path(root)
    try:
        policy = read_json(root / "req/native-authored-attributions.json")
    except OSError as error:
        raise ValueError("required native authored-attribution policy missing") from error
    sections = policy.get("sections", [])
    titles = [entry.get("title") for entry in sections]
    if (policy.get("schema") != 1 or len(titles) != len(set(titles))
            or set(titles) != REQUIRED_NATIVE_AUTHORED_SECTIONS):
        raise ValueError("required native authored-attribution policy coverage differs")
    evidence = policy.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("authored policy lacks approval evidence")
    checked = set()
    for entry in evidence:
        path = root / entry["path"]
        if (not path.resolve().is_relative_to(root.resolve()) or not path.is_file()
                or sha256(path) != entry["sha256"]):
            raise ValueError(f"authored policy evidence differs: {entry['path']}")
        checked.add(entry["path"])
    decision = root / policy["decision"]
    if not decision.resolve().is_relative_to(root.resolve()) or not decision.is_file():
        raise ValueError("authored policy decision record missing")
    for entry in sections:
        fields = entry.get("requirements", {})
        if set(fields) != REQUIRED_AUTHORED_FIELDS:
            raise ValueError("authored policy required fields differ")
        for value in fields.values():
            fragments = value if isinstance(value, list) else [value]
            if not fragments or any(not isinstance(x, str) or not x.strip() for x in fragments):
                raise ValueError("invalid authored policy requirement")
        if entry.get("approvedManifest") not in checked:
            raise ValueError("authored policy manifest is not pinned evidence")
        approved = read_json(root / entry["approvedManifest"])
        artifact = next((a for a in approved.get("artifacts", [])
                         if a.get("path") == "juffa-notice-candidate.txt"), {})
        if entry.get("sectionSha256") != artifact.get("sectionSha256"):
            raise ValueError("authored section digest differs from approved manifest")
        if not entry.get("upstreamSources"):
            raise ValueError("authored policy upstream evidence missing")
    return policy


def check_authored_attribution(data, root=ROOT):
    """Check authored text without pretending it is an upstream excerpt."""
    sections = raw_notice_sections(data)
    if not REQUIRED_NATIVE_AUTHORED_SECTIONS <= sections.keys():
        raise ValueError("required native authored notice section missing")
    policy = authored_policy(root)
    result = {}
    for entry in policy["sections"]:
        section = sections[entry["title"]]
        text = section.decode("utf-8", errors="strict")
        for key, value in entry["requirements"].items():
            for fragment in value if isinstance(value, list) else [value]:
                if fragment not in text:
                    raise ValueError(f"authored attribution missing {key}: {entry['title']}")
        digest = hashlib.sha256(section).hexdigest()
        if digest != entry["sectionSha256"]:
            raise ValueError("authored attribution raw section differs from approved bytes")
        result[entry["title"]] = digest
    return result


def native_pom_report(data):
    """Compare complete literal license tuples independently of producer Gradle logic."""
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("native POM document declarations are forbidden")
    try:
        project = ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise ValueError("invalid native POM XML") from error
    namespace = "{http://maven.apache.org/POM/4.0.0}" if project.tag.startswith("{") else ""
    if project.tag != namespace + "project":
        raise ValueError("invalid native POM root")
    groups = project.findall(namespace + "licenses")
    if len(groups) != 1:
        raise ValueError("native POM license entries differ")
    actual = []
    for entry in groups[0]:
        fields = [child.tag for child in entry]
        if (entry.tag != namespace + "license" or len(fields) != 3
                or set(fields) != {namespace + key for key in ("name", "url", "comments")}
                or any(len(child) for child in entry)):
            raise ValueError("native POM license fields differ")
        actual.append(tuple(entry.findtext(namespace + key, "") for key in ("name", "url", "comments")))
    if len(set(actual)) != len(actual) or sorted(actual) != sorted(REQUIRED_NATIVE_POM_LICENSES):
        raise ValueError("native POM license entries differ from approved literal tuples")
    return {"licenses": len(actual), "noticePermalink": APPROVED_NOTICE_PERMALINK}


def verify_notice_permalink(root, data, *, reviewed_ref="refs/remotes/origin/main",
                            candidate_ref="HEAD"):
    """Require the linked commit in fetched origin/main and candidate history, then compare bytes."""
    root = Path(root)
    match = re.fullmatch(r"https://github.com/Alipsa/jmlx/blob/([0-9a-f]{40})/"
                         r"jmlx-native-macos-arm64/THIRD-PARTY-NOTICES", APPROVED_NOTICE_PERMALINK)
    if not match:
        raise ValueError("invalid approved notice permalink")
    commit = match[1]
    local = raw_notice_sections(data)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
    try:
        if git("rev-parse", "--is-shallow-repository").strip() == b"true":
            raise ValueError("approved notice history unavailable; shallow checkout requires "
                             "full history: run git fetch --unshallow origin before explicit "
                             "release verification")
        origin_error = ("origin must point to Alipsa/jmlx on github.com for approved "
                        "notice verification")
        try:
            origin = git("remote", "get-url", "origin").decode("utf-8").strip()
        except subprocess.CalledProcessError as error:
            raise ValueError(origin_error) from error
        if not re.fullmatch(r"(?:https://github\.com/|ssh://git@github\.com/|git@github\.com:)"
                            r"Alipsa/jmlx(?:\.git)?/?", origin, flags=re.IGNORECASE):
            raise ValueError(origin_error)
        git("cat-file", "-e", commit + "^{commit}")
        for ref in (reviewed_ref, candidate_ref):
            git("rev-parse", "--verify", ref + "^{commit}")
            try:
                git("merge-base", "--is-ancestor", commit, ref)
            except subprocess.CalledProcessError as error:
                if error.returncode == 1:
                    raise ValueError(f"linked notice commit is not an ancestor of {ref}") from error
                raise
        linked = raw_notice_sections(git("show", commit + ":jmlx-native-macos-arm64/THIRD-PARTY-NOTICES"))
    except subprocess.CalledProcessError as error:
        raise ValueError("approved notice history unavailable; fetch origin/main and "
                         "the linked commit before explicit release verification") from error
    for title in REQUIRED_NATIVE_AUTHORED_SECTIONS:
        if title not in local or title not in linked or local[title] != linked[title]:
            raise ValueError("stale approved notice permalink: raw section bytes differ or missing")
    return commit


def native_release_licenses(root, pom, jars):
    """Explicit pre-tag verification, separate from candidate-only bundle guards."""
    root = Path(root)
    if len(jars) != 3 or len({Path(jar).resolve() for jar in jars}) != 3:
        raise ValueError("exactly three distinct native jars are required")
    report = native_pom_report(Path(pom).read_bytes())
    local = (root / "jmlx-native-macos-arm64/THIRD-PARTY-NOTICES").read_bytes()
    check_authored_attribution(local, root)
    report["noticeCommit"] = verify_notice_permalink(root, local)
    inventory = read_json(root / "req/native-license-inventory.json")
    for jar in jars:
        native_notice_report(jar, inventory, root=root)
        with zipfile.ZipFile(jar) as archive:
            verify_notice_permalink(root, archive.read("META-INF/THIRD-PARTY-NOTICES"))
    report["checkedJars"] = [Path(jar).name for jar in jars]
    return report


def validate_native_legal_inventory(inventory):
    texts = inventory.get("packagedTexts")
    if not isinstance(texts, dict) or not REQUIRED_NATIVE_LEGAL_TEXTS <= texts.keys():
        raise ValueError("native license inventory lacks required packaged texts")


def native_notice_report(jar, inventory, forbidden=(), *, skip_native_payload=False, root=ROOT):
    """Check reviewed legal texts in either the runtime or documentation artifact."""
    validate_native_legal_inventory(inventory)
    with zipfile.ZipFile(jar) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate jar entries: {jar}")
        result = {}
        for name, digest in inventory["packagedTexts"].items():
            entry = "META-INF/" + name
            if entry not in names:
                raise ValueError(f"{jar}: missing native legal text: {name}")
            data = archive.read(entry)
            if not data.strip() or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"{jar}: native legal text differs from reviewed inventory: {name}")
            result[name] = digest
        check_authored_attribution(archive.read("META-INF/THIRD-PARTY-NOTICES"), root)
        for entry in archive.infolist():
            if skip_native_payload and entry.filename in {
                    "se/alipsa/jmlx/native/macos-aarch64/" + name for name in NATIVE_FILES}:
                # native_report checks these bytes separately, including the pinned MLX exception.
                continue
            if not entry.is_dir() and forbidden:
                data = archive.read(entry)
                if any(prefix and prefix.encode("utf-8") in data for prefix in forbidden):
                    raise ValueError(f"{jar}: personal path leaked in {entry.filename}")
        return result


def native_jar_report(jar, bootstrap, forbidden):
    inventory = read_json(Path(bootstrap).resolve().parents[1] / "req/native-license-inventory.json")
    native_notice_report(jar, inventory, forbidden, skip_native_payload=True,
                         root=Path(bootstrap).resolve().parents[1])
    with tempfile.TemporaryDirectory(prefix="jmlx-native-verify-") as temp:
        with zipfile.ZipFile(jar) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate jar entries")
            for name in NATIVE_FILES:
                data = archive.read("se/alipsa/jmlx/native/macos-aarch64/" + name)
                (Path(temp) / name).write_bytes(data)
            notice = archive.read("META-INF/NOTICE").decode("utf-8")
        report = native_report(temp, bootstrap, forbidden)
        if inventory.get("nativePins") != report["pins"]:
            raise ValueError("native license inventory targets different upstream pins")
        wheel_hash = re.search(r'^MLX_METAL_WHEEL_SHA256="([0-9a-f]{64})"$',
                               Path(bootstrap).read_text(), re.M)
        if not wheel_hash or inventory.get("wheelSha256") != wheel_hash[1]:
            raise ValueError("native license inventory targets a different wheel digest")
        if not all(value in notice for value in (*report["pins"].values(), wheel_hash[1])):
            raise ValueError("native NOTICE provenance differs from bootstrap pins/wheel digest")
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
            if name == "jmlx-native-macos-arm64":
                native_pom_report(archive.read(prefix + ".pom"))
                inventory = read_json(Path(root) / "req/native-license-inventory.json")
                for suffix in (".jar", "-sources.jar", "-javadoc.jar"):
                    jar_bytes = archive.read(prefix + suffix)
                    native_notice_report(io.BytesIO(jar_bytes), inventory, root=root)
                    with zipfile.ZipFile(io.BytesIO(jar_bytes)) as jar:
                        data = jar.read("META-INF/THIRD-PARTY-NOTICES")
                        verify_notice_permalink(root, data, candidate_ref=tag["sha"])
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
    native.add_argument("--documentation-jar", type=Path, action="append", default=[])
    bundle = sub.add_parser("bundle")
    bundle.add_argument("bundle", type=Path)
    bundle.add_argument("--record", type=Path, required=True)
    bundle.add_argument("--record-sha256", required=True)
    bundle.add_argument("--manifest", type=Path, required=True)
    bundle.add_argument("--root", type=Path, required=True)
    bundle.add_argument("--extract", type=Path)
    licenses = sub.add_parser("licenses")
    licenses.add_argument("--root", type=Path, required=True)
    licenses.add_argument("--pom", type=Path, required=True)
    licenses.add_argument("--jar", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.command == "select":
        for name, entry in manifest(args.manifest, args.profile).items():
            print("\t".join((name, entry["version"], entry["source"], MODULE_KEYS[name])))
    elif args.command == "preflight":
        preflight(manifest(args.manifest, args.profile), args.root)
    elif args.command == "licenses":
        print(json.dumps(native_release_licenses(args.root, args.pom, args.jar), indent=2))
    elif args.command == "bundle":
        check_bundle(args.bundle, args.record, args.record_sha256, args.manifest, args.root)
        if args.extract:
            if args.extract.exists():
                raise ValueError("bundle extraction directory must not exist")
            with zipfile.ZipFile(args.bundle) as archive:
                archive.extractall(args.extract)
    else:
        verify = native_jar_report if args.directory.suffix == ".jar" else native_report
        report = verify(args.directory, args.bootstrap, args.forbid)
        if args.documentation_jar:
            inventory = read_json(args.bootstrap.resolve().parents[1] / "req/native-license-inventory.json")
            for jar in args.documentation_jar:
                native_notice_report(jar, inventory, args.forbid, root=args.bootstrap.resolve().parents[1])
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
