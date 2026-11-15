"""Release preflights reject corrupt metadata before any release mutation."""

import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("release_inputs", Path(__file__).parents[1] / "release_inputs.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class ReleaseInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "manifest.json"
        self.doc = {"schema": 1, "baselineSha": "a" * 40, "audit": "req/audit.md",
                    "modules": [{"module": name, "version": "0.1.0", "source": "local"}
                                for name in R.MODULE_KEYS]}

    def save(self, doc):
        self.path.write_text(json.dumps(doc))
        return self.path

    def test_selection_and_duplicates(self):
        self.assertEqual(set(R.manifest(self.save(self.doc), "pure-java")), R.PURE)
        self.path.write_text('{"schema":1,"schema":1}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            R.manifest(self.path)

    def test_invalid_manifest_variants(self):
        for field, value in (("version", "0.1.0-SNAPSHOT"), ("version", "0.1.0 "),
                             ("source", "mavenLocal"), ("extra", True)):
            doc = copy.deepcopy(self.doc)
            doc["modules"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                R.manifest(self.save(doc))
        for modules in (self.doc["modules"][:-1], self.doc["modules"] + [self.doc["modules"][0]]):
            with self.assertRaises(ValueError):
                R.manifest(self.save(dict(self.doc, modules=modules)))

    def test_macho_minimum_and_malformed_commands(self):
        header = struct.pack("<8I", 0xFEEDFACF, 0x100000C, 0, 6, 1, 24, 0, 0)
        command = struct.pack("<6I", 0x32, 24, 1, 0x1A0200, 0x1B0200, 0)
        self.assertEqual(R.macho_build(header + command)["minos"], (26, 2, 0))
        for data in (b"", header, header + command[:12], header + command.replace(b'\x18', b'\x04', 1)):
            with self.subTest(data=data), self.assertRaises(ValueError):
                R.macho_build(data)

    def test_preflight_does_not_probe_until_local_versions_are_valid(self):
        entries = R.manifest(self.save(self.doc))
        versions = {name: entry["version"] for name, entry in entries.items()}
        probes = []
        def absent(url):
            probes.append(url)
            return False
        versions["jmlx-core"] = "0.1.0-SNAPSHOT"
        with self.assertRaises(ValueError):
            R.preflight(entries, Path(self.temp.name), versions, absent)
        self.assertEqual(probes, [])
        versions["jmlx-core"] = "0.1.0"
        with patch.dict("os.environ", {"JMLX_CORE_VERSION": "9.0.0"}):
            with self.assertRaises(ValueError):
                R.preflight(entries, Path(self.temp.name), versions, absent)
        self.assertEqual(probes, [])
        R.preflight(entries, Path(self.temp.name), versions, absent)
        self.assertEqual(len(probes), 45)
        entries["jmlx-jinja"]["source"] = "central"
        with self.assertRaisesRegex(ValueError, "availability"):
            R.preflight(entries, Path(self.temp.name), versions, absent)

    def test_native_target_and_local_paths_are_blocking(self):
        root = Path(self.temp.name)
        bootstrap = root / "bootstrap.sh"
        bootstrap.write_text('MLX_METAL_VERSION="0.32.4"\nMLX_C_COMMIT="' + 'a'*40 +
                             '"\nMACOS_DEPLOYMENT_TARGET="26.2"\n')
        (root / "native-pin.properties").write_text('mlxMetalVersion=0.32.4\nmlxcCommit=' + 'a'*40 + '\n')
        header = struct.pack("<8I", 0xFEEDFACF, 0x100000C, 0, 6, 1, 24, 0, 0)
        command = struct.pack("<6I", 0x32, 24, 1, 0x1A0200, 0x1B0200, 0)
        for name in R.NATIVE_FILES[:3]:
            (root / name).write_bytes(header + command)
        (root / "mlx.metallib").write_bytes(b'fixture')
        self.assertEqual(R.native_report(root, bootstrap, ["/Users/release"])["minimum"], (26,2,0))
        upstream = header + command + R.UPSTREAM_PATH + b'\0'
        (root / "libmlx.dylib").write_bytes(upstream)
        with patch.object(R, "UPSTREAM_MLX_HASH", hashlib.sha256(upstream).hexdigest()):
            R.native_report(root, bootstrap, ["/Users/runner"])
            altered = upstream + b'/Users/runner/private/source.cpp\0'
            (root / "libmlx.dylib").write_bytes(altered)
            with patch.object(R, "UPSTREAM_MLX_HASH", hashlib.sha256(altered).hexdigest()):
                with self.assertRaisesRegex(ValueError, "path leaked"):
                    R.native_report(root, bootstrap, ["/Users/runner"])
        (root / "libmlx.dylib").write_bytes(header + command)
        (root / "libmlxc.dylib").write_bytes(header + command + b'/Users/release/source.cpp\0')
        with self.assertRaisesRegex(ValueError, "path leaked"):
            R.native_report(root, bootstrap, ["/Users/release"])
        newer = struct.pack("<6I", 0x32, 24, 1, 0x1B0000, 0x1B0200, 0)
        (root / "libmlxc.dylib").write_bytes(header + newer)
        with self.assertRaisesRegex(ValueError, "minos"):
            R.native_report(root, bootstrap, [])

    def test_bundle_digest_tag_and_path_checks(self):
        root = Path(self.temp.name)
        manifest_path = self.save(self.doc)
        bundle = root / "bundle.zip"
        def write_bundle(extra=None):
            with zipfile.ZipFile(bundle, "w") as archive:
                for entry in self.doc["modules"]:
                    base = f"se/alipsa/{entry['module']}/0.1.0/{entry['module']}-0.1.0"
                    for suffix in (".pom", ".module", ".jar", "-sources.jar", "-javadoc.jar"):
                        archive.writestr(base + suffix, "artifact")
                        archive.writestr(base + suffix + ".asc", "signature")
                if extra:
                    archive.writestr(extra, "untrusted")
        write_bundle()
        record_path = root / "preliminary.json"
        record = {"schema": "jmlx-preliminary-bundle-1", "bundleSha256": R.sha256(bundle),
                  "releaseManifestSha256": R.sha256(manifest_path),
                  "tag": {"name": "jmlx-phase8.4-rc.1", "object": "b"*40, "sha": "c"*40}}
        def save_record():
            record_path.write_text(json.dumps(record))
            return R.sha256(record_path)
        def git(command, **kwargs):
            if command[3] == "cat-file":
                return "tag\n"
            return ("b"*40 if command[-1] == "refs/tags/jmlx-phase8.4-rc.1" else "c"*40) + "\n"
        with patch.object(R.subprocess, "check_output", side_effect=git):
            digest = save_record()
            R.check_bundle(bundle, record_path, digest, manifest_path, root)
            with self.assertRaisesRegex(ValueError, "record digest"):
                R.check_bundle(bundle, record_path, "0"*64, manifest_path, root)
            with patch.object(R.subprocess, "check_output", return_value="wrong\n"):
                with self.assertRaisesRegex(ValueError, "tag/object/HEAD"):
                    R.check_bundle(bundle, record_path, digest, manifest_path, root)
            write_bundle("../escape")
            record["bundleSha256"] = R.sha256(bundle)
            with self.assertRaisesRegex(ValueError, "unsafe bundle path"):
                R.check_bundle(bundle, record_path, save_record(), manifest_path, root)

    def test_notice_wheel_digest_is_checked_against_bootstrap_not_only_its_inventory_hash(self):
        root = Path(self.temp.name)
        bootstrap = root / "scripts/bootstrap-native.sh"
        bootstrap.parent.mkdir()
        wheel_hash = "e"*64
        bootstrap.write_text('MLX_METAL_WHEEL_SHA256="' + wheel_hash + '"\n')
        pins = {"mlxMetalVersion": "0.32.4", "mlxcCommit": "a"*40}
        legal = {"LICENSE": b"MIT", "LICENSE-APACHE-2.0": b"Apache-2.0",
                 "NOTICE": ("0.32.4 " + "a"*40 + " " + wheel_hash).encode()}
        inventory_file = root / "req/native-license-inventory.json"
        inventory_file.parent.mkdir()
        jar = root / "native.jar"
        def write_jar():
            inventory_file.write_text(json.dumps({"nativePins": pins, "wheelSha256": wheel_hash,
                "packagedTexts": {name: hashlib.sha256(data).hexdigest() for name, data in legal.items()}}))
            with zipfile.ZipFile(jar, "w") as archive:
                for name in R.NATIVE_FILES:
                    archive.writestr("se/alipsa/jmlx/native/macos-aarch64/" + name, b"payload")
                for name, data in legal.items():
                    archive.writestr("META-INF/" + name, data)
        write_jar()
        with patch.object(R, "native_report", return_value={"pins": pins}):
            R.native_jar_report(jar, bootstrap, [])
            legal["NOTICE"] = legal["NOTICE"].replace(wheel_hash.encode(), b"f"*64)
            write_jar()  # Its new NOTICE hash matches the inventory, but provenance is stale.
            with self.assertRaisesRegex(ValueError, "NOTICE provenance"):
                R.native_jar_report(jar, bootstrap, [])


if __name__ == "__main__":
    unittest.main()
