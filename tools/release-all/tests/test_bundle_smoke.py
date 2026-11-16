"""Producer acceptance rejects unsupported hosts before launching the consumer."""

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle_smoke as B


class BundleSmokeTest(unittest.TestCase):
    def setUp(self):
        self.env = {"system": "Darwin", "architecture": "arm64", "osVersion": "27.2.1",
                    "virtualized": False, "chip": "Apple fixture", "swVers": "macOS 27.2.1"}
        self.minimum = [27, 2, 0]

    def test_minimum_host_accepts_patch_releases_and_rejects_other_minor_versions(self):
        for version in ("27.2", "27.2.0", "27.2.1", "27.2.9", "26.2", "27.1", "27.3",
                        "28.0", "27", "27.2bad", "", None):
            with self.subTest(version=version):
                env = dict(self.env, osVersion=version)
                if version in ("27.2", "27.2.0", "27.2.1", "27.2.9"):
                    B.verify_smoke_environment("minimum-host", env, self.minimum, "42")
                else:
                    with self.assertRaisesRegex(ValueError, "minimum-host evidence"):
                        B.verify_smoke_environment("minimum-host", env, self.minimum, "42")

    def test_minimum_host_requires_physical_arm64_and_complete_identity(self):
        for key, value in (("system", "Linux"), ("architecture", "x86_64"),
                           ("virtualized", True), ("virtualized", None),
                           ("chip", ""), ("swVers", "")):
            with self.subTest(field=key, value=value):
                with self.assertRaisesRegex(ValueError, "minimum-host evidence"):
                    B.verify_smoke_environment("minimum-host", dict(self.env, **{key: value}),
                                               self.minimum, "42")
        with self.assertRaisesRegex(ValueError, "minimum-host evidence"):
            B.verify_smoke_environment("minimum-host", self.env, self.minimum)

    def test_other_macos_reports_require_at_least_supported_minor(self):
        for kind in ("macos-full", "macos-pure"):
            for version in ("26.2", "27.1", "27.2", "27.2.1", "27.3", "28.0", "27", None):
                with self.subTest(kind=kind, version=version):
                    env = dict(self.env, osVersion=version)
                    if version in ("27.2", "27.2.1", "27.3", "28.0"):
                        B.verify_smoke_environment(kind, env, self.minimum)
                    else:
                        with self.assertRaisesRegex(ValueError, "supported macOS"):
                            B.verify_smoke_environment(kind, env, self.minimum)

    def test_main_checks_environment_before_consumer_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = {"native": {"supportedMinimum": self.minimum}}
            output = root / "report.json"
            for kind in ("minimum-host", "macos-full", "macos-pure"):
                argv = ["bundle_smoke.py", "--root", temp, "--bundle", str(root / "bundle.zip"),
                        "--preliminary", str(root / "preliminary.json"),
                        "--preliminary-sha256", "a" * 64, "--kind", kind, "--output", str(output)]
                with self.subTest(kind=kind), patch.object(sys, "argv", argv), \
                        patch.object(B, "check_bundle", return_value=copy.deepcopy(record)), \
                        patch.object(B, "environment",
                                     return_value=dict(self.env, osVersion="27.1")), \
                        patch.dict(os.environ, {"GITHUB_RUN_ID": "42"}), \
                        patch.object(B.subprocess, "run") as consumer:
                    with self.assertRaises(ValueError):
                        B.main()
                    consumer.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_suffix(".log").exists())

    def test_main_explains_that_legacy_preliminary_records_must_be_regenerated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "report.json"
            argv = ["bundle_smoke.py", "--root", temp, "--bundle", str(root / "bundle.zip"),
                    "--preliminary", str(root / "preliminary.json"),
                    "--preliminary-sha256", "a" * 64, "--kind", "macos-full",
                    "--output", str(output)]
            for record in ({}, {"native": None}, {"native": []},
                           {"native": {"minimum": [26, 2, 0]}}):
                with self.subTest(record=record), patch.object(sys, "argv", argv), \
                        patch.object(B, "check_bundle", return_value=record), \
                        patch.object(B, "environment") as environment, \
                        patch.object(B.subprocess, "run") as consumer:
                    with self.assertRaisesRegex(ValueError, "native.supportedMinimum; regenerate"):
                        B.main()
                    environment.assert_not_called()
                    consumer.assert_not_called()
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
