"""Toolchain measurement works with CLT-only hosts and rejects incomplete JVM identities."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_evidence as E


class RuntimeEvidenceTest(unittest.TestCase):
    def test_clt_only_does_not_require_xcodebuild(self):
        def output(*args):
            if args == ("xcode-select", "-p"):
                return "/Library/Developer/CommandLineTools"
            if args[0] == "xcodebuild":
                self.fail("CLT-only hosts must not invoke xcodebuild")
            return "fixture " + " ".join(args)
        with patch.object(E, "command_output", side_effect=output):
            toolchain = E.native_toolchain()
        self.assertIn("commandLineTools", toolchain)
        self.assertNotIn("xcode", toolchain)

    def test_xcodebuild_failure_falls_back_to_clt_metadata(self):
        def output(*args):
            if args[0] == "xcodebuild":
                raise subprocess.CalledProcessError(1, args)
            return "fixture " + " ".join(args)
        with patch.object(E, "command_output", side_effect=output):
            self.assertIn("commandLineTools", E.native_toolchain())

    def test_explicit_java_home_returns_measured_vendor_build_and_path(self):
        text = """Property settings:
    java.home = /fixture/jdk-25
    java.version = 25.0.1
    java.runtime.version = 25.0.1+7
    java.specification.version = 25
    java.vendor = Example VM vendor
    java.vm.name = Example VM
"""
        with patch.object(E, "command_output", return_value=text) as command:
            runtime = E.java_runtime(Path("/fixture/jdk-25"))
        self.assertEqual(runtime["java.vendor"], "Example VM vendor")
        self.assertEqual(runtime["java.runtime.version"], "25.0.1+7")
        self.assertEqual(command.call_args.args[0], "/fixture/jdk-25/bin/java")
        with patch.object(E, "command_output", return_value="java.version = 25"):
            with self.assertRaisesRegex(ValueError, "complete runtime"):
                E.java_runtime(Path("/fixture/jdk-25"))


if __name__ == "__main__":
    unittest.main()
