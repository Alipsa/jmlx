#!/usr/bin/env python3
"""Inspect explicit JDKs and the selected Apple developer tools without building anything."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required for release toolchain evidence")


def command_output(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def java_runtime(java_home):
    text = command_output(str(Path(java_home) / "bin/java"), "-XshowSettings:properties", "-version")
    values = dict(re.findall(r"^\s*(java\.[\w.]+) = (.*)$", text, re.M))
    names = ("java.version", "java.runtime.version", "java.vendor", "java.home", "java.vm.name")
    if not all(values.get(name) for name in names) or not values.get("java.specification.version", "").isdigit():
        raise ValueError("explicit JDK did not report complete runtime properties")
    return {**{name: values[name] for name in names}, "java.feature": values["java.specification.version"]}


def native_toolchain():
    result = {
        "cc": command_output("cc", "--version"),
        "cmake": command_output("cmake", "--version"),
        "sdk": command_output("xcrun", "--show-sdk-version"),
        "developerDirectory": command_output("xcode-select", "-p"),
    }
    if result["developerDirectory"].endswith("/CommandLineTools"):
        result["commandLineTools"] = command_output("pkgutil", "--pkg-info=com.apple.pkg.CLTools_Executables")
    else:
        try:
            result["xcode"] = command_output("xcodebuild", "-version")
        except (subprocess.CalledProcessError, FileNotFoundError):
            result["commandLineTools"] = command_output("pkgutil", "--pkg-info=com.apple.pkg.CLTools_Executables")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("native", "java"))
    parser.add_argument("--java-home", type=Path)
    args = parser.parse_args()
    try:
        if args.kind == "java" and not args.java_home:
            raise ValueError("--java-home is required")
        print(json.dumps(native_toolchain() if args.kind == "native" else java_runtime(args.java_home), indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")
