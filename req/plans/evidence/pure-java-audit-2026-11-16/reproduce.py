#!/usr/bin/env python3
"""Rebuild the recorded audit in a new workspace; requires the recorded JDK and Gradle."""

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


REMOVED_ENVIRONMENT = ("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS",
                       "JAVA_OPTS", "GRADLE_OPTS", "GRADLE_RO_DEP_CACHE", "CLASSPATH")
FIXED_ENVIRONMENT = {"LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC"}


def build_environment(caller, jdk):
    environment = {key: value for key, value in caller.items()
                   if key not in REMOVED_ENVIRONMENT and not key.startswith("LC_")}
    environment.update(FIXED_ENVIRONMENT, JAVA_HOME=str(jdk))
    environment["PATH"] = str(jdk / "bin") + os.pathsep + caller.get("PATH", "")
    return environment


def verify_gradle(output, expected):
    versions = re.findall(r"^Gradle (\S+)\s*$", output, re.MULTILINE)
    if versions != [expected]:
        raise ValueError(f"expected Gradle {expected}; reported versions: {versions}")


def verify_toolchains(log, report, jdk):
    def records(prefix):
        result = [json.loads(line[len(prefix):]) for line in log.splitlines()
                  if line.startswith(prefix)]
        if not result:
            raise ValueError(f"build log is missing {prefix}")
        return result

    expected_daemon = report["validation"]["review"]["freshReproduction"]["daemon"]
    # Init scripts also run in buildSrc. Require every recorded daemon to match.
    for daemon in records("AUDIT_DAEMON="):
        if (daemon.get("vendor") != expected_daemon["vendor"]
                or daemon.get("runtimeVersion") != expected_daemon["runtimeVersion"]
                or daemon.get("gradleVersion") != report["pins"]["gradle"]
                or Path(daemon.get("home", "")).resolve() != jdk):
            raise ValueError(f"unexpected Gradle daemon identity: {daemon}")
    expected_tools = report["comparisonJdk"]["compilerAndJavadoc"]
    for tools in records("AUDIT_TOOLCHAINS="):
        if tools.keys() != expected_tools.keys():
            raise ValueError("unexpected compiler/Javadoc module set")
        for module, roles in expected_tools.items():
            if tools[module].keys() != roles.keys():
                raise ValueError(f"unexpected compiler/Javadoc roles: {module}")
            for role, expected in roles.items():
                actual = tools[module][role]
                if (any(actual.get(key) != value for key, value in expected.items())
                        or Path(actual.get("home", "")).resolve() != jdk):
                    raise ValueError(f"unexpected {module} {role} identity: {actual}")


def verify_central(name, data, expected):
    if hashlib.sha256(data).hexdigest() != expected["artifacts"][name]["rawSha256"]["central"]:
        raise ValueError(f"Central SHA-256 differs from recorded audit: {name}")


def verify_result(actual, expected):
    if actual != expected:
        difference = "".join(difflib.unified_diff(
            json.dumps(expected, indent=2, sort_keys=True).splitlines(True),
            json.dumps(actual, indent=2, sort_keys=True).splitlines(True),
            fromfile="committed comparison", tofile="reproduction comparison"))
        raise ValueError(f"reproduction differs from committed comparison:\n{difference}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--jdk", required=True, type=Path)
    parser.add_argument("--gradle", required=True, type=Path)
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    repo = evidence.parents[3]
    workspace = args.workspace.resolve()
    workspace.mkdir()  # Refuse reuse of possibly stale outputs.
    command = json.loads((evidence / "build-command.json").read_text())
    report = json.loads((repo / "req/phase8-4-pure-java-audit.json").read_text())
    expected = json.loads((evidence / "comparison.json").read_text())
    jdk = args.jdk.resolve()
    environment = build_environment(os.environ, jdk)
    version = subprocess.run([str(args.gradle.resolve()), "--version", "--console=plain",
                              "--gradle-user-home", str(workspace / "gradle-home")],
                             env=environment, capture_output=True, text=True,
                             check=True, timeout=30)
    (workspace / "gradle-version.log").write_text(version.stdout + version.stderr)
    verify_gradle(version.stdout, report["pins"]["gradle"])
    subprocess.run(["git", "clone", "--no-local", str(repo), str(workspace / "checkout")],
                   check=True, timeout=120)
    subprocess.run(["git", "-C", str(workspace / "checkout"), "checkout", "--detach",
                    report["auditedSha"]], check=True, timeout=30)
    subprocess.run(["git", "-C", str(workspace / "checkout"), "apply",
                    str(evidence / "comparison-version-overrides.patch")], check=True, timeout=30)
    shutil.copyfile(evidence / "toolchains.init.gradle", workspace / "toolchains.init.gradle")

    def expand(value):
        return (value.replace("$AUDIT/toolchain/jdk-21.0.12.1.jdk", str(args.jdk.resolve()))
                .replace("$AUDIT", str(workspace))
                .replace("$GRADLE_9_9_0", str(args.gradle.resolve())))

    argv = [expand(value) for value in command["argv"]]
    argv += ["-I", str(evidence / "record-daemon.init.gradle")]
    with (workspace / "build.log").open("w") as log:
        subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
    verify_toolchains((workspace / "build.log").read_text(), report, jdk)
    for name in ("candidate", "central"):
        (workspace / name).mkdir()
    for artifact in command["publicationCopies"]:
        target = workspace / artifact["target"]
        shutil.copyfile(workspace / artifact["source"], target)
    publications = json.loads((evidence / "publications.json").read_text())
    for module, version in publications.items():
        for suffix in (".jar", "-sources.jar", "-javadoc.jar", ".pom", ".module"):
            name = f"{module}-{version}{suffix}"
            url = f"https://repo.maven.apache.org/maven2/se/alipsa/{module}/{version}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            verify_central(name, data, expected)
            (workspace / "central" / name).write_bytes(data)
    subprocess.run([sys.executable, str(repo / "tools/release-all/compare_publications.py"),
                    "--central", str(workspace / "central"),
                    "--candidate", str(workspace / "candidate"),
                    "--publications", str(evidence / "publications.json"),
                    "--output", str(workspace / "comparison.json")], check=True, timeout=30)
    diff = []
    for path in sorted((workspace / "central").iterdir()):
        if path.suffix in (".pom", ".module"):
            diff.extend(difflib.unified_diff(path.read_text().splitlines(True),
                        (workspace / "candidate" / path.name).read_text().splitlines(True),
                        fromfile=f"central/{path.name}", tofile=f"candidate/{path.name}"))
    (workspace / "metadata.diff").write_text("".join(diff))
    verify_result(json.loads((workspace / "comparison.json").read_text()), expected)
    print(f"Verified toolchains, Central hashes and every comparison record: {workspace}")


if __name__ == "__main__":
    main()
