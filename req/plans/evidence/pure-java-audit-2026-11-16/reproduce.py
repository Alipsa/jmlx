#!/usr/bin/env python3
"""Rebuild the recorded audit in a new workspace; requires the recorded JDK and Gradle."""

import argparse
import difflib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


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

    environment = dict(os.environ, JAVA_HOME=expand(command["environment"]["JAVA_HOME"]))
    environment["PATH"] = expand(command["environment"]["PATH_prefix"]) + os.pathsep + os.environ["PATH"]
    argv = [expand(value) for value in command["argv"]]
    argv += ["-I", str(evidence / "record-daemon.init.gradle")]
    with (workspace / "build.log").open("w") as log:
        subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=300)
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
                (workspace / "central" / name).write_bytes(response.read())
    subprocess.run(["python3", str(repo / "tools/release-all/compare_publications.py"),
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
    print(f"Comparison written to {workspace / 'comparison.json'}; inspect allEqual and artifacts.")


if __name__ == "__main__":
    main()
