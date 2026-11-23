#!/usr/bin/env python3
"""Read-only RC2 recovery probe; print JSON for the explicitly selected checkout.

Run from the repository root:
python3 req/plans/evidence/rc3-preparation-2026-11-23/recovery-check.py --root .
Native/JUnit outputs and backup configuration may change between invocations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import types


EXPECTED_RELEASE_INPUTS = "85b4f940f67faf916a9263a3920a69763837176af6685cb8c8db131e4ca4e849"
EXPECTED_PRELIMINARY = "3a530d8386834abb20de23b96d4ec0a46e6cf62207952b193c2fc9d6e29689af"


def command(*argv):
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {"argv": list(argv), "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    preliminary = root / "req/plans/evidence/rc2-final-review-2026-11-22/preliminary.json"
    if not preliminary.is_file():
        parser.error(f"--root must contain the retained RC2 preliminary record: {preliminary}")
    try:
        data = preliminary.read_bytes()
    except OSError as error:
        parser.error(f"cannot read preliminary.json: {error}")
    preliminary_digest = hashlib.sha256(data).hexdigest()
    if preliminary_digest != EXPECTED_PRELIMINARY:
        parser.error("preliminary.json differs from the reviewed RC2 record")
    record = json.loads(data)
    for name in record["junit"]:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            parser.error(f"JUnit path must stay relative to --root: {name}")

    # Use the probe repository's reviewed implementation, never code from --root.
    tools = Path(__file__).resolve().parents[4] / "tools/release-all"
    implementation = tools / "release_inputs.py"
    try:
        source = implementation.read_bytes()
    except OSError as error:
        parser.error(f"cannot read the probe's release_inputs.py: {error}")
    implementation_digest = hashlib.sha256(source).hexdigest()
    if implementation_digest != EXPECTED_RELEASE_INPUTS:
        parser.error("probe release_inputs.py differs from the reviewed SHA-256")
    # Execute the checked bytes directly to avoid importing stale bytecode or rereading.
    helper = types.ModuleType("rc3_reviewed_release_inputs")
    helper.__file__ = str(implementation)
    exec(compile(source, str(implementation), "exec"), helper.__dict__)
    native_report, sha256 = helper.native_report, helper.sha256

    forbidden = [str(Path.home()), str(root)]
    try:
        actual = json.loads(json.dumps(native_report(
            root / "native/install/lib", root / "scripts/bootstrap-native.sh", forbidden)))
    except (OSError, ValueError) as error:
        parser.error(f"cannot inspect native inputs under --root: {error}")
    present = [name for name in record["junit"] if (root / name).is_file()]
    matching = [name for name in present if sha256(root / name) == record["junit"][name]]
    head = command("git", "-C", str(root), "rev-parse", "HEAD")
    status = command("git", "--no-optional-locks", "-C", str(root),
                     "-c", "core.fsmonitor=false", "status", "--porcelain")
    if head["returncode"] or status["returncode"]:
        parser.error("cannot record source checkout identity with git")
    result = {
        "schema": 1,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "sourceCommit": head["stdout"].strip(),
        "sourceCommitNote": "HEAD at check time, not necessarily a clean tree; native and "
                            "JUnit inputs are ignored outputs. See workingTreeStatus.",
        "workingTreeStatus": status["stdout"].splitlines(),
        "gitStatusArgv": status["argv"],
        "probeSha256": sha256(Path(__file__)),
        "nativeReportImplementationSha256": implementation_digest,
        "nativeReportImplementationSource": "probe repository, SHA-256 checked before execution",
        "bootstrapSha256": sha256(root / "scripts/bootstrap-native.sh"),
        "preliminarySha256": preliminary_digest,
        "nativeReportArguments": {
            "directory": str(root / "native/install/lib"),
            "bootstrap": str(root / "scripts/bootstrap-native.sh"),
            "forbidden": forbidden,
        },
        "nativeReportMatchesRc2": actual == record["native"],
        "nativeReport": actual,
        "junit": {
            "required": len(record["junit"]),
            "presentInMainCheckout": len(present),
            "matchingInMainCheckout": len(matching),
            "method": "sha256 of <root>/<relative path> from preliminary.json junit map",
        },
        "backupRecovery": {
            "timeMachineDestinations": command("tmutil", "destinationinfo"),
            "localSnapshots": command("tmutil", "listlocalsnapshots", "/"),
            "apfsSnapshots": command("diskutil", "apfs", "listSnapshots", "/"),
            "mountedVolumes": sorted(path.name for path in Path("/Volumes").iterdir()),
            "limit": "Local observations only; nonzero command results do not establish "
                     "absence. External/offline backups were not searched. An OS update "
                     "snapshot alone does not establish recovery of candidate data.",
        },
        "conclusion": "This probe reports native equivalence and JUnit matches only. It "
                      "does not recover missing producer inputs or restore RC2 revalidation.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
