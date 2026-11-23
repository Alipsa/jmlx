#!/usr/bin/env python3
"""Read-only RC2 recovery probe; print JSON for the explicitly selected checkout.

Run from the repository root:
python3 req/plans/evidence/rc3-preparation-2026-11-23/recovery-check.py --root .
Native/JUnit outputs and backup configuration may change between invocations.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def command(*argv):
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return {"argv": list(argv), "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "tools/release-all"))
    from release_inputs import native_report, sha256

    preliminary = root / "req/plans/evidence/rc2-final-review-2026-11-22/preliminary.json"
    record = json.loads(preliminary.read_text())
    forbidden = [str(Path.home()), str(root)]
    actual = json.loads(json.dumps(native_report(
        root / "native/install/lib", root / "scripts/bootstrap-native.sh", forbidden)))
    present = [name for name in record["junit"] if (root / name).is_file()]
    matching = [name for name in present if sha256(root / name) == record["junit"][name]]
    head = command("git", "-C", str(root), "rev-parse", "HEAD")
    status = command("git", "-C", str(root), "status", "--porcelain")
    if head["returncode"] or status["returncode"]:
        raise RuntimeError("Cannot record source checkout identity")
    result = {
        "schema": 1,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "sourceCommit": head["stdout"].strip(),
        "sourceCommitNote": "HEAD at check time, not necessarily a clean tree; native and "
                            "JUnit inputs are ignored outputs. See workingTreeStatus.",
        "workingTreeStatus": status["stdout"].splitlines(),
        "probeSha256": sha256(Path(__file__)),
        "nativeReportImplementationSha256": sha256(root / "tools/release-all/release_inputs.py"),
        "bootstrapSha256": sha256(root / "scripts/bootstrap-native.sh"),
        "preliminarySha256": sha256(preliminary),
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
