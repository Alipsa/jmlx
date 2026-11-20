#!/usr/bin/env python3
"""Verify native suites and bundle task execution, including native payload verification.

The default CLI checks suite results and prints verified XML digests; --tasks lists all
required bundle tasks, including the payload task that has no JUnit results.
"""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

from native_suites import required_non_test_tasks, results_dirs, suites
from release_inputs import sha256


def required_tasks():
    tests = {f":{module}:{task}" for module, names in suites.items() for name in names
             for task in results_dirs.get(module, {}).get(name, ["test"])}
    return tests | required_non_test_tasks


def verify_events(events, invocation):
    tasks = events.get("tasks", [])
    if (events.get("schema") != "jmlx-task-evidence-1" or events.get("passed") is not True
            or events.get("invocationId") != invocation
            or len(tasks) != len(required_tasks())
            or {task.get("path") for task in tasks} != required_tasks()
            or any(task.get("outcome") != "EXECUTED" for task in tasks)):
        raise ValueError("bundle task outcomes failed, missing or identify another invocation")


def verify(root):
    results = {}
    for module, names in suites.items():
        for name in names:
            for task in results_dirs.get(module, {}).get(name, ["test"]):
                directory = Path(root) / module / "build/test-results" / task
                matches = list(directory.glob(f"TEST-*.{name}.xml"))
                if len(matches) != 1:
                    raise ValueError(f"{module}/{task}/{name}: expected one XML, found {len(matches)}")
                result = ElementTree.parse(matches[0]).getroot()
                tests = int(result.attrib["tests"])
                if tests <= 0 or any(int(result.attrib.get(k, 0)) for k in ("skipped", "failures", "errors")):
                    raise ValueError(f"{module}/{task}/{name}: tests skipped, failed or empty")
                results[str(matches[0].relative_to(root))] = sha256(matches[0])
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tasks", action="store_true")
    args = parser.parse_args()
    try:
        if args.tasks:
            print(",".join(sorted(required_tasks())))
        else:
            print(json.dumps(verify(args.root), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
