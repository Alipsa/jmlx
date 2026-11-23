"""Validate an operator-supplied, digest-pinned final publication acceptance.

This checks the recorded decision and its evidence, not the signer's identity.
The expected digest must come from the owner's separate publication authorization.
"""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import PurePosixPath

from release_inputs import unique_object


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(data, expected, label):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{label}: SHA-256 pin missing or malformed")
    if digest(data) != expected:
        raise ValueError(f"{label}: digest differs")
    return data


def verify(root, receipt_path, record, acceptance_path, expected):
    if not acceptance_path or not expected:
        raise ValueError("upload requires owner acceptance and its authorized SHA-256")
    acceptance = json.loads(checked(acceptance_path.read_bytes(), expected, "owner acceptance"),
                            object_pairs_hook=unique_object)
    if not isinstance(acceptance, dict) or acceptance.get("schema") != "jmlx-publication-acceptance-1":
        raise ValueError("unsupported owner acceptance schema")
    for key in ("reviewer", "acceptedAt", "approvalStatement"):
        if not isinstance(acceptance.get(key), str) or not acceptance[key].strip():
            raise ValueError(f"owner acceptance missing: {key}")
    timestamp = acceptance["acceptedAt"]
    try:
        accepted_at = datetime.fromisoformat(timestamp)
    except ValueError:
        raise ValueError("owner acceptance acceptedAt must be a UTC timestamp") from None
    if ("T" not in timestamp or accepted_at.tzinfo is None or
            accepted_at.utcoffset() != timedelta(0)):
        raise ValueError("owner acceptance acceptedAt must be a UTC timestamp")
    decisions = acceptance.get("decisions", {})
    if not isinstance(decisions, dict):
        raise ValueError("owner acceptance decisions must be an object")
    for key in ("signedCandidateAndDifferences", "deliveryTerms", "licenseFindingDisposition",
                "jinjaHumanSignOff", "portalCapacityConfirmed", "publicationAuthorized"):
        if decisions.get(key) is not True:
            raise ValueError(f"owner acceptance decision required: {key}")
    identity = {"candidate": record["tag"], "bundleSha256": record["bundleSha256"],
                "verifiedReceiptSha256": digest(receipt_path.read_bytes())}
    for key, value in identity.items():
        if acceptance.get(key) != value:
            raise ValueError(f"owner acceptance identity differs: {key}")
    review = acceptance.get("reviewPacket")
    if not isinstance(review, dict):
        raise ValueError("owner acceptance reviewPacket must be an object")
    commit = review.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("review packet requires a full commit SHA")

    result = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                             record["tag"]["sha"], commit], capture_output=True, text=True)
    if result.returncode == 1:
        raise ValueError("review packet commit must descend from the candidate commit")
    if result.returncode != 0:
        raise ValueError("cannot check review packet ancestry; fetch the packet and candidate "
                         f"commits first (git: {result.stderr.strip()})")

    def blob(pin):
        if not isinstance(pin, dict):
            raise ValueError("review packet pin must be an object with path and sha256")
        name = pin.get("path")
        if (not isinstance(name, str) or not name or
                PurePosixPath(name).is_absolute() or
                any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("review packet path must be repository-relative without traversal")
        # Read immutable Git objects, never filters or working-tree files. A symlink blob
        # is just its link text; it cannot redirect this read outside the repository.
        data = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{name}"])
        return checked(data, pin.get("sha256"), name)

    packet = json.loads(blob(review), object_pairs_hook=unique_object)
    if not isinstance(packet, dict) or packet.get("schema") != "jmlx-publication-review-1":
        raise ValueError("unsupported publication review schema")
    for key, value in identity.items():
        if packet.get(key) != value:
            raise ValueError(f"review packet identity differs: {key}")
    blob(packet.get("reviewDocument"))
    files = packet.get("retainedFiles")
    if not isinstance(files, list) or not files:
        raise ValueError("review packet retained files missing")
    for pin in files:
        blob(pin)
    if not any(pin.get("sha256") == identity["verifiedReceiptSha256"] for pin in files):
        raise ValueError("review packet must retain the final verified receipt")
    return acceptance
