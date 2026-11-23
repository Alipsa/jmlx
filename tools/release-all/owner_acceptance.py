"""Validate an operator-supplied, digest-pinned final publication acceptance.

This checks the recorded decision and its evidence, not the signer's identity.
The expected digest must come from the owner's separate publication authorization.
"""

import hashlib
import json
import re
import subprocess
from pathlib import PurePosixPath


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
    acceptance = json.loads(checked(acceptance_path.read_bytes(), expected, "owner acceptance"))
    if not isinstance(acceptance, dict) or acceptance.get("schema") != "jmlx-publication-acceptance-1":
        raise ValueError("unsupported owner acceptance schema")
    for key in ("reviewer", "acceptedAt", "approvalStatement"):
        if not isinstance(acceptance.get(key), str) or not acceptance[key].strip():
            raise ValueError(f"owner acceptance missing: {key}")
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
    review = acceptance["reviewPacket"]
    commit = review["commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("review packet requires a full commit SHA")

    def blob(pin):
        name = pin["path"]
        if (not isinstance(name, str) or not name or
                PurePosixPath(name).is_absolute() or
                any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError("review packet path must be repository-relative without traversal")
        # Read immutable Git objects, never filters or working-tree files. A symlink blob
        # is just its link text; it cannot redirect this read outside the repository.
        data = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{name}"])
        return checked(data, pin["sha256"], name)

    packet = json.loads(blob(review))
    if not isinstance(packet, dict) or packet.get("schema") != "jmlx-publication-review-1":
        raise ValueError("unsupported publication review schema")
    for key, value in identity.items():
        if packet.get(key) != value:
            raise ValueError(f"review packet identity differs: {key}")
    blob(packet["reviewDocument"])
    files = packet["retainedFiles"]
    if not isinstance(files, list) or not files:
        raise ValueError("review packet retained files missing")
    for pin in files:
        blob(pin)
    return acceptance
