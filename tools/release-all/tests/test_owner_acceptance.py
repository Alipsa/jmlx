"""Publication authorization binds a decision to immutable reviewed evidence."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import owner_acceptance as A


class OwnerAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.record = {"tag": {"name": "rc3", "sha": "a" * 40, "object": "b" * 40},
                       "bundleSha256": "c" * 64}
        self.receipt = self.root / "receipt.json"
        self.receipt.write_text('{}')
        identity = {"candidate": self.record["tag"], "bundleSha256": "c" * 64,
                    "verifiedReceiptSha256": A.digest(self.receipt.read_bytes())}
        self.packet = {"schema": "jmlx-publication-review-1", **identity,
                       "reviewDocument": self.write("review.md", "Reviewed differences and delivery"),
                       "retainedFiles": [self.write("jinja-signoff.md", "Human sign-off")]}
        packet_pin = self.write("packet.json", json.dumps(self.packet))
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@localhost",
                 "commit", "-qm", "review packet")
        commit = self.git("rev-parse", "HEAD").strip()
        self.acceptance = {"schema": "jmlx-publication-acceptance-1", **identity,
                           "reviewPacket": {**packet_pin, "commit": commit},
                           "decisions": {key: True for key in (
                               "signedCandidateAndDifferences", "deliveryTerms", "licenseFindingDisposition",
                               "jinjaHumanSignOff", "portalCapacityConfirmed", "publicationAuthorized")}}
        self.acceptance.update(reviewer="Fixture", acceptedAt="2026-11-23T12:00:00Z",
                               approvalStatement="Fixture authorization only")
        self.path = self.root / "acceptance.json"

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return {"path": name, "sha256": A.digest(path.read_bytes())}

    def verify(self):
        self.path.write_text(json.dumps(self.acceptance))
        return A.verify(self.root, self.receipt, self.record, self.path,
                        A.digest(self.path.read_bytes()))

    def test_valid_packet_is_read_at_reviewed_commit(self):
        self.assertEqual(self.verify(), self.acceptance)
        (self.root / "review.md").write_text("later living document")
        self.assertEqual(self.verify(), self.acceptance)

    def test_missing_or_changed_authorized_record(self):
        self.verify()
        for path, pin in ((None, None), (self.path, ""), (self.path, "0" * 64)):
            with self.subTest(pin=pin), self.assertRaises(ValueError):
                A.verify(self.root, self.receipt, self.record, path, pin)

    def test_every_decision_requires_literal_true(self):
        for key in self.acceptance["decisions"]:
            for value in (False, None, "true", 1):
                original = copy.deepcopy(self.acceptance)
                self.acceptance["decisions"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.verify()
                self.acceptance = original

    def test_identity_and_packet_substitution_rejected(self):
        for key in ("candidate", "bundleSha256", "verifiedReceiptSha256", "schema"):
            original = copy.deepcopy(self.acceptance)
            self.acceptance[key] = "wrong"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify()
            self.acceptance = original
        self.acceptance["reviewPacket"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest differs"):
            self.verify()

    def test_review_paths_cannot_escape_git_tree(self):
        for name in ("/etc/passwd", "../review.md", "a/../review.md", "./review.md"):
            self.acceptance["reviewPacket"]["path"] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "repository-relative"):
                self.verify()

    def test_changed_receipt_rejected(self):
        self.receipt.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "verifiedReceiptSha256"):
            self.verify()

    def test_packet_identity_and_all_evidence_pins_are_checked(self):
        original = copy.deepcopy(self.packet)
        for mutation in ("identity", "document", "retained", "empty"):
            self.packet = copy.deepcopy(original)
            if mutation == "identity":
                self.packet["bundleSha256"] = "0" * 64
            elif mutation == "document":
                self.packet["reviewDocument"]["sha256"] = "0" * 64
            elif mutation == "retained":
                self.packet["retainedFiles"][0]["sha256"] = "0" * 64
            else:
                self.packet["retainedFiles"] = []
            pin = self.write("packet.json", json.dumps(self.packet))
            self.git("add", "packet.json")
            self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@localhost",
                     "commit", "-qm", mutation)
            self.acceptance["reviewPacket"] = {**pin, "commit": self.git("rev-parse", "HEAD").strip()}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.verify()
