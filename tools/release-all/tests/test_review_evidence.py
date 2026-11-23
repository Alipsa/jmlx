"""Exercise the actual evidence guard in a small, disposable Gradle project."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
GRADLE = os.environ.get("JMLX_RELEASE_TEST_GRADLE")


@unittest.skipUnless(GRADLE, "set JMLX_RELEASE_TEST_GRADLE to run Gradle integration fixtures")
class ReviewEvidenceTest(unittest.TestCase):
    def test_repository_relative_packet_and_checklist_reference(self):
        with tempfile.TemporaryDirectory(prefix="jmlx-review-evidence-") as temp:
            root = Path(temp)
            packet = root / "evidence/packet"
            packet.mkdir(parents=True)
            (root / "settings.gradle").write_text("rootProject.name = 'review-fixture'\n")
            shutil.copyfile(ROOT / "tools/release-all/review-evidence.gradle",
                            root / "review-evidence.gradle")
            (root / "build.gradle").write_text("""
apply plugin: 'base'
apply from: 'review-evidence.gradle'
registerReviewEvidenceCheck('verifyPacket', 'evidence/packet', 'Fixture', true,
        'publication-review.json', true, 'checklist.md')
check.dependsOn tasks.named('verifyPacket')
""")
            payload = packet / "nested/payload.json"
            payload.parent.mkdir()
            payload.write_bytes(b'{"fixture":true}\n')
            document = root / "review.md"
            document.write_text("Fixture review\n")

            def pin(path):
                data = path.read_bytes()
                return {"path": path.relative_to(root).as_posix(), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()}

            original = {"reviewDocument": pin(document), "retainedFiles": [pin(payload)]}
            manifest = packet / "publication-review.json"
            checklist = root / "checklist.md"
            env = dict(os.environ, GRADLE_USER_HOME=str(root / "gradle-home"))

            def prepare(record=None, refresh_reference=True):
                payload.unlink(missing_ok=True)
                payload.write_bytes(b'{"fixture":true}\n')
                manifest.write_text(json.dumps(record or original, indent=2) + "\n")
                if refresh_reference:
                    checklist.write_text("publication manifest\nSHA-256 is `" +
                                         hashlib.sha256(manifest.read_bytes()).hexdigest() + "`.\n")

            def run(passed, diagnostic=None):
                result = subprocess.run([GRADLE, "-p", str(root), "--console=plain", "check"],
                                        env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=90)
                self.assertEqual(result.returncode == 0, passed, result.stdout[-5000:])
                if diagnostic:
                    self.assertIn(diagnostic, result.stdout)

            prepare()
            run(True)
            finder = payload.parent / ".DS_Store"
            finder.write_bytes(b"Finder")
            run(True)
            finder.unlink()
            extra = payload.parent / "unlisted.json"
            extra.write_text("{}")
            run(False, "inventory differs")
            extra.unlink()
            payload.write_bytes(b"changed")
            run(False, "digest/size differs")
            prepare()
            payload.unlink()
            run(False, "missing or outside its root")
            for label, mutate, diagnostic in [
                ("missing size", lambda r: r["retainedFiles"][0].pop("bytes"),
                 "nonnegative integer size"),
                ("wrong size", lambda r: r["retainedFiles"][0].update(bytes=1),
                 "digest/size differs"),
                ("outside prefix", lambda r: r["retainedFiles"][0].update(path="review.md"),
                 "path must be inside"),
                ("absolute", lambda r: r["retainedFiles"][0].update(path=str(payload)),
                 "path must be inside"),
                ("traversal", lambda r: r["retainedFiles"][0].update(
                    path="evidence/packet/../outside.json"), "relative file path"),
                ("duplicate", lambda r: r["retainedFiles"].append(copy.deepcopy(
                    r["retainedFiles"][0])), "duplicate or reserved paths"),
                ("missing document", lambda r: r.pop("reviewDocument"),
                 "requires its pinned reviewDocument"),
            ]:
                with self.subTest(label=label):
                    record = copy.deepcopy(original)
                    mutate(record)
                    prepare(record)
                    run(False, diagnostic)
            prepare()
            payload.unlink()
            payload.symlink_to(document)
            run(False, "missing or outside its root")
            prepare()
            manifest.write_text(json.dumps(original, separators=(",", ":")) + "\n")
            run(False, "manifest digest differs from the checklist reference")


if __name__ == "__main__":
    unittest.main()
