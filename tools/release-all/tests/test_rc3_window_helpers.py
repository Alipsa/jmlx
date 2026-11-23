"""Inspect the review-only helper without executing its administrative entry point."""

import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "req/plans/evidence/rc3-final-review-2026-11-23/window/admin-window.py"


class Rc3WindowHelperTest(unittest.TestCase):
    def test_revised_checks_do_not_use_assert(self):
        tree = ast.parse(HELPER.read_text())
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))

    def staging_statements(self):
        tree = ast.parse(HELPER.read_text())
        # Execute only the staging fragment. Never run the administrative entry point,
        # subprocess calls, or cleanup against a directory owned by another user.
        block = next(node for node in tree.body if isinstance(node, ast.Try))
        statements = []
        for node in block.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "output"
                    for target in node.targets):
                break
            statements.append(node)
        return compile(ast.Module(body=statements, type_ignores=[]), str(HELPER), "exec")

    def test_probe_is_written_before_directory_ownership_transfer(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            probe = stage / "account-probe.py"
            data = b"fixture probe\n"
            events = []

            def open_probe(path, flags, mode):
                self.assertEqual(path, probe)
                self.assertEqual(flags, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW)
                self.assertEqual(mode, 0o400)
                events.append("open")
                return os.open(path, flags, mode)

            def own_file(fd, uid, gid):
                self.assertEqual((uid, gid), (503, 20))
                self.assertEqual(os.fstat(fd).st_size, len(data))
                self.assertEqual(probe.read_bytes(), data)
                events.append("file ownership")

            def own_directory(path, uid, gid, *, follow_symlinks):
                self.assertEqual((path, uid, gid, follow_symlinks), (stage, 503, 20, False))
                self.assertEqual(probe.read_bytes(), data)
                self.assertEqual(probe.stat().st_mode & 0o777, 0o400)
                events.append("directory ownership")

            fake_os = SimpleNamespace(open=open_probe, fdopen=os.fdopen, fchown=own_file,
                                      chmod=os.chmod, chown=own_directory,
                                      O_WRONLY=os.O_WRONLY, O_CREAT=os.O_CREAT,
                                      O_EXCL=os.O_EXCL, O_NOFOLLOW=os.O_NOFOLLOW)
            exec(self.staging_statements(), {"stage": stage, "data": data, "os": fake_os})
            self.assertEqual(events, ["open", "file ownership", "directory ownership"])

    def test_preexisting_symlink_is_not_followed(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            target = stage / "target"
            target.write_bytes(b"unchanged")
            (stage / "account-probe.py").symlink_to(target)
            fake_os = SimpleNamespace(open=os.open, fdopen=os.fdopen, fchown=Mock(),
                                      chmod=Mock(), chown=Mock(),
                                      O_WRONLY=os.O_WRONLY, O_CREAT=os.O_CREAT,
                                      O_EXCL=os.O_EXCL, O_NOFOLLOW=os.O_NOFOLLOW)
            with self.assertRaises(FileExistsError):
                exec(self.staging_statements(), {"stage": stage, "data": b"replacement",
                                                 "os": fake_os})
            self.assertEqual(target.read_bytes(), b"unchanged")
            fake_os.fchown.assert_not_called()
            fake_os.chown.assert_not_called()


if __name__ == "__main__":
    unittest.main()
