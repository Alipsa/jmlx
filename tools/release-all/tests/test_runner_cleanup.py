"""A failed bundle extraction must not leak the disposable repository's parent directory."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class RunnerCleanupTest(unittest.TestCase):
    def test_extraction_failure_cleans_partial_repository(self):
        with tempfile.TemporaryDirectory(prefix="jmlx-runner-cleanup-") as temp:
            root = Path(temp)
            smoke = root / "tools/release-smoke"
            smoke.mkdir(parents=True)
            shutil.copy(ROOT / "tools/release-smoke/run.sh", smoke / "run.sh")
            helper = root / "tools/release-all/release_inputs.py"
            helper.parent.mkdir()
            helper.write_text("""
import sys
from pathlib import Path
if sys.argv[1] == 'select':
    print('jmlx-core\\t0.5.0\\tlocal\\tjmlxCoreVersion')
elif '--extract' in sys.argv:
    path = Path(sys.argv[sys.argv.index('--extract') + 1])
    path.mkdir(parents=True)
    (path / 'partial').write_text('partial extraction')
    raise SystemExit(2)
""")
            wrapper = root / "gradlew"
            wrapper.write_text("#!/bin/sh\nexit 0\n")
            wrapper.chmod(0o755)
            props = root / "gradle/wrapper/gradle-wrapper.properties"
            props.parent.mkdir(parents=True)
            props.write_text("distributionUrl=https://example.invalid/gradle-9.9.0-bin.zip\n")
            gradle_home = root / "gradle-home"
            gradle = gradle_home / "wrapper/dists/gradle-9.9.0-bin/hash/gradle-9.9.0/bin/gradle"
            gradle.parent.mkdir(parents=True)
            gradle.write_text("#!/bin/sh\nexit 0\n")
            gradle.chmod(0o755)
            temporary = root / "temporary"
            temporary.mkdir()
            env = dict(os.environ, TMPDIR=str(temporary), GRADLE_USER_HOME=str(gradle_home))
            # Ensure the Python used by the copied runner is the same verified runtime as this test.
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
            result = subprocess.run(["bash", str(smoke / "run.sh"), "candidate", "--bundle", "fixture.zip",
                                     "--bundle-record", "fixture.json", "--bundle-record-sha256", "fixture-hash"],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertEqual(list(temporary.iterdir()), [], result.stdout)


if __name__ == "__main__":
    unittest.main()
