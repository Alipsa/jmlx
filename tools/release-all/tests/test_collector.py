"""The producer listener distinguishes executed work from reused or skipped task results."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRADLE = os.environ.get("JMLX_RELEASE_TEST_GRADLE")


@unittest.skipUnless(GRADLE, "set JMLX_RELEASE_TEST_GRADLE to run Gradle integration fixtures")
class CollectorTest(unittest.TestCase):
    def test_task_outcomes_and_configuration_cache_reuse(self):
        with tempfile.TemporaryDirectory(prefix="jmlx-collector-") as temp:
            root = Path(temp)
            (root / "settings.gradle").write_text("rootProject.name = 'release-collector-fixture'\n")
            (root / "build.gradle").write_text("""
import org.gradle.api.tasks.*
@CacheableTask
abstract class Work extends DefaultTask {
    @OutputFile abstract RegularFileProperty getOutput()
    @TaskAction void run() { output.get().asFile.text = 'fixture' }
}
tasks.register('required', Work) { output = layout.buildDirectory.file('result') }
tasks.register('skipped') { onlyIf { false }; doLast { throw new GradleException('must skip') } }
""")
            output = root / "events.json"
            common = [GRADLE, "-p", str(root), "--configuration-cache", "--configuration-cache-problems=fail",
                      "--init-script", str(ROOT / "tools/release-all/task-evidence.init.gradle"),
                      "-PreleaseInvocationId=collector-fixture", f"-PreleaseEvidenceFile={output}"]
            env = dict(os.environ, GRADLE_USER_HOME=str(root / "gradle-home"))
            def run(task, flags, passed, outcome):
                result = subprocess.run(common + [task, "-PreleaseRequiredTasks=:" + task] + flags,
                                        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode == 0, passed, result.stdout[-6000:])
                record = json.loads(output.read_text())
                self.assertEqual(record["passed"], passed)
                self.assertEqual(record["tasks"][0]["outcome"], outcome)
                return result.stdout
            run("required", ["--rerun", "--no-build-cache"], True, "EXECUTED")
            reused = run("required", ["--rerun", "--no-build-cache"], True, "EXECUTED")
            self.assertIn("Reusing configuration cache", reused)
            run("required", ["--no-build-cache"], False, "UP-TO-DATE")
            run("skipped", ["--no-build-cache"], False, "SKIPPED")
            run("required", ["--rerun", "--build-cache"], True, "EXECUTED")
            (root / "build/result").unlink()
            run("required", ["--build-cache"], False, "FROM-CACHE")


if __name__ == "__main__":
    unittest.main()
