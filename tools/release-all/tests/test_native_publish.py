"""Exercise the actual native build's hard guard on a generated combined-staging task."""

import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRADLE = os.environ.get("JMLX_RELEASE_TEST_GRADLE")


@unittest.skipUnless(GRADLE, "set JMLX_RELEASE_TEST_GRADLE to run Gradle integration fixtures")
class NativePublishTest(unittest.TestCase):
    def test_unstaged_local_jar_succeeds_but_combined_publish_fails(self):
        with tempfile.TemporaryDirectory(prefix="jmlx-native-publish-") as temp:
            root = Path(temp)
            module = root / "jmlx-native-macos-arm64"
            module.mkdir()
            for name in ("build.gradle", "LICENSE", "NOTICE", "LICENSE-APACHE-2.0",
                         "LICENSE-CC-BY-SA-4.0", "THIRD-PARTY-NOTICES"):
                shutil.copy(ROOT / "jmlx-native-macos-arm64" / name, module)
            (root / "settings.gradle").write_text("rootProject.name='native-publish-fixture'\ninclude 'jmlx-native-macos-arm64'\n")
            with (module / "build.gradle").open("a") as stream:
                stream.write("""
group = 'fixture'
publishing.repositories.maven {
    name = 'combinedReleaseStaging'
    url = rootProject.layout.buildDirectory.dir('combined-staging')
}
""")
            command = [GRADLE, "-p", str(root), "--no-daemon", "--no-build-cache"]
            env = dict(os.environ, GRADLE_USER_HOME=str(root / "gradle-home"))
            local = subprocess.run(command + [":jmlx-native-macos-arm64:jar"], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertEqual(local.returncode, 0, local.stdout)
            with zipfile.ZipFile(next((module / "build/libs").glob("*.jar"))) as archive:
                self.assertFalse(any(name.endswith(".dylib") for name in archive.namelist()))
            publish = subprocess.run(command + [":jmlx-native-macos-arm64:publishMavenPublicationToCombinedReleaseStagingRepository"],
                                     env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.assertNotEqual(publish.returncode, 0)
            self.assertIn("Refusing to publish jmlx-native-macos-arm64 without staged native binaries", publish.stdout)
            self.assertFalse(list((root / "build/combined-staging").rglob("*.jar")))


if __name__ == "__main__":
    unittest.main()
