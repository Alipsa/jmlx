"""Exercise jmlx's upload wiring with the published plugin and a TLS-only Portal fixture.

The fixture substitutes synthetic consumer evidence for physical/native verification;
FinalReceiptTest exercises the real receipt validator separately. No Central endpoint or
real signing/Portal credential participates in this test.
"""

import email.parser
import email.policy
import hashlib
import http.server
import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRADLE = os.environ.get("JMLX_RELEASE_TEST_GRADLE")


@unittest.skipUnless(GRADLE, "set JMLX_RELEASE_TEST_GRADLE to run the published-plugin TLS fixture")
class PluginUploadTest(unittest.TestCase):
    def test_reviewed_hashes_and_receipt_gate_protect_the_actual_upload(self):
        with tempfile.TemporaryDirectory(prefix="jmlx-plugin-upload-") as temp:
            root = Path(temp)
            uploads = []
            requests = []

            class Portal(http.server.BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_HEAD(self):
                    requests.append(self.path)
                    self.send_response(200 if uploads else 404)
                    self.end_headers()

                def do_GET(self):
                    requests.append(self.path)
                    body = ({"deploymentState": "PUBLISHED"} if "/publisher/status" in self.path
                            else {"published": False})
                    self.send_response(200 if "/repo/" not in self.path or uploads else 404)
                    self.end_headers()
                    self.wfile.write(json.dumps(body).encode())

                def do_POST(self):
                    requests.append(self.path)
                    if "/publisher/status" in self.path:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'{"deploymentState":"PUBLISHED"}')
                        return
                    self.assert_upload_path()
                    body = self.rfile.read(int(self.headers["Content-Length"]))
                    message = email.parser.BytesParser(policy=email.policy.default).parsebytes(
                        ("Content-Type: " + self.headers["Content-Type"] + "\r\n\r\n").encode() + body)
                    uploads.append(next(part.get_payload(decode=True) for part in message.iter_parts()
                                        if part.get_filename() == "jmlx-mega-bundle.zip"))
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"fixture-deployment")

                def assert_upload_path(self):
                    if "/publisher/upload" not in self.path:
                        raise AssertionError("unexpected POST: " + self.path)

            certificate = root / "localhost.pem"
            key = root / "localhost.key"
            truststore = root / "truststore.p12"
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                            "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
                            "-keyout", str(key), "-out", str(certificate)], check=True, capture_output=True)
            subprocess.run(["keytool", "-importcert", "-noprompt", "-alias", "fixture", "-file", str(certificate),
                            "-keystore", str(truststore), "-storepass", "fixture-password"],
                           check=True, capture_output=True)
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Portal)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certificate, key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                project = root / "project"
                project.mkdir()
                (project / ".gitignore").write_text("build/\n.gradle/\n")
                (project / "marker.txt").write_text("reviewed")
                version = re.search(r'nexus-release-plugin", version = "([^"]+)"',
                                    (ROOT / "gradle/libs.versions.toml").read_text()).group(1)
                text = (ROOT / "build.gradle").read_text()
                start = text.index("if (providers.gradleProperty('releaseAllMode').isPresent()) {")
                end = text.index("\ndef projectJavaVersion", start)
                block = text[start:end]
                modules = ["jmlx-core", "jmlx-ffi", "jmlx-jinja", "jmlx-models", "jmlx-native-macos-arm64",
                           "jmlx-tokenizer", "jmlx-vision", "jmlx-spring-ai", "jmlx-langchain4j"]
                (project / "settings.gradle").write_text("rootProject.name='jmlx'\ninclude " +
                                                        ", ".join(repr(name) for name in modules) + "\n")
                (project / "build.gradle").write_text(
                    "plugins { id 'se.alipsa.nexus-release-plugin' version '" + version + "' apply false }\n" + block)
                for name in modules:
                    module = project / name
                    module.mkdir()
                    (module / "build.gradle").write_text("""
plugins { id 'java-library'; id 'maven-publish' }
group = 'se.alipsa'; version = '1.0'
java { withSourcesJar(); withJavadocJar() }
publishing { publications { maven(MavenPublication) { from components.java } } }
""")
                (project / "req").mkdir()
                shutil.copy(ROOT / "req/releaseAll-portal-fixture.json", project / "req")
                shutil.copy(ROOT / "releaseAll.sh", project / "releaseAll.sh")
                (project / "gradlew").write_text("""#!/usr/bin/env python3
import json, sys
from pathlib import Path
p = Path(__file__).resolve().parent / 'build/wrapper-argv.json'
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(sys.argv[1:]))
""")
                (project / "gradlew").chmod(0o755)
                verifier = project / "tools/release-all/final_receipt.py"
                verifier.parent.mkdir(parents=True)
                # Only preflight/runtime evidence is synthetic; test the actual shell argv bridge.
                (verifier.parent / "preflight.sh").write_text(
                    "release_tools_preflight() { RELEASE_JAVA_HOME=/fixture/jdk25; "
                    "RELEASE_JAVA21_HOME=/fixture/jdk21; }\n")
                verifier.write_text('''import argparse, hashlib, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--root'); p.add_argument('--verify'); p.add_argument('--plugin-digests', action='store_true')
a = p.parse_args()
r = json.loads(Path(a.verify).read_text())
if not r['testEvidencePassed']: raise SystemExit('fixture required test evidence rejected')
for name, expected in r['files'].items():
    if hashlib.sha256((Path(a.root) / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('fixture final receipt mismatch')
if a.plugin_digests:
    print(r['files']['build/releaseAll/jmlx-mega-bundle.zip'] + '\t' + r['files']['build/releaseAll/manifest.json'])
''')
                def git(*args):
                    return subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
                git("init")
                git("add", ".")
                git("-c", "user.name=Fixture", "-c", "user.email=fixture@localhost", "commit", "-m", "fixture")
                home = root / "gradle-home"
                home.mkdir()
                # Share only downloaded artifacts, never user credentials or init scripts.
                for name in ("caches", "jdks"):
                    (home / name).symlink_to(Path(os.environ.get("GRADLE_USER_HOME", Path.home() / ".gradle")) / name,
                                           target_is_directory=True)
                env = dict(os.environ, GRADLE_USER_HOME=str(home))
                url = "https://localhost:" + str(server.server_port)
                command = [GRADLE, "-p", str(project), "--no-daemon", "--no-configuration-cache",
                           "-PreleaseAllMode=true", "-PsonatypeUsername=fixture", "-PsonatypePassword=fixture",
                           "-PreleaseAllRepositoryUrl=" + url + "/repo", "-PreleaseAllPortalUrl=" + url + "/api/v1",
                           "-Djavax.net.ssl.trustStore=" + str(truststore),
                           "-Djavax.net.ssl.trustStorePassword=fixture-password"]
                def run(*args, success=True):
                    result = subprocess.run(command + list(args), env=env, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
                    self.assertEqual(result.returncode == 0, success, result.stdout[-7000:])
                    self.assertNotIn("> Task :bundleAll", result.stdout)
                    return result.stdout
                run("releaseAllPlan")
                build = project / "build/releaseAll"
                bundle = build / "jmlx-mega-bundle.zip"
                with zipfile.ZipFile(bundle, "w") as archive:
                    archive.writestr("verified-fixture.txt", b"reviewed signed payload fixture")
                original = bundle.read_bytes()
                digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                plan = build / "manifest.json"
                bundle_hash, plan_hash = digest(bundle), digest(plan)
                self.assertEqual(sorted(entry["artifactId"] for entry in json.loads(plan.read_text())["selected"]),
                                 sorted(modules))
                receipt = build / "bundle-receipt.json"
                receipt.write_text(json.dumps({"bundleHash": bundle_hash, "manifestHash": plan_hash}))
                final = build / "verified fixture.json"
                evidence = {"testEvidencePassed": True, "files": {
                    "build/releaseAll/jmlx-mega-bundle.zip": bundle_hash,
                    "build/releaseAll/manifest.json": plan_hash}}
                final.write_text(json.dumps(evidence))
                shell = subprocess.run(["bash", str(project / "releaseAll.sh"), "--publish-verified", str(final)],
                                       env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(shell.returncode, 0, shell.stdout)
                wrapper_args = json.loads((project / "build/wrapper-argv.json").read_text())
                self.assertIn("-PreleaseAllExpectedBundleSha256=" + bundle_hash, wrapper_args)
                self.assertIn("-PreleaseAllExpectedManifestSha256=" + plan_hash, wrapper_args)
                self.assertIn("-PreleaseAllVerifiedReceipt=" + str(final), wrapper_args)
                upload = ["releaseAll", "-PreleaseAllUseExistingBundle=true", "-PreleaseAllVerifiedReceipt=" + str(final),
                          "-PreleaseAllExpectedBundleSha256=" + bundle_hash,
                          "-PreleaseAllExpectedManifestSha256=" + plan_hash]
                # The jmlx graph gate must reject ordinary rebuilding uploads before any tasks.
                self.assertIn("Use releaseAll.sh --publish-verified", run("releaseAll", success=False))
                self.assertFalse(uploads)
                evidence["testEvidencePassed"] = False
                final.write_text(json.dumps(evidence))
                (project / "build/wrapper-argv.json").unlink()
                rejected_shell = subprocess.run(["bash", str(project / "releaseAll.sh"), "--publish-verified", str(final)],
                                                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertNotEqual(rejected_shell.returncode, 0)
                self.assertFalse((project / "build/wrapper-argv.json").exists())
                self.assertIn("fixture required test evidence rejected", run(*upload, success=False))
                self.assertFalse(uploads)
                evidence["testEvidencePassed"] = True
                final.write_text(json.dumps(evidence))
                (project / "marker.txt").write_text("drift")
                self.assertIn("Stale manifest", run(*upload, success=False))
                self.assertFalse(uploads)
                (project / "marker.txt").write_text("reviewed")
                bundle.write_bytes(original + b"tampered")
                self.assertIn("fixture final receipt mismatch", run(*upload, success=False))
                self.assertFalse(uploads)
                bundle.write_bytes(original)
                plan_bytes = plan.read_bytes()
                plan.write_bytes(plan_bytes + b"\n")
                self.assertIn("expectedManifestSha256", run(*upload, success=False))
                self.assertFalse(uploads)
                plan.write_bytes(plan_bytes)
                run(*upload)
                self.assertEqual(uploads, [original])
                self.assertEqual(bundle.read_bytes(), original)
                self.assertEqual(json.loads((build / "deployment.json").read_text())["bundleHash"], bundle_hash)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
