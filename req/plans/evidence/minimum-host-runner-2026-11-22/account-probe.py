"""Read-only account probe for a future isolated window-time invocation.

The original November 22 measurements used original-account-probe.py.
"""

import hashlib
import json
import os
from pathlib import Path
import platform
import pwd
import shutil
import subprocess


home = Path(pwd.getpwuid(os.getuid()).pw_dir)
runner = home / "actions-runner"
clean = {
    "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "HOME": str(home),
    "USER": "jmlxrunner",
    "LOGNAME": "jmlxrunner",
}


def command(*args):
    return subprocess.check_output(
        args, text=True, stderr=subprocess.STDOUT, env=clean
    ).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


result = {
    "uid": os.getuid(),
    "user": pwd.getpwuid(os.getuid()).pw_name,
    "homeMode": oct(home.stat().st_mode & 0o777),
    "groups": command("/usr/bin/id", "-Gn").split(),
    "host": {
        "swVers": command("/usr/bin/sw_vers"),
        "architecture": platform.machine(),
        "virtualized": command("/usr/sbin/sysctl", "-n", "kern.hv_vmm_present"),
    },
    "runner": {},
    "savedEnvironment": {},
    "isolation": {},
    "tools": {},
}
config = json.loads((runner / ".runner").read_text(encoding="utf-8-sig"))
result["runner"] = {
    key: config.get(key) for key in ("agentId", "agentName", "workFolder")
}
result["runner"]["listenerVersion"] = command(str(runner / "bin/Runner.Listener"), "--version")
result["runner"]["listenerSha256"] = digest(runner / "bin/Runner.Listener")

for name in (".env", ".path"):
    path = runner / name
    data = path.read_bytes()
    result["savedEnvironment"][name] = {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "containsOwnerHome": b"/Users/per" in data,
        "mode": oct(path.stat().st_mode & 0o777),
    }

# Access checks include all parent directories. They do not isolate leaf-file permissions.
for name, path, x_access_name in (
    ("ownerGradleProperties", "/Users/per/.gradle/gradle.properties", "executable"),
    ("ownerSshDirectory", "/Users/per/.ssh", "traversable"),
    ("ownerSigningDirectory", "/Users/per/.gnupg", "traversable"),
):
    result["isolation"][name] = {
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
        x_access_name: os.access(path, os.X_OK),
    }

for name in ("python3", "git"):
    executable = shutil.which(name, path=clean["PATH"])
    path = Path(executable).resolve()
    metadata = path.stat()
    result["tools"][name] = {
        "selected": executable,
        "resolved": str(path),
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": oct(metadata.st_mode & 0o777),
        "version": command(executable, "--version"),
    }

print(json.dumps(result, indent=2))
