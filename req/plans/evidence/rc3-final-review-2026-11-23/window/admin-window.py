# Post-window revision for review; this version was not used in the RC3 window.
import datetime, hashlib, json, os, plistlib, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
root = Path('<CANDIDATE>')
e = root / 'build/releaseAll/rc3-verification/minimum-host-window'
if os.geteuid() != 0:
    raise SystemExit('Administrator authentication required')
mode = sys.argv[1]
if mode not in ['start', 'connected', 'stop']:
    raise SystemExit("Expected window mode: ['start', 'connected', 'stop']")
now = datetime.datetime.now(datetime.timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%S.%fZ')
commit = subprocess.check_output(['/usr/bin/git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
if commit != '33fd8d36aa7468a51f3e6ecc57bb581bf83731d6':
    raise SystemExit("Candidate commit changed; restore the reviewed RC3 checkout")
plist = Path('/Library/LaunchDaemons/com.alipsa.jmlx-release-runner.plist')
expected = root / 'req/plans/evidence/minimum-host-runner-2026-11-16/com.alipsa.jmlx-release-runner.plist'
if plistlib.loads(plist.read_bytes()) != plistlib.loads(expected.read_bytes()):
    raise SystemExit('Runner service definition differs from reviewed file')
service = 'system/com.alipsa.jmlx-release-runner'
if mode == 'stop':
    result = subprocess.run(['/bin/launchctl', 'bootout', service], capture_output=True, text=True)
    (e / ('bootout-' + stamp + '.json')).write_text(json.dumps({'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit('Runner bootout failed; inspect retained result')
sharing = subprocess.check_output(['/usr/sbin/sharing', '-l'], text=True)
(e / (mode + '-sharing-' + stamp + '.txt')).write_text(sharing)
if mode == 'start':
    flags = re.findall('(?m)^\\s*shared:\\s*(\\d+)', sharing)
    if not (flags and all((value == '0' for value in flags))):
        raise SystemExit('An SMB share is enabled; review sharing before the window')
    queue = json.loads((e / 'queue-review.json').read_text())
    if queue['matchingJobs'] or queue['approvalRequiredRuns']:
        raise SystemExit('Unexpected release-label jobs or approval-required runs')
    age = (now - datetime.datetime.fromisoformat(queue['observedAt'])).total_seconds()
    if not 0 <= age <= 180:
        raise SystemExit('Refresh queue review immediately before starting the worker')
probe_source = 'req/plans/evidence/minimum-host-runner-2026-11-22/account-probe.py'
data = subprocess.check_output(['/usr/bin/git', '-C', str(root), 'show', commit + ':' + probe_source])
manifest = json.loads(subprocess.check_output(['/usr/bin/git', '-C', str(root), 'show', commit + ':req/plans/evidence/minimum-host-runner-2026-11-22/account-probe.json']))
if hashlib.sha256(data).hexdigest() != manifest['windowTimeProbe']['scriptSha256']:
    raise SystemExit("Account probe digest differs from the reviewed manifest")
stage = Path(tempfile.mkdtemp(prefix='jmlx-rc3-account-probe-', dir='/private/tmp'))
try:
    # Keep the directory root-owned until all privileged file writes are finished.
    probe = stage / 'account-probe.py'
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fchown(stream.fileno(), 503, 20)
    os.chmod(stage, 0o700)
    os.chown(stage, 503, 20, follow_symlinks=False)
    output = subprocess.check_output(['/usr/bin/sudo', '-H', '-u', 'jmlxrunner', '/usr/bin/env', '-i', 'PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', '/opt/homebrew/bin/python3', '-I', str(probe)], cwd='/')
finally:
    shutil.rmtree(stage)
name = ('pre-window' if mode == 'start' else 'post-window' if mode == 'stop' else 'connected') + '-account-probe-' + stamp
(e / (name + '.json')).write_bytes(output)
(e / (name + '-source-commit.txt')).write_text(commit + '\n')
(e / (name + '-method.json')).write_text(json.dumps({'startedAt': now.isoformat(), 'finishedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'sourceCommit': commit, 'scriptSha256': hashlib.sha256(data).hexdigest(), 'outputSha256': hashlib.sha256(output).hexdigest(), 'method': 'root-staged runner-owned 0700 temporary directory, script mode 0400; sudo -H -u jmlxrunner env -i python3 -I; cwd /'}, indent=2) + '\n')
r = json.loads(output)
if mode == 'start':
    baseline = json.loads((root / 'req/plans/evidence/minimum-host-runner-2026-11-16/review.json').read_text())
    if not (r['uid'] == 503 and r['user'] == 'jmlxrunner' and (r['homeMode'] == '0o700') and ('admin' not in r['groups'])):
        raise SystemExit('Account isolation changed')
    if not all((not value for fields in r['isolation'].values() for value in fields.values())):
        raise SystemExit('Owner-private paths are accessible')
    if r['savedEnvironment']['.path']['sha256'] != baseline['savedPath']['sha256']:
        raise SystemExit("Runner saved path differs from the reviewed baseline")
    if r['savedEnvironment']['.env']['sha256'] != baseline['savedEnvironment']['sha256']:
        raise SystemExit("Runner saved environment differs from the reviewed baseline")
    if any(value['containsOwnerHome'] for value in r['savedEnvironment'].values()):
        raise SystemExit("Runner saved environment exposes the owner home")
    if not (r['runner']['agentId'] == 23 and r['runner']['listenerVersion'] == baseline['runner']['listenerVersion'] and (r['runner']['listenerSha256'] == baseline['runner']['listenerSha256'])):
        raise SystemExit('Runner identity/version changed; review it before starting')
    for key, value in r['tools'].items():
        b = baseline['tools'][key]
        if not (value['selected'] == b['selected']['path'] and value['resolved'] == b['resolved']['path'] and (value['version'] == b['version'])):
            raise SystemExit('Toolchain changed: ' + key)
    status = subprocess.run(['/bin/launchctl', 'print', service], capture_output=True, text=True)
    (e / ('service-before-' + stamp + '.txt')).write_text(status.stdout + status.stderr)
    if 'state = running' in status.stdout:
        raise SystemExit('Runner already running; inspect it first')
    if status.returncode:
        subprocess.run(['/bin/launchctl', 'bootstrap', 'system', str(plist)], check=True)
    subprocess.run(['/bin/launchctl', 'kickstart', service], check=True)
status = subprocess.run(['/bin/launchctl', 'print', service], capture_output=True, text=True)
(e / (mode + '-service-' + stamp + '.json')).write_text(json.dumps({'observedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'returncode': status.returncode, 'stdout': status.stdout, 'stderr': status.stderr}, indent=2) + '\n')
if mode == 'stop':
    deadline = time.monotonic() + 30
    observations = []
    while status.returncode == 0 and time.monotonic() < deadline:
        time.sleep(1)
        status = subprocess.run(['/bin/launchctl', 'print', service], capture_output=True, text=True)
        observations.append({'observedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'returncode': status.returncode, 'stdout': status.stdout, 'stderr': status.stderr})
    (e / ('shutdown-confirmation-' + stamp + '.json')).write_text(json.dumps(observations, indent=2) + '\n')
    if status.returncode == 0:
        raise SystemExit('Runner service still loaded after waiting 30 seconds; inspect retained observations')
print(mode + ' completed; account probe and service/sharing evidence retained', flush=True)
