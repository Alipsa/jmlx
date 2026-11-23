import datetime, hashlib, json, os, plistlib, re, shutil, subprocess, sys, tempfile
from pathlib import Path
root=Path('<CANDIDATE>')
e=root/'build/releaseAll/rc3-verification/minimum-host-window'
assert os.geteuid()==0, 'Administrator authentication required'
mode=sys.argv[1]
assert mode in ['start','connected','stop']
now=datetime.datetime.now(datetime.timezone.utc)
stamp=now.strftime('%Y%m%dT%H%M%S.%fZ')
commit=subprocess.check_output(['/usr/bin/git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
assert commit=='33fd8d36aa7468a51f3e6ecc57bb581bf83731d6'
plist=Path('/Library/LaunchDaemons/com.alipsa.jmlx-release-runner.plist')
expected=root/'req/plans/evidence/minimum-host-runner-2026-11-16/com.alipsa.jmlx-release-runner.plist'
assert plistlib.loads(plist.read_bytes())==plistlib.loads(expected.read_bytes()), 'Runner service definition differs from reviewed file'
service='system/com.alipsa.jmlx-release-runner'
if mode=='stop':
    result=subprocess.run(['/bin/launchctl','bootout',service],capture_output=True,text=True)
    (e/('bootout-'+stamp+'.json')).write_text(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},indent=2)+'\n')
    if result.returncode: raise SystemExit('Runner bootout failed; inspect retained result')
sharing=subprocess.check_output(['/usr/sbin/sharing','-l'],text=True)
(e/(mode+'-sharing-'+stamp+'.txt')).write_text(sharing)
if mode=='start':
    flags=re.findall(r'(?m)^\s*shared:\s*(\d+)',sharing)
    assert flags and all(value=='0' for value in flags), 'An SMB share is enabled; review sharing before the window'
    queue=json.loads((e/'queue-review.json').read_text())
    assert not queue['matchingJobs'] and not queue['approvalRequiredRuns'], 'Unexpected release-label jobs or approval-required runs'
    age=(now-datetime.datetime.fromisoformat(queue['observedAt'])).total_seconds()
    assert 0<=age<=180, 'Refresh queue review immediately before starting the worker'
probe_source='req/plans/evidence/minimum-host-runner-2026-11-22/account-probe.py'
data=subprocess.check_output(['/usr/bin/git','-C',str(root),'show',commit+':'+probe_source])
manifest=json.loads(subprocess.check_output(['/usr/bin/git','-C',str(root),'show',commit+':req/plans/evidence/minimum-host-runner-2026-11-22/account-probe.json']))
assert hashlib.sha256(data).hexdigest()==manifest['windowTimeProbe']['scriptSha256']
stage=Path(tempfile.mkdtemp(prefix='jmlx-rc3-account-probe-',dir='/private/tmp'))
try:
    os.chown(stage,503,20); os.chmod(stage,0o700)
    probe=stage/'account-probe.py'; probe.write_bytes(data)
    os.chown(probe,503,20); os.chmod(probe,0o400)
    output=subprocess.check_output(['/usr/bin/sudo','-H','-u','jmlxrunner','/usr/bin/env','-i','PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin','/opt/homebrew/bin/python3','-I',str(probe)],cwd='/')
finally:
    shutil.rmtree(stage)
name=('pre-window' if mode=='start' else 'post-window' if mode=='stop' else 'connected')+'-account-probe-'+stamp
(e/(name+'.json')).write_bytes(output)
(e/(name+'-source-commit.txt')).write_text(commit+'\n')
(e/(name+'-method.json')).write_text(json.dumps({'startedAt':now.isoformat(),'finishedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sourceCommit':commit,'scriptSha256':hashlib.sha256(data).hexdigest(),'outputSha256':hashlib.sha256(output).hexdigest(),'method':'root-staged runner-owned 0700 temporary directory, script mode 0400; sudo -H -u jmlxrunner env -i python3 -I; cwd /'},indent=2)+'\n')
r=json.loads(output)
if mode=='start':
    baseline=json.loads((root/'req/plans/evidence/minimum-host-runner-2026-11-16/review.json').read_text())
    assert r['uid']==503 and r['user']=='jmlxrunner' and r['homeMode']=='0o700' and 'admin' not in r['groups'], 'Account isolation changed'
    assert all(not value for fields in r['isolation'].values() for value in fields.values()), 'Owner-private paths are accessible'
    assert r['savedEnvironment']['.path']['sha256']==baseline['savedPath']['sha256']
    assert r['savedEnvironment']['.env']['sha256']==baseline['savedEnvironment']['sha256']
    assert all(not value['containsOwnerHome'] for value in r['savedEnvironment'].values())
    assert r['runner']['agentId']==23 and r['runner']['listenerVersion']==baseline['runner']['listenerVersion'] and r['runner']['listenerSha256']==baseline['runner']['listenerSha256'], 'Runner identity/version changed; review it before starting'
    for key,value in r['tools'].items():
        b=baseline['tools'][key]
        assert value['selected']==b['selected']['path'] and value['resolved']==b['resolved']['path'] and value['version']==b['version'], 'Toolchain changed: '+key
    status=subprocess.run(['/bin/launchctl','print',service],capture_output=True,text=True)
    (e/('service-before-'+stamp+'.txt')).write_text(status.stdout+status.stderr)
    assert 'state = running' not in status.stdout, 'Runner already running; inspect it first'
    if status.returncode:
        subprocess.run(['/bin/launchctl','bootstrap','system',str(plist)],check=True)
    subprocess.run(['/bin/launchctl','kickstart',service],check=True)
status=subprocess.run(['/bin/launchctl','print',service],capture_output=True,text=True)
(e/(mode+'-service-'+stamp+'.json')).write_text(json.dumps({'observedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':status.returncode,'stdout':status.stdout,'stderr':status.stderr},indent=2)+'\n')
if mode=='stop': assert status.returncode!=0, 'Runner service remains loaded'
print(mode+' completed; account probe and service/sharing evidence retained',flush=True)
