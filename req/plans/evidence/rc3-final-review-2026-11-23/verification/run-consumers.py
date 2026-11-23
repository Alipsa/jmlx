import datetime, hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
root=Path('<CANDIDATE>')
e=root/'build/releaseAll/rc3-verification'
records=list((root/'build/releaseAll').glob('preliminary-*.json')); assert len(records)==1
preliminary=records[0]; digest=hashlib.sha256(preliminary.read_bytes()).hexdigest()
bundle=root/'build/releaseAll/jmlx-mega-bundle.zip'
record=json.loads(preliminary.read_text()); assert hashlib.sha256(bundle.read_bytes()).hexdigest()==record['bundleSha256']
with zipfile.ZipFile(bundle) as archive:
    assert archive.testzip() is None
    files=[i for i in archive.infolist() if not i.is_dir()]
    assert len(files)==len({i.filename for i in files})
    measurement={'bundleSha256':record['bundleSha256'],'measuredAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bundleBytes':bundle.stat().st_size,'withinConservativeUploadLimit':bundle.stat().st_size<=1_000_000_000,'zipFileCount':len(files),'zipEntryBytes':sum(i.file_size for i in files),'files':[{'name':i.filename,'bytes':i.file_size} for i in sorted(files,key=lambda i:i.filename)]}
    assert measurement['withinConservativeUploadLimit']
(root/'build/releaseAll/portal-bundle-measurement.json').write_text(json.dumps(measurement,indent=2)+'\n')
reports=e/'consumers'; reports.mkdir()
env=dict(os.environ); env['JAVA_HOME']='<HOME>/.sdkman/candidates/java/25.0.5.0+11-librca'
env['PATH']=env['JAVA_HOME']+'/bin:'+env['PATH']
env['GRADLE_OPTS']='-Dorg.gradle.java.installations.paths='+env['JAVA_HOME']+',/Library/Java/JavaVirtualMachines/Temurin-21.jdk/Contents/Home -Dorg.gradle.java.installations.auto-detect=false -Dorg.gradle.java.installations.auto-download=false'
for kind in ['macos-full','macos-pure']:
    argv=[sys.executable,str(root/'tools/release-all/bundle_smoke.py'),'--root',str(root),'--bundle',str(bundle),'--preliminary',str(preliminary),'--preliminary-sha256',digest,'--kind',kind,'--output',str(reports/(kind+'.json'))]
    print('START '+kind,flush=True)
    (reports/(kind+'-invocation.json')).write_text(json.dumps({'startedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'environment':{k:env[k] for k in ['JAVA_HOME','GRADLE_OPTS']}},indent=2)+'\n')
    subprocess.run(argv,cwd=root,env=env,check=True)
    print('PASS '+kind,flush=True)
transport=Path('<REPOSITORY>/build/releaseAll/candidates/rc.3/transport'); transport.mkdir(parents=True)
for path in [bundle,preliminary]:
    shutil.copyfile(path,transport/path.name)
    assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((transport/path.name).read_bytes()).digest()
container=json.loads(subprocess.check_output(['container','inspect','ubuntu'],text=True))
(reports/'linux-container-inspect.json').write_text(json.dumps(container,indent=2)+'\n')
image=container[0]['configuration']['image']['descriptor']['digest']
linuxroot='/root/release-candidates/jmlx-phase8.4-rc.3'
linuxtransport='/projects/jmlx/build/releaseAll/candidates/rc.3/transport'
argv=['bash','<HOME>/.local/bin/linux-exec','env','GRADLE_USER_HOME=/root/release-candidates/jmlx-phase8.4-rc.3-gradle-home','JMLX_LINUX_CONTAINER_IMAGE_ID='+image,'python3',linuxroot+'/tools/release-all/bundle_smoke.py','--root',linuxroot,'--bundle',linuxtransport+'/'+bundle.name,'--preliminary',linuxtransport+'/'+preliminary.name,'--preliminary-sha256',digest,'--kind','linux-pure','--output',linuxtransport+'/linux-pure.json']
(reports/'linux-pure-invocation.json').write_text(json.dumps({'startedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv},indent=2)+'\n')
print('START linux-pure',flush=True)
subprocess.run(argv,check=True)
for suffix in ['.json','.log','.runtime.xml']:
    shutil.copyfile(transport/('linux-pure'+suffix),reports/('linux-pure'+suffix))
print('PASS linux-pure',flush=True)
