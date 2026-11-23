import datetime, hashlib, json, os, shutil, subprocess
from pathlib import Path
root=Path('<CANDIDATE>')
evidence=root/'build/releaseAll/rc3-verification'
env=dict(os.environ)
env.update(JAVA_HOME='<HOME>/.sdkman/candidates/java/25.0.5.0+11-librca',JMLX_RELEASE_JAVA_HOME='<HOME>/.sdkman/candidates/java/25.0.5.0+11-librca',JMLX_RELEASE_JAVA21_HOME='/Library/Java/JavaVirtualMachines/Temurin-21.jdk/Contents/Home',JMLX_RELEASE_CANDIDATE_TAG='jmlx-phase8.4-rc.3')
env['PATH']='<HOME>/.nvm/versions/node/v26.7.0/bin:'+env['JAVA_HOME']+'/bin:'+env['PATH']
steps=[('buildsrc-check',['./gradlew','-p','buildSrc','check']),('release-selftest',['bash','tools/release-all/selftest.sh']),('jinja-pass-1',['./gradlew','-I',str(evidence/'toolchain-observation.init.gradle'),':jmlx-jinja:releaseVerification']),('jinja-pass-2',['./gradlew','-I',str(evidence/'toolchain-observation.init.gradle'),':jmlx-jinja:releaseVerification']),('dry-run',['./releaseAll.sh','--dry-run']),('bundle-only',['./releaseAll.sh','--bundle-only'])]
for name,argv in steps[2:]:
    record={'sourceCommit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'workingTreeStatus':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),'startedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':argv,'environment':{k:env[k] for k in ['JAVA_HOME','JMLX_RELEASE_JAVA_HOME','JMLX_RELEASE_JAVA21_HOME','JMLX_RELEASE_CANDIDATE_TAG','PATH']}}
    path=evidence/(name+'-invocation.json')
    assert not path.exists()
    path.write_text(json.dumps(record,indent=2)+'\n')
    print('START '+name,flush=True)
    with (evidence/(name+'.log')).open('xb') as log:
        result=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(finishedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=result.returncode)
    path.write_text(json.dumps(record,indent=2)+'\n')
    if result.returncode: raise SystemExit(name+' failed; inspect retained log')
    if name.startswith('jinja-pass'):
        shutil.copytree(root/'jmlx-jinja/build/reports',evidence/name/'reports')
    print('PASS '+name,flush=True)
