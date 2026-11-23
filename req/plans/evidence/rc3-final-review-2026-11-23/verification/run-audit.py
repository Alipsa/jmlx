import datetime, hashlib, json, os, shutil, subprocess, urllib.request
from pathlib import Path
root=Path('<CANDIDATE>')
audit=Path('<AUDIT>')
evidence=root/'build/releaseAll/rc3-verification/audit'
evidence.mkdir()
source=root/'req/plans/evidence/pure-java-audit-2026-11-16'
jdk_source=Path('<REPOSITORY>/build/releaseAll/pure-java-audit-2026-11-16/toolchain/jdk-21.0.12.1.jdk')
jdk=evidence/'toolchain/jdk-21.0.12.1.jdk'
shutil.copytree(jdk_source,jdk,symlinks=True)
assert (jdk/'bin/java').is_file()
versions={'jmlx-jinja':('0.6.2','0.6.1'),'jmlx-tokenizer':('0.1.2','0.1.1'),'jmlx-vision':('0.1.2','0.1.1')}
for module,(old,new) in versions.items():
    p=audit/module/'build.gradle'
    text=p.read_text()
    original="version = '"+old+"'"
    assert text.count(original)==1, module
    p.write_text(text.replace(original,"version = '"+new+"'"))
(evidence/'comparison-version-overrides.patch').write_bytes(subprocess.check_output(['git','-C',str(audit),'diff']))
init=evidence/'toolchains.init.gradle'
init.write_text((source/'toolchains.init.gradle').read_text()+"\nprintln('AUDIT_DAEMON=' + groovy.json.JsonOutput.toJson([home: System.getProperty('java.home'), vendor: System.getProperty('java.vendor'), runtimeVersion: System.getProperty('java.runtime.version'), gradleVersion: gradle.gradleVersion]))\n")
gradle=next(Path('<HOME>/.gradle/wrapper/dists/gradle-9.9.0-bin').glob('*/gradle-9.9.0/bin/gradle'))
template=json.loads((source/'build-command.json').read_text())
replacements={'$GRADLE_9_9_0':str(gradle),'$AUDIT/checkout':str(audit),'$AUDIT':str(evidence)}
argv=[]
for value in template['argv']:
    for key,replacement in replacements.items(): value=value.replace(key,replacement)
    argv.append(value)
env={key:os.environ[key] for key in ['HOME','PATH','TMPDIR','USER'] if key in os.environ}
env.update(LC_ALL='C',LANG='C',LANGUAGE='C',TZ='UTC',JAVA_HOME=str(jdk))
env['PATH']=str(jdk/'bin')+os.pathsep+env.get('PATH','')
record={'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tag':'jmlx-phase8.4-rc.3','sourceCommit':subprocess.check_output(['git','-C',str(audit),'rev-parse','HEAD'],text=True).strip(),'overrides':versions,'argv':argv,'environment':env,'jdkProvenance':'Existing installation copied to durable storage; identity verified from build log, archive origin not re-established'}
(evidence/'invocation.json').write_text(json.dumps(record,indent=2)+'\n')
with (evidence/'build.log').open('wb') as log:
    subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
central=evidence/'central'; central.mkdir()
candidate=evidence/'candidate'; candidate.mkdir()
expected=json.loads((source/'comparison.json').read_text())
for copy in template['publicationCopies']:
    source_path=audit/Path(copy['source']).relative_to('checkout')
    dest=candidate/Path(copy['target']).name
    shutil.copyfile(source_path,dest)
    name=dest.name
    module=next(m for m in versions if name.startswith(m+'-'))
    version=versions[module][1]
    url='https://repo.maven.apache.org/maven2/se/alipsa/'+module+'/'+version+'/'+name
    data=urllib.request.urlopen(url).read()
    assert hashlib.sha256(data).hexdigest()==expected['artifacts'][name]['rawSha256']['central'], name
    (central/name).write_bytes(data)
subprocess.run(['python3',str(root/'tools/release-all/compare_publications.py'),'--central',str(central),'--candidate',str(candidate),'--publications',str(source/'publications.json'),'--output',str(evidence/'comparison.json')],check=True)
actual=json.loads((evidence/'comparison.json').read_text())
assert actual==expected, 'Comparison differs from reviewed historical result; inspect report'
print('RC3 audit comparison matches reviewed historical result',flush=True)
