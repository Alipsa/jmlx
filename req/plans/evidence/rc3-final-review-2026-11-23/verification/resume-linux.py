from pathlib import Path
import datetime,json,shutil,subprocess
root=Path('<CANDIDATE>')
e=root/'build/releaseAll/rc3-verification/consumers'
record=json.loads((e/'linux-launch-failure-invocation.json').read_text())
record['argv']=['bash','<HOME>/.local/bin/linux-exec']+record['argv'][1:]
record['startedAt']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(e/'linux-pure-invocation.json').write_text(json.dumps(record,indent=2)+'\n')
subprocess.run(record['argv'],check=True)
transport=Path('<REPOSITORY>/build/releaseAll/candidates/rc.3/transport')
for suffix in ['.json','.log','.runtime.xml']:
    shutil.copyfile(transport/('linux-pure'+suffix),e/('linux-pure'+suffix))
print('PASS linux-pure',flush=True)
