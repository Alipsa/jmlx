import datetime, hashlib, json, shutil, sys
from pathlib import Path
root=Path('<CANDIDATE>')
sys.path.insert(0,str(root/'tools/release-all'))
from release_inputs import check_bundle, native_report, read_json, sha256
from verify_native_tests import verify_events
preliminary=next((root/'build/releaseAll').glob('preliminary-*.json'))
record=check_bundle(root/'build/releaseAll/jmlx-mega-bundle.zip',preliminary,sha256(preliminary),root/'tools/release-smoke/released-8.4.json',root)
assert sha256(root/'build/releaseAll/bundle-receipt.json')==record['pluginReceiptSha256']
assert sha256(root/'build/releaseAll/manifest.json')==record['pluginReceipt']['manifestHash']
assert json.loads(json.dumps(native_report(root/'native/install/lib',root/'scripts/bootstrap-native.sh',[str(Path.home()),str(root)])))==record['native']
verify_events(read_json(record['taskEvidence']['path']),record['invocationId'])
assert sha256(record['taskEvidence']['path'])==record['taskEvidence']['sha256']
assert sha256(record['gradleArgvFile']['path'])==record['gradleArgvFile']['sha256']
assert read_json(record['gradleArgvFile']['path'])==record['gradleArgv']
for name,digest in record['junit'].items(): assert sha256(root/name)==digest,name
backup=Path('<CANDIDATE>-backup-2026-11-23')
backup.mkdir(exist_ok=False)
paths={root/name for name in record['junit']}
paths.update((root/'native/install/lib').iterdir())
paths.update([preliminary,root/'build/releaseAll/jmlx-mega-bundle.zip',root/'build/releaseAll/bundle-receipt.json',root/'build/releaseAll/manifest.json',Path(record['taskEvidence']['path']),Path(record['gradleArgvFile']['path']),root/'scripts/bootstrap-native.sh',root/'tools/release-smoke/released-8.4.json'])
pins={}
for source in sorted(paths):
    relative=source.relative_to(root)
    dest=backup/'producer-inputs'/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,dest)
    digest=sha256(source)
    assert sha256(dest)==digest,relative
    pins[str(relative)]={'bytes':source.stat().st_size,'sha256':digest}
report={'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'candidateRoot':str(root),'backupRoot':str(backup),'sourceCommit':record['tag']['sha'],'bundleSha256':record['bundleSha256'],'preliminarySha256':sha256(preliminary),'junitReportsVerified':len(record['junit']),'producerInputsVerified':True,'pins':pins,'note':'Producer inputs copied and verified. Keep original candidate paths; backup is restoration material, not a relocatable verifier. Consumer/minimum-host reports and final receipt are preserved separately when available.'}
(backup/'producer-input-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'build/releaseAll/rc3-verification/producer-input-backup.json').write_text(json.dumps(report,indent=2)+'\n')
print('Verified and preserved '+str(len(pins))+' producer inputs',flush=True)
