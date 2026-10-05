#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
OUTPUT="$ROOT/build/phase71-probes"
mkdir -p "$OUTPUT"
RESULT=0
for CASE in se.alipsa.jmlx.core.Phase71ConvolutionProbe; do
  CASE_DIR="$OUTPUT/${CASE##*.}"
  rm -rf "$CASE_DIR"
  mkdir -p "$CASE_DIR"
  rm -rf jmlx-core/build/test-results/phase71NativeProbe
  ./gradlew :jmlx-core:phase71NativeProbe --tests "$CASE" --rerun > "$CASE_DIR/gradle.log" 2>&1
  STATUS=$?
  if ! python3 - "$CASE" "$STATUS" "$CASE_DIR" <<'PY'
import json,sys,shutil
from pathlib import Path
from xml.etree import ElementTree as ET
case,status,destination=sys.argv[1],int(sys.argv[2]),Path(sys.argv[3])
reports=Path('jmlx-core/build/test-results/phase71NativeProbe')
if reports.exists():
    shutil.copytree(reports,destination/'results',dirs_exist_ok=True)
xml=reports/('TEST-'+case+'.xml')
outcome={'case':case,'exitStatus':status,'outcome':'infrastructure-error'}
try:
    root=ET.parse(xml).getroot()
    tests=list(root.iter('testcase'))
    skipped=sum(t.find('skipped') is not None for t in tests)
    executed=len(tests)-skipped
    failures=sum(t.find('failure') is not None or t.find('error') is not None for t in tests)
    outcome.update(executed=executed,skipped=skipped,failures=failures)
    if executed>0 and skipped==0:
        outcome['outcome']='executed-success' if status==0 and failures==0 else 'unexpected-failure'
except (OSError,ET.ParseError) as error:
    outcome['reason']=str(error)
# No process termination is an expected outcome for the current cases. Missing XML always fails.
(destination/'summary.json').write_text(json.dumps(outcome,indent=2)+'\n')
print(json.dumps(outcome))
sys.exit(outcome['outcome']!='executed-success')
PY
  then RESULT=1; fi
  cat "$CASE_DIR/gradle.log"
done
python3 - "$OUTPUT" <<'PYCODE'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
cases=[json.loads((root/'Phase71ConvolutionProbe'/'summary.json').read_text())]
summary={'cases':cases,'executed':sum(c.get('executed',0) for c in cases),
         'skipped':sum(c.get('skipped',0) for c in cases),
         'passed':all(c['outcome']=='executed-success' for c in cases)}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
PYCODE
exit "$RESULT"
