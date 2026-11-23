import datetime,json,subprocess
from pathlib import Path
base='repos/Alipsa/jmlx/actions'
def gh(endpoint,*args):return subprocess.check_output(['gh','api','--paginate',endpoint,*args],text=True,timeout=60)
ids=set()
for status in ['queued','in_progress','waiting','requested','pending','action_required']:
    ids.update(gh(base+'/runs?status='+status,'--jq','.workflow_runs[].id').split())
record={'observedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[],'matchingJobs':[],'approvalRequiredRuns':[],'limits':'Paginated snapshot of exposed jobs; needs-blocked jobs may not be exposed. Reviewed workflows restrict release label to manual main dispatch.'}
for runid in sorted(ids,key=int):
    run=json.loads(gh(base+'/runs/'+runid))
    info={k:run[k] for k in ['id','path','head_branch','head_sha','event','status','conclusion']}
    record['runs'].append(info)
    if run['status']=='action_required' or run['conclusion']=='action_required':record['approvalRequiredRuns'].append(info)
    jobs=gh(base+'/runs/'+runid+'/jobs','--jq','.jobs[] | select(.status != "completed") | select(.labels | index("jmlx-release-minimum")) | {id,name,status,labels}')
    if jobs.strip():record['matchingJobs'].append({'run':info,'jobs':jobs})
p=Path(__file__).resolve().parent
(p/'queue-review.json').write_text(json.dumps(record,indent=2)+'\n')
(p/('queue-review-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')).write_text(json.dumps(record,indent=2)+'\n')
assert not record['matchingJobs'] and not record['approvalRequiredRuns'], 'Unexpected queued/approval-required jobs; leave runner offline and review queue evidence'
print('Queue review passed')
