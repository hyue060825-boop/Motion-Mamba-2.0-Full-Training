from common import *
import subprocess,ast
dest=ROOT/'acceleration_v3';g=read(dest/'EIGHT_RANK_GATE.json');s=read(dest/'SINGLE_GATE.json');t=read(dest/'WARM_TIMING.json');latest=read(ROOT/'LATEST.json')
assert g['passed'] and g['complete'] and s['passed'] and s['complete']
assert g['source_sha256']==s['source_sha256']==sha(dest/'history_v3.py')
assert g['checkpoint']==latest['checkpoint'] and latest['stage_step']==3530 and sha(latest['checkpoint'])==g['checkpoint_sha256']
eligible=t['paired_median_speedup']>1.02 and min(t['pairs'])>1
if eligible:
 write(ROOT/'V3_ADMISSION.json',dict(passed=True,version='canonical_history_only_v3_legacy_transfer',source_sha256=g['source_sha256'],base_backend_sha256=sha(ROOT/'speed_backend.py'),single_gate=str(dest/'SINGLE_GATE.json'),eight_rank_gate=str(dest/'EIGHT_RANK_GATE.json'),warm_timing=str(dest/'WARM_TIMING.json'),paired_speedup=t['paired_median_speedup'],gradient_weight_Adam_exact=True,host_transfer='original admitted transient packing',persistent_pinned_label_cache=False,production_average_speedup_not_assumed=True,time=time.time()))
else:
 (ROOT/'V3_ADMISSION.json').unlink(missing_ok=True)
 write(dest/'FINAL_REJECTION.json',dict(correctness_pass=True,admitted=False,retained_backend='dense',timing=t,time=time.time()))
# Refresh execution metadata even when the optimizer checkpoint came from v3.
path=ROOT/'train.py';code=path.read_text();marker="    if (ROOT/'V3_ADMISSION.json').exists():\n"
assert code.count(marker)==1
code=code.replace(marker,"    ck['_history_acceleration']=dict(version='dense',current_gradient_path='unchanged reference')\n"+marker)
ast.parse(code);(dest/'train_before_transfer_fix.py').write_text(path.read_text());path.write_text(code)
request=ROOT/'BOUNDARY_REQUEST.json';assert read(request)['kind']=='USER_SPEED_V3_PACKING_FIX'
state=read(ROOT/'STATUS.json');assert state['stage']=='PAUSED_FOR_MAINTENANCE'
append(ROOT/'budget/GPU_TIME.jsonl',dict(name='history_v3_transfer_revert_checks_upper_bound',seconds=time.time()-state['time'],devices=8,scope='Conservative full maintenance interval; all updates discarded'))
request.rename(dest/'BOUNDARY_REQUEST_transfer_fix_consumed.json')
if (ROOT/'BUDGET_STOP.json').exists():(ROOT/'BUDGET_STOP.json').rename(dest/'BUDGET_STOP_transfer_fix.json')
with (ROOT/'logs/controller_v3_transfer_resume.log').open('a') as log:child=subprocess.Popen([PY,'-u',str(ROOT/'controller.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(ROOT/'CONTROLLER_PID.json',dict(pid=child.pid,time=time.time(),resume_step=3530,reason='Final verified history computation optimization with original data transfer'))
write(dest/'RESUME_RECEIPT.json',dict(pid=child.pid,time=time.time(),checkpoint=latest['checkpoint'],pool=latest['pool'],same_Adam_and_weights=True,recipe_unchanged=True,global_batch=512,GT_slots=448,teacher_slots=64,backend='canonical_history_v3' if eligible else 'dense',persistent_pinned_cache_removed=True))
print('TRANSFER_FIX_RESUMED',child.pid,eligible,t['paired_median_speedup'],flush=True)
