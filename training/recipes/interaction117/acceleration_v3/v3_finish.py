from common import *
import subprocess
dest=ROOT/'acceleration_v3';gate=read(dest/'EIGHT_RANK_GATE.json');timing=read(dest/'WARM_TIMING.json')
assert gate['complete']
assert read(ROOT/'LATEST.json')['stage_step']==3490
eligible=gate['passed'] and timing['paired_median_speedup']>1.02 and min(timing['pairs'])>1
pause=read(ROOT/'STATUS.json')['time'];assert read(ROOT/'STATUS.json')['stage']=='PAUSED_FOR_MAINTENANCE'
append(ROOT/'budget/GPU_TIME.jsonl',dict(name='user_requested_history_v3_study_conservative_upper_bound',seconds=time.time()-pause,devices=8,scope='Charges all eight devices throughout maintenance, including CPU-only and idle intervals. Formal 10k update and 192 GPU-hour caps unchanged. All study updates discarded.',time=time.time()))
if eligible:
 exec(compile((dest/'v3_admit_resume.py').read_text(),str(dest/'v3_admit_resume.py'),'exec'),globals())
else:
 write(dest/'REJECTION.json',dict(admitted=False,correctness_pass=gate['passed'],timing=timing,reason='No stable measured full-step gain above admission threshold; retain already admitted dense backend',online_runtime_not_changed=True,time=time.time()))
 request=ROOT/'BOUNDARY_REQUEST.json';assert read(request)['kind']=='USER_SPEED_MAINTENANCE_V3';request.rename(dest/'BOUNDARY_REQUEST_consumed.json')
 if (ROOT/'BUDGET_STOP.json').exists():(ROOT/'BUDGET_STOP.json').rename(dest/'BUDGET_STOP_maintenance_only.json')
 with (ROOT/'logs/controller_v3_resume.log').open('a') as log:child=subprocess.Popen([PY,'-u',str(ROOT/'controller.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(ROOT/'CONTROLLER_PID.json',dict(pid=child.pid,time=time.time(),resume_step=3490,reason='Optimization study complete; preserve dense backend'))
 write(dest/'RESUME_RECEIPT.json',dict(pid=child.pid,checkpoint=read(ROOT/'LATEST.json')['checkpoint'],pool=read(ROOT/'LATEST.json')['pool'],same_Adam_and_weights=True,recipe_unchanged=True,global_batch=512,GT_slots=448,teacher_slots=64,backend='dense',time=time.time()))
 print('DENSE_RESUMED',child.pid,flush=True)
