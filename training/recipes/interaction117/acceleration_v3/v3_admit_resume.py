from common import *
import ast,subprocess,statistics
dest=ROOT/'acceleration_v3';gate=read(dest/'EIGHT_RANK_GATE.json');single=read(dest/'SINGLE_GATE.json');timing=read(dest/'WARM_TIMING.json')
assert gate['passed'] and gate['complete'] and single['passed'] and single['complete']
assert gate['source_sha256']==single['source_sha256']==sha(dest/'history_v3.py')
assert timing['paired_median_speedup']>1.02,'Do not admit without a measured full-update gain'
assert timing['pairs'] and min(timing['pairs'])>1
latest=read(ROOT/'LATEST.json');assert latest['stage_step']==3490 and latest['checkpoint']==gate['checkpoint']
assert sha(latest['checkpoint'])==gate['checkpoint_sha256']
recipe_sha=sha(ROOT/'config.json')
path=ROOT/'train.py';source=path.read_text();backup=dest/'train_before_v3.py'
assert not backup.exists(),'Avoid double admission'
backup.write_text(source)
marker="    speed_mode=admission['mode'];speed_backend.install(speed_mode,audit=False)\n"
assert source.count(marker)==1
insert="""    if (ROOT/'V3_ADMISSION.json').exists():
        v3=read(ROOT/'V3_ADMISSION.json')
        assert v3['passed'] and v3['source_sha256']==sha(ROOT/'acceleration_v3/history_v3.py')
        assert v3['base_backend_sha256']==sha(ROOT/'speed_backend.py')
        sys.path.insert(0,str(ROOT/'acceleration_v3'))
        import history_v3
        history_v3.install('canonical',audit=False)
        speed_mode='canonical_history_v3'
        ck['_history_acceleration']=dict(version=history_v3.VERSION,admission=str(ROOT/'V3_ADMISSION.json'),current_gradient_path='unchanged reference')
"""
updated=source.replace(marker,marker+insert);ast.parse(updated)
write(ROOT/'V3_ADMISSION.json',dict(passed=True,source_sha256=sha(dest/'history_v3.py'),base_backend_sha256=sha(ROOT/'speed_backend.py'),single_gate=str(dest/'SINGLE_GATE.json'),eight_rank_gate=str(dest/'EIGHT_RANK_GATE.json'),warm_timing=str(dest/'WARM_TIMING.json'),paired_speedup=timing['paired_median_speedup'],gradient_weight_Adam_exact=True,scope='Training-only history reconstruction and immutable host labels; current-gradient forward and online control runtime unchanged',time=time.time()))
path.write_text(updated)
assert sha(ROOT/'config.json')==recipe_sha # Recipe file is never rewritten.
request=ROOT/'BOUNDARY_REQUEST.json';assert read(request)['kind']=='USER_SPEED_MAINTENANCE_V3'
request.rename(dest/'BOUNDARY_REQUEST_consumed.json')
if (ROOT/'BUDGET_STOP.json').exists():(ROOT/'BUDGET_STOP.json').rename(dest/'BUDGET_STOP_maintenance_only.json')
processes=subprocess.check_output(['ps','-u','yekangjie','-o','args='],text=True)
assert not any('python' in line and 'controller.py' in line and str(ROOT) in line for line in processes.splitlines()),'Another controller is already active'
with (ROOT/'logs/controller_v3_resume.log').open('a') as log:
 child=subprocess.Popen([PY,'-u',str(ROOT/'controller.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(ROOT/'CONTROLLER_PID.json',dict(pid=child.pid,time=time.time(),resume_step=3490,reason='Admitted exact canonical history acceleration v3'))
write(dest/'RESUME_RECEIPT.json',dict(pid=child.pid,checkpoint=latest['checkpoint'],pool=latest['pool'],same_Adam_and_weights=True,recipe_unchanged=True,global_batch=512,GT_slots=448,teacher_slots=64,time=time.time(),train_sha256=sha(path),train_backup=str(backup)))
print('V3_ADMITTED_RESUMED',child.pid,timing['paired_median_speedup'],flush=True)
