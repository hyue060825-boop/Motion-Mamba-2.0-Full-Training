"""Admit only measured accelerator, preserve Adam/recipe and restart controller."""
from common import *
import hashlib,shutil,subprocess
from motion_training.io import sha
full=read(ROOT/'gates/GEOMETRY_FULL_UPDATE.json');assert full['passed'] and full['four_rank_states_exact']
for micro,name in [(8,'GEOMETRY_BATCH_PROBE.json'),(64,'GEOMETRY_BATCH64_PROBE.json')]:
 result=read(ROOT/'gates'/name);assert len(result['cases'])==6
 passed=all(x['batch_gradient_relative']<=1e-3 and x['batch_loss_relative']<=1e-5 for x in result['cases'])
 if micro==64:assert passed
 write(ROOT/'gates'/f'GEOMETRY_BATCH{micro}_ADMISSION.json',dict(passed=passed,selected_backend='B1_geometry_reference' if micro==8 else 'batch_geometry',per_rank_microbatch=micro,cases=result['cases'],checkpoint=result['checkpoint'],loss_relative_limit=1e-5,gradient_relative_limit=1e-3))
latest=read(ROOT/'LATEST.json');assert Path(full['checkpoint']).name==Path(latest['checkpoint']).name
assert sha(latest['checkpoint'])==full['checkpoint_sha256']
assert read(ROOT/'STATUS.json')['stage']=='SAVED_AT_COMPLETE_BOUNDARY'
train=ROOT/'train.py';backup=ROOT/'train.py.before_native_geometry_accel'
assert not backup.exists(),'Already patched: inspect before retry'
shutil.copy2(train,backup);text=train.read_text();assert text.count('engine.update=accelerated_loss.install()')==1
text=text.replace('engine.update=accelerated_loss.install()','engine.update=__import__("native_geometry_accel").install()')
train.write_text(text);compile(text,str(train),'exec')
proof=dict(passed=True,architecture=read(ROOT/'PLAN.json')['architecture'],time=time.time(),resume_checkpoint=latest,checkpoint_sha256=full['checkpoint_sha256'],source_hashes={name:sha(ROOT/name) for name in ['pipeline_accel.py','codec_graph.py','accelerated_loss.py','native_geometry_accel.py','train.py']},gradient_relative_limit=1e-3,loss_relative_limit=1e-5,full_four_rank_update='GEOMETRY_FULL_UPDATE.json',microbatch_gates=['GEOMETRY_BATCH8_ADMISSION.json','GEOMETRY_BATCH64_ADMISSION.json'],same_loss_data_noise_optimizer_scheduler=True,small_batch_backend='unchanged_B1_reference',large_batch_backend='admitted_B64_geometry',precision='BF16 denoiser; FP32 frozen codec, geometry, masters; batched geometry GEMM tolerance verified; not bitwise',discarded_probe_updates=8)
write(ROOT/'gates/NATIVE_GEOMETRY_ADMISSION.json',proof)
plan=read(ROOT/'PLAN.json');plan['training_acceleration']='Native B64 geometry admitted; B8 original B1 geometry retained; frozen B1 encoder Graph; CPU prefetch4; denoiser microbatch';plan['acceleration_changed_at_step']=latest['step'];write(ROOT/'PLAN.json',plan)
request=ROOT/'BOUNDARY_REQUEST.json';assert read(request)['resume_after_validation']
request.rename(ROOT/'gates/ACCELERATION_BOUNDARY_REQUEST_COMPLETED.json')
log=(ROOT/'logs/controller_native_geometry_accel.log').open('ab')
proc=subprocess.Popen([PY,'controller.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
write(ROOT/'gates/ACCELERATION_RESTART.json',dict(pid=proc.pid,time=time.time(),resume=latest,optimizer_and_schedule_preserved=True))
print('RESTARTED',ROOT.name,latest['step'],proc.pid,flush=True)
