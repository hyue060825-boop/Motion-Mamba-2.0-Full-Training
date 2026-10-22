"""Add three Native checkpoints without mutating or rerunning completed baselines."""
from pathlib import Path
import os,sys,time,subprocess,fcntl,traceback,json
import execute
from execute import R,rd,wr,sha
from controller import free,acquire,PY
ADDED=['native110','native117','native122']
E=R/'extension_native_finetunes_20261021'
def valid_complete(model,task,scored=False):
 root=R/'scoring/models' if scored else R/'models';p=root/model/task/('COMPLETE.json' if scored else 'GENERATION_COMPLETE.json')
 return p.exists() and rd(p).get('passed',False)
def preflight():
 execute.verify();registry=rd(R/'CHECKPOINTS.json');rows=[]
 for m in ADDED:
  info=registry[m];cp=Path(info['checkpoint']);st=cp.stat();f=info['frozen_file_stat']
  assert st.st_size==f['size'] and st.st_mtime_ns==f['mtime_ns'],('changed_checkpoint',m)
  h=sha(cp);assert h==info['sha256'],('checkpoint_sha256',m)
  assert (execute.H/'models'/m/'sampler/checkpoint_loader.py').exists()
  rows.append(dict(model=m,step=info['step'],checkpoint=str(cp),sha256=h))
 original=R/'delivery/COMPLETE.json';assert rd(original)['passed']
 wr(E/'PREFLIGHT.json',dict(passed=True,checkpoints=rows,original_protocol=sha(R/'FROZEN.json'),original_delivery=sha(original),execution_engine=sha(R/'execute.py'),score_engine=sha(R/'score.py'),scoring_assets=rd(R/'scoring/ASSET_BINDINGS.json'),weights_or_training_changed=False,time=time.time()))
def main():
 E.mkdir(exist_ok=True);lock=(E/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wr(E/'CONTROLLER_PID.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),script=str(Path(__file__).resolve()),time=time.time()))
 preflight();tasks=['nominal40','nominal20','nominal9','short52'];jobs=[(m,t) for t in tasks for m in ADDED]
 plan=dict(added_models=ADDED,jobs=[list(x) for x in jobs],new_conditions=12,new_formal_rollouts=4608,independent_sources_per_condition=128,noises_per_source=3,combined_models=12,combined_conditions=48,combined_formal_rollouts=18432,native122_selection='regular117k_continue5k_8gpu; not long_horizon_distill122k',frozen_protocol=sha(R/'FROZEN.json'),same_reference_banks_and_pools=True,original_36_conditions_not_regenerated=True,time=time.time())
 if not (E/'EXECUTION_PLAN.json').exists():wr(E/'EXECUTION_PLAN.json',plan)
 queue=[k for k in jobs if not valid_complete(*k)];active={};scores={};genfailed=[];scorefailed=[];start=time.time()
 while True:
  for key,v in list(active.items()):
   rc=v['p'].poll()
   if rc is None:continue
   v['log'].close()
   for f in v['locks']:f.close()
   del active[key]
   if rc:genfailed.append(list(key));print('GENERATION_FAILED',key,rc,flush=True)
   else:print('GENERATION_COMPLETE',key,flush=True)
  for key,v in list(scores.items()):
   rc=v['p'].poll()
   if rc is None:continue
   v['log'].close();del scores[key]
   if rc:scorefailed.append(list(key));print('SCORE_FAILED',key,rc,flush=True)
   else:print('SCORE_COMPLETE',key,flush=True)
  for card in free():
   if not queue:break
   files=acquire(card)
   if not files:continue
   m,t=queue.pop(0);d=R/'models'/m/t;d.mkdir(parents=True,exist_ok=True);log=(d/'worker.log').open('a')
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(card),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
   p=subprocess.Popen([PY,str(R/'native_extension_worker.py'),'--model',m,'--task',t],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   active[(m,t)]=dict(p=p,card=card,log=log,locks=files,started=time.time());print('REAL_GPU_STARTED',m,t,card,p.pid,flush=True)
  ready=[k for k in jobs if valid_complete(*k) and not valid_complete(*k,scored=True) and k not in scores and list(k) not in scorefailed]
  for m,t in ready[:max(0,3-len(scores))]:
   d=R/'scoring/models'/m/t;d.mkdir(parents=True,exist_ok=True);log=(d/'worker.log').open('a')
   p=subprocess.Popen([PY,str(R/'score.py'),'--model',m,'--task',t],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   scores[(m,t)]=dict(p=p,log=log);print('CPU_SCORE_STARTED',m,t,p.pid,flush=True)
  complete=[list(k) for k in jobs if valid_complete(*k)];scored=[list(k) for k in jobs if valid_complete(*k,scored=True)]
  workers=[dict(model=k[0],task=k[1],pid=v['p'].pid,card=v['card'],started=v['started']) for k,v in active.items()]
  wr(E/'ACTIVE_CHILD.json',dict(generation=workers,scoring=[dict(model=k[0],task=k[1],pid=v['p'].pid) for k,v in scores.items()],time=time.time()))
  wr(E/'STATUS.json',dict(stage='GATING_GENERATING_AND_SCORING_ADDITIONAL_NATIVE_CHECKPOINTS',completed_generation=complete,completed_scoring=scored,queued=[list(k) for k in queue],active_generation=workers,generation_failures=genfailed,score_failures=scorefailed,total_new_conditions=12,wall_seconds=time.time()-start,time=time.time()))
  if len(scored)==len(jobs):break
  if not queue and not active and not scores and (genfailed or scorefailed):
   wr(E/'FAILURE.json',dict(generation=genfailed,scoring=scorefailed,time=time.time()));raise RuntimeError('Additional evaluation requires execution repair')
  time.sleep(10)
 with (E/'finalize.log').open('a') as log:subprocess.run([PY,str(E/'finalize_extension.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,check=True)
 wr(E/'STATUS.json',dict(stage='ADDITIONAL_THREE_NATIVE_MODELS_COMPLETE_COMBINED_RESULTS_READY',completed_generation=12,completed_scoring=12,combined_conditions=48,delivery=str(E/'delivery'),local_manuscript_compilation_pending=True,time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:wr(E/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
