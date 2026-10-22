from motion_mamba2.paths import artifact_path
import os,json,time,subprocess,fcntl,hashlib,traceback
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'));PYTHON=artifact_path('python_9f8003379b')
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def read(p):return json.loads(Path(p).read_text())
def write(p,a):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):write(D/'RUN_STATE.json',dict(stage=stage,utc_unix=time.time(),supervisor_pid=os.getpid(),physical_devices=[4,5],logical_global_batch=32,per_rank_batch=8,accumulation_steps=2,**kw));print('STAGE',stage,kw,flush=True)
def exists_pid(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
def free():
 names=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);uuids={line.split(',')[1].strip() for line in names.splitlines() if int(line.split(',')[0]) in [4,5]}
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);return not any(line.split(',')[0].strip() in uuids for line in apps.splitlines() if ',' in line)
def job(name,args,gpus='4,5'):
 log=D/'logs'/f'{name}.log';status('RUNNING_'+name,log=str(log),training_started=name.startswith('train'))
 with log.open('w') as f:
  process=subprocess.Popen([PYTHON]+args,env=dict(ENV,CUDA_VISIBLE_DEVICES=gpus),stdout=f,stderr=subprocess.STDOUT)
  write(D/'ACTIVE_CHILD.json',dict(name=name,pid=process.pid,log=str(log)));rc=process.wait()
 assert rc==0,(name,rc,str(log))
def text_gate():
 import numpy as np
 cfg=read(D/'configs/train.json');captions=read(D/'train/data/CAPTIONS.json');catalog={}
 for root in cfg['text_roots'][::-1]:
  if Path(root).exists():catalog.update({p.stem:p for p in Path(root).glob('*.npy')})
 for i,t in enumerate(captions):
  key=hashlib.sha256(t.encode()).hexdigest();assert key in catalog,('missing_text',t)
  a=np.load(catalog[key]);assert a.size==4096 and np.isfinite(a).all(),('bad_text',str(catalog[key]))
 write(D/'train/TEXT_READY.json',dict(total=len(captions),all_finite=True,all_4096=True,exact_official_encoder=True,prepared_shards=[read(D/f'text_shard{i}_DONE.json') for i in [0,1]]))
def freeze():
 hashes={}
 for root in [D/'code',D/'configs']:
  for p in sorted(root.rglob('*')):
   if p.is_file() and '__pycache__' not in str(p):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 write(D/'EXECUTED_SOURCE_HASHES.json',hashes)
lock=(D/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 assert read(D/'SPEED_GATE.json')['status']=='PASS'
 assert read(D/'DECOMPOSED_EQUIVALENCE.json')['all_pass']
 assert not (D/'train/train.jsonl').exists(),'Explicit resume supervisor required for existing training'
 status('WAITING_OFFICIAL_TEXT',worker_pids=[2343101,2343189],training_started=False)
 deadline=time.time()+7200
 while not all((D/f'text_shard{i}_DONE.json').exists() for i in [0,1]):
  assert time.time()<deadline,'Text preparation exceeded two-hour budget'
  for i,pid in enumerate([2343101,2343189]):
   assert (D/f'text_shard{i}_DONE.json').exists() or exists_pid(pid),('text_worker_failed',i,pid)
  time.sleep(10)
 text_gate();status('WAITING_FREE_GPU4_5',training_started=False)
 while not free():
  assert time.time()<deadline,'Resource wait exceeded preparation deadline';time.sleep(10)
 # Saved state must restore exactly; the next BF16/DDP update is separately checked to registered tolerance.
 job('resume_gate',['-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(D/'resume_gate.py')])
 assert read(D/'RESUME_GATE.json')['status']=='PASS_TOLERANCE_BASED'
 freeze();write(D/'LAUNCH_AUTHORIZATION.json',dict(user_authorized=True,training_adt_enabled=False,devices=[4,5],max_updates=100000,max_sample_exposures=3200000,validation_updates_separate=True,do_not_stop_on_early_quality_alone=True,do_not_modify_production=True))
 job('train_to10',['-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(D/'train.py'),'--until','10'])
 # Keep all exposure and Adam/schedule history from this point. Never restart at step0.
 previous=10
 for target in [10000,50000,100000]:
  job(f'train_to{target}',['-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(D/'train.py'),'--until',str(target),'--resume',str(D/'train'/f'checkpoint{previous}.pt')])
  job(f'evaluation_{target}',[str(D/'evaluate.py'),'--step',str(target)],gpus='4')
  previous=target
 status('COMPLETE_TRAINING_AND_PANELS',step=100000,training_started=True,checkpoint=str(D/'train/checkpoint100000.pt'),model_qualified=False,qualification='Read complete DEV/FINAL evidence and compare with original Native50k; no automatic success/deployment claim')
except BaseException as e:
 status('STOPPED_ERROR',error=repr(e),traceback=traceback.format_exc(),production_modified=False);raise
