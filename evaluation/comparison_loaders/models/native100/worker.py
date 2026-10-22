from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,shutil,subprocess,traceback
D=Path(__file__).resolve().parent;R=D.parent;MODEL=os.environ['EVAL_MODEL']
sys.path[:0]=[str(D),str(R)]
import util,harness
P=util.read(D/'PLAN.json');TASKS=['short52','aligned_w04','aligned_w09','aligned_w19'];native=MODEL.startswith('native')

def run(task,kind,backend,phase,batch):
 import run_job
 q=D/'panels'/f'{task}_{kind}.jsonl';dest=D/'panels'/f'{task}_shard00.jsonl';shutil.copy2(q,dest)
 util.write(D/'PHASE_PANEL.json',dict(sha256=util.sha(dest),kind=kind,N=len(util.rows(dest))))
 os.environ.update(RUN_PHASE=phase,EVAL_BACKEND=backend,EVAL_BATCH_SIZE=str(batch))
 sys.argv=['run_job.py','--model',MODEL,'--task',task,'--shard','0']
 out=D/phase/MODEL/task/'shard00'
 if (out/'COMPLETE.json').exists():return out
 util.write(D/'STATUS.json',dict(stage=phase,task=task,backend=backend,batch=batch,time=time.time()))
 run_job.main();harness.flush(out)
 return out
try:
 if native:
  gates=[]
  for task in TASKS:
   ref=run(task,'gate','reference','gate_reference',2);opt=run(task,'gate','optimized','gate_optimized',2)
   a=util.read(ref/'TIMING_DETAIL.json')['rows'];b=util.read(opt/'TIMING_DETAIL.json')['rows']
   assert len(a)==len(b)==2
   same=[x['hashes']==y['hashes'] for x,y in zip(a,b)];assert all(same),(MODEL,task,a,b)
   import numpy as np
   with np.load(ref/'geometry.npz') as x,np.load(opt/'geometry.npz') as y:
    assert x.files==y.files
    for k in x.files:assert np.array_equal(x[k],y[k]),(MODEL,task,k)
   replays=sum(x['graph_delta']['replays'] for x in b);assert replays>0,(MODEL,task,'Graph never used')
   gates.append(dict(task=task,output_state_RNG_exact=True,full_geometry_exact=True,graph_replays=replays))
   util.write(D/'KERNEL_GATE.json',dict(passed=True,checks=gates,complete=len(gates)==4,source=util.read(D/'CHECKPOINTS.json')[MODEL],fixed_tolerances='bitwise exact, no relaxation'))
 else:util.write(D/'KERNEL_GATE.json',dict(passed=True,complete=True,Native_kernel_not_applied_to_external_model=True))
 formal=run('short52','full','optimized' if native else 'reference','evaluation',2)
 # Dedicated B1 latencies on the same preselected32 sources for every method.
 timing=[]
 for task in TASKS:
  dest=run(task,'timing','optimized' if native else 'reference','timing',1)
  rows=util.read(dest/'TIMING_DETAIL.json')['rows'];timed=[r for r in rows if not r['warmup']]
  assert len(timed)==32 and all(r['batch']==1 for r in timed)
  if native:assert all(r['graph_delta']['builds']==0 for r in timed),(MODEL,task,'unexpected timed Graph capture')
  import numpy as np
  a=[r['elapsed_seconds'] for r in timed]
  timing.append(dict(model=MODEL,task=task,N=32,batch=1,sequence_p50_s=float(np.median(a)),sequence_mean_s=float(np.mean(a)),sequence_p95_s=float(np.quantile(a,.95)),generated_windows=P['tasks'][task]['goal_window'],amortized_window_p50_ms=1000*float(np.median(a))/P['tasks'][task]['goal_window'],backend=os.environ['EVAL_BACKEND'],graph_replays=sum(r.get('graph_delta',{}).get('replays',0) for r in timed),new_timed_graphs=sum(r.get('graph_delta',{}).get('builds',0) for r in timed),scope='warmed entire sequence B1; two full warmups;32 paired fixed source IDs; excludes loading/text encoding/first Graph capture/audit IO; not derived from old219ms'))
  util.write(D/'LATENCY.json',dict(passed=True,rows=timing))
 # CPU feature extraction is saved separately from GPU generation time.
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
 with (D/'score52.log').open('a') as log:
  code=subprocess.run([artifact_path('python_9f8003379b'),str(D/'score_job.py'),'--job',str(formal)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
 assert code==0,('52 features failed',code)
 util.write(D/'COMPLETE_ALL.json',dict(passed=True,model=MODEL,formal52_N=506,latency32x4=True,kernel_gate=True,time=time.time()))
 util.write(D/'STATUS.json',dict(stage='COMPLETE_KERNEL_LATENCY_AND_FULL52',time=time.time()))
except BaseException:
 util.write(D/'FAILURE_NEW.json',dict(error=traceback.format_exc(),time=time.time()));raise
