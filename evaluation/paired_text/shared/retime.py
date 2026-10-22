"""Install admitted external literal-core caches, gate, and remeasure saved outputs."""
import os,sys,time,traceback,argparse
from execute import R,rd,wr,sha,setup,backend,run,schedule
import numpy as np
METHODS=['fd2','omg','cmdm','motionstreamer']
def worker(method,task):
 out=R/'retiming'/method/task;out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():return
 registry=rd(R/'CHECKPOINTS.json');root=registry[method]['training_root']
 receipt=__import__('pathlib').Path(root)/'gates/CORE_ACCELERATION.json'
 assert receipt.exists(),'Missing original core-acceleration admission receipt'
 admission=rd(receipt)
 if not admission['admit']:
  # A rejected speed candidate is a required reference fallback, not an
  # inference failure. Never force a kernel past the original admission.
  formal=R/'models'/method/task
  gate=rd(formal/'ACCELERATION_GATE.json');assert gate['passed'] and len(gate['rows'])==4
  assert all(all(v['checks'].values()) for v in gate['rows'])
  latency=rd(formal/'LATENCY.json');assert latency['backend']=='reference' and latency['timed_sequences']==32 and latency['warmup_sequences']==2
  latency.update(original_core_candidate_admitted=False,reference_fallback_reason=admission['reason'],original_admission_sha256=sha(receipt),formal_latency_unchanged=True)
  wr(out/'LATENCY.json',latency)
  wr(out/'ACCELERATION_GATE.json',dict(passed=True,backend='reference',candidate_admitted=False,original_admission_sha256=sha(receipt),formal_gate_sha256=sha(formal/'ACCELERATION_GATE.json'),reason=admission['reason'],no_kernel_or_model_substitution=True))
  wr(out/'COMPLETE.json',dict(passed=True,model=method,task=task,reference_fallback=True,formal_geometry_and_RNG_unchanged=True,time=time.time()))
  return
 os.environ['BASELINE_RUN_ROOT']=root
 model,m,client=setup(method,out);raw=m.denoiser.model
 assert hasattr(raw,'_set_exact_core_acceleration'),('admitted_external_core_not_attached',method)
 assets=dict(np.load(R/'INPUT_ASSETS.npz'));cases=[__import__('json').loads(s) for s in (R/'CASES.jsonl').open()];dev=[c for c in cases if c['split']=='development'][:4];test=[c for c in cases if c['split']=='locked_test']
 features=dict(zip(rd(R/'TEXT_INDEX.json')['prompts'],np.load(R/'TEXT_FEATURES.npy')));checks=[]
 for c in dev:
  backend(method,m,client,False);a,aa=run(method,model,m,client,c,task,c['noise_seeds'][0],assets,features,gate=True)
  candidate=backend(method,m,client,True);b,bb=run(method,model,m,client,c,task,c['noise_seeds'][0],assets,features,gate=True)
  chk={k:bool(np.array_equal(a[k],b[k])) for k in a};chk.update(RNG=aa['final_RNG']==bb['final_RNG'],events=aa['events']==bb['events'])
  checks.append(dict(end=c['selected_end_effector'],checks=chk));wr(out/'GATE_PROGRESS.json',dict(rows=checks))
  assert all(chk.values()),('external_core_exact_gate_failed',method,task,checks[-1])
 wr(out/'ACCELERATION_GATE.json',dict(passed=True,backend=candidate,rows=checks,original_admission=sha(receipt),source_sha256=sha(__file__),inference_source_sha256=sha(R/'execute.py')))
 backend(method,m,client,True)
 for c in test[:2]:run(method,model,m,client,c,task,c['noise_seeds'][0],assets,features,timing=True)
 rows=[]
 for i,c in enumerate(test[:32]):
  _,a=run(method,model,m,client,c,task,c['noise_seeds'][0],assets,features,timing=True)
  formal=rd(R/'models'/method/task/'trials'/f'{i:03d}_s0'/'INPUT_AND_EVENTS.json')
  for k in ['physical_sha256','joint_sha256','root_sha256','final_RNG']:assert a[k]==formal[k],('retiming_changed_formal_result',method,task,i,k)
  rows.append(dict(case_id=c['case_id'],elapsed_s=a['elapsed_s'],formal_output_RNG_identical=True))
 W=len(schedule(task)[0]);times=np.asarray([x['elapsed_s'] for x in rows])*1000/W
 wr(out/'LATENCY.json',dict(backend=candidate,batch=1,warmup_sequences=2,timed_sequences=32,windows=W,mean_ms_window=float(times.mean()),median_ms_window=float(np.median(times)),p95_ms_window=float(np.quantile(times,.95)),rows=rows,scope='amortized per52-frame accepted window, model+decode+commit, cached current text; exact matched formal motion and RNG',time=time.time()))
 wr(out/'COMPLETE.json',dict(passed=True,model=method,task=task,formal_geometry_and_RNG_exact=True,time=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--task',required=True);a=p.parse_args()
 try:worker(a.model,a.task)
 except BaseException:wr(R/'retiming'/a.model/a.task/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
