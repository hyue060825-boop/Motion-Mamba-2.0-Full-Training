"""Prepare the unchanged controlled endpoint/evaluator_v2 study after the final save."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,shutil,hashlib,time,py_compile
T=Path(artifact_path('g1_native117k_continue5k_8gpu_20261018_bef070b7cc'))
O=Path(artifact_path('native100k_controlled_endpoint_20261017_4c3b70479f'))
R=Path(artifact_path('native122k_controlled_endpoint_20261018_9d1c42d4fe'))
P=T/'train/checkpoint10000.pt'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert P.exists() and read(T/'LATEST.json')['global_step']==122000
if (R/'PREPARED.json').exists():print('REUSE_PREPARED',R);raise SystemExit(0)
R.mkdir(parents=True,exist_ok=True)
for d in ['sampler']:
 if not (R/d).exists():shutil.copytree(O/d,R/d,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['panels','source_protocol','text_cache','manifest.jsonl']:
 if not (R/name).exists():(R/name).symlink_to(O/name,target_is_directory=(O/name).is_dir())
for name in ['util.py','controller.py','run_job.py','score_job.py','aggregate.py','aggregate_reference.py','physical.py']:
 s=(O/name).read_text().replace('native100','native122').replace('NATIVE100K','NATIVE122K').replace('Native100k','Native122k')
 if name=='util.py':s=s.replace(str(O).replace('native100','native122'),str(R))
 if name=='run_job.py':
  assert "'native122':'hybrid'" in s
 if name=='controller.py':
  s=s.replace("in [4,5,6,7]","in range(8)").replace('waiting for unused GPUs4-7','waiting for unused GPUs0-7')
 if name=='aggregate.py':
  s=s.replace("load['global_step']==100000","load['global_step']==122000")
  s=s.replace("command['checkpoint_sha256']=='d21b9a22616fcfa2244629d1d4176a6de0c4f567cc5c69022458321afea0c9ca'","command['checkpoint_sha256']==read(ROOT/'CHECKPOINTS.json')['native122']['sha256']")
 (R/name).write_text(s);py_compile.compile(str(R/name),doraise=True)
shutil.copy2(T/'combine_final_evaluation.py',R/'combine.py')
shutil.copy2(T/'REFERENCE_RESULTS.json',R/'REFERENCE_RESULTS.json')
info=dict(training_root=str(T),checkpoint=str(P),sha256=sha(P),stats_dir=artifact_path('stats_a5227494f3'),step=122000,framework_global_start=112000,mode='hybrid',exposures=None,frozen_file_stat=dict(size=P.stat().st_size,mtime_ns=P.stat().st_mtime_ns))
write(R/'CHECKPOINTS.json',dict(native122=info))
write(R/'sampler/CHECKPOINTS.json',dict(native122=info))
plan=read(O/'PLAN.json');plan.update(status='FROZEN_NATIVE122K_PAIRED117K_CONTINUATION_EVALUATION',created=time.time(),models=['native122'],cards=list(range(8)),scope='Identical506 rootXZ endpoint-only cases, targets, deadlines, seeds, real features and evaluator_v2 as frozen117k reference',prior_corrected_results=artifact_path('horizon_controlled_four_models_20261017_132d28341d'),training_unchanged=True,source_exposure='Known generator training sources; paired checkpoint diagnosis, not unseen-source benchmark')
# Freeze source, checkpoint and all inherited material hashes in the same protocol schema.
for key in ['execution_source_hashes','prepared_hashes']:
 plan[key]={name:sha(R/name) for name in plan[key]}
plan['prepared_hashes']['sampler/CHECKPOINTS.json']=sha(R/'sampler/CHECKPOINTS.json')
write(R/'PLAN.json',plan)
write(R/'PREPARED.json',dict(status='PASS_CPU_COHORT_GOALS_SEEDS_AND_CHECKPOINT_IDENTITIES',checkpoint_identity=info,plan_sha256=sha(R/'PLAN.json'),time=time.time()))
print('PREPARED_FIXED506_PAIRED117K_TO122K_EVALUATION',R)
