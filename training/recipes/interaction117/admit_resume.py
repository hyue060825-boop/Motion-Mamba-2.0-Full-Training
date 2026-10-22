from common import *
import ast,shutil,subprocess,collections
import torch
v=ROOT/'acceleration_v5'
assert not read(ROOT/'budget/ACTIVE.json')
assert sha(ROOT/'config.json')=='b20fa0fed92dac5ca8ac4ddbbe0169a3da4eeb386a4eb690a00e0193839471c4'
assert subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()==''
for fn,mode in [('CANDIDATE_GATES.json','graph'),('EXTRA_GATES.json','graph_extra'),('ENCODE_GATES.json','graph_extra_encode')]:
 checks=read(v/fn)['candidates'];assert next(x for x in checks if x['mode']==mode)['passed']
assert read(v/'LIVE_WEIGHT_GATE.json')['passed']
final=read(v/'FINAL_TIMING.json');assert final['complete']
assert final['medians']['optimized']<final['medians']['current']
condition=final['conditioning']
if condition:assert read(v/'CONDITION_GATES.json')['candidates'][0]['passed']
saved=read(ROOT/'LATEST.json');assert saved['stage_step']==1180 and saved['rank_state_hashes_verified']
cp=saved['checkpoint'];pool=saved['pool'];cfg=read(ROOT/'config.json')
ck=torch.load(cp,map_location='cpu',weights_only=False);f=ck['framework']
assert ck['step']==113180 and f['stage_step']==1180
assert f['recipe_hash']==identity(cfg) and f['world_size']==8
assert f['pool_path']==pool and f['pool_hash']==sha(pool)
assert f['manifest_hash']==sha(ROOT/'train/data/MANIFEST.json')
assert len(ck['rng_by_rank'])==8 and len(f['rank_state_hashes'])==8 and len(set(f['rank_state_hashes']))==1
checkpoint_sha=sha(cp);del ck
receipt=dict(passed=True,mode='graph',extra_graph=True,encode_graph=True,condition_cache=condition,
 backend_sha256=sha(v/'history_v5.py'),extra_graph_sha256=sha(v/'extra_graph.py'),encode_graph_sha256=sha(v/'encode_graph.py'),
 condition_cache_sha256=sha(v/'condition_cache.py') if condition else None,
 checkpoint=cp,checkpoint_sha256=checkpoint_sha,pool=pool,pool_sha256=sha(pool),
 strict_complete_update_gate='exact input/state/gradient/loss/clip/Adam/weights/RNG',
 fresh_weight_gate=str(v/'LIVE_WEIGHT_GATE.json'),matched_final_timing=str(v/'FINAL_TIMING.json'),
 no_gradient_history_only=True,current_gradient_path_unchanged=True,recipe_sha256=sha(ROOT/'config.json'),time=time.time())
write(v/'ADMISSION.json',receipt)
write(v/'RESUME_ADMISSION.json',dict(passed=True,checkpoint=cp,checkpoint_sha256=checkpoint_sha,pool=pool,pool_sha256=sha(pool),step=1180,recipe_sha256=sha(ROOT/'config.json'),full8rank_optimizer_and_rng_retained=True))
for name in ['train_future.py','controller_future.py']:
 backup=v/(name+'.before')
 if not backup.exists():shutil.copy2(ROOT/name,backup)
 candidate=v/('candidate_'+name);ast.parse(candidate.read_text());shutil.copy2(candidate,ROOT/name)
if (ROOT/'BOUNDARY_REQUEST.json').exists():
 (ROOT/'BOUNDARY_REQUEST.json').rename(v/'BOUNDARY_REQUEST.saved.json')
if (ROOT/'FAILURE.json').exists():(ROOT/'FAILURE.json').rename(v/'PREVIOUS_FAILURE.json')
env=dict(os.environ,PYTHONUNBUFFERED='1')
log=(ROOT/'logs/controller_future_v5.log').open('a')
p=subprocess.Popen([PY,'controller_future.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(v/'RESUMPTION.json',dict(pid=p.pid,start=1180,continuation='1180→2500; refresh actual student pool→5000; original evaluation and stop rules',
 checkpoint_sha256=checkpoint_sha,recipe_sha256=sha(ROOT/'config.json'),entry_sha256=sha(ROOT/'train_future.py'),controller_sha256=sha(ROOT/'controller_future.py'),time=time.time()))
print('RESUMED_CONTROLLER',p.pid,flush=True)
