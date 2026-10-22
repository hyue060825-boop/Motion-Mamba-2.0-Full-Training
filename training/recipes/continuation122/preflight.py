from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,sys,ast,torch
R=Path(artifact_path('g1_native117k_continue5k_8gpu_20261018_bef070b7cc'));sys.path.insert(0,str(R/'code/src'))
from motion_training.io import identity
cfg=json.loads((R/'config.json').read_text());reg=json.loads((R/'CONTINUATION_REGISTRATION.json').read_text());P=Path(cfg['parent'])
c=torch.load(P,map_location='cpu',weights_only=False,mmap=True);f=c['framework'];assert c['step']==117000 and f['stage_step']==5000
assert len(c['optimizer']['state'])==422 and len(c['rng_by_rank'])==8 and len(set(f['rank_state_hashes']))==1
assert f['scheduler']['completed']==5000 and f['scheduler']['spec']==cfg['scheduler']
assert f['recipe_hash']==reg['old_recipe_hash'],(f['recipe_hash'],reg['old_recipe_hash'])
# Record the inherited GT stream continuation without editing the original template.
sys.path[:0]=[str(R/'code/src/motion_training/_runtime'),str(R/'accel'),str(R)]
import lazy_plan
new=lazy_plan.Plan(json.loads((R/'MANIFEST_TEMPLATE.json').read_text()))
old=lazy_plan.Plan(json.loads((Path(artifact_path('g1_native112k_future_interaction_20261011_d06c37e3c2'))/'MANIFEST_TEMPLATE.json').read_text()))
a=new.records_at(5000,448);b=old.records_at(5000,448);assert identity(a)==identity(b)
(R/'PARENT_STATE_GATE.json').write_text(json.dumps(dict(passed=True,parent=str(P),global_step=c['step'],stage_step=f['stage_step'],Adam_states=len(c['optimizer']['state']),eight_rank_RNG=True,rank_state_hashes_equal=True,scheduler=f['scheduler'],GT_sampler_next_update5001_exact=True,GT_next_batch_hash=identity(a),no_parent_checkpoint_rewritten=True),indent=2)+'\n')
print('PREFLIGHT_PASS_COMPLETE117K_ADAM_EIGHT_RNG_AND_CONTINUED_GT_STREAM')
