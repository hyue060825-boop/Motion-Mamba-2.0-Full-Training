from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,time,statistics,math,hashlib,os
r=Path(artifact_path('g1_native112k_future_interaction_20261011_d06c37e3c2'))
rows=[json.loads(l) for l in (r/'train/train.jsonl').read_text().splitlines() if l.strip()]
new=[row for row in rows if row['step']>1000]
assert new and all(math.isfinite(row['loss']) and math.isfinite(row['clip']) for row in new)
assert all(row['world_size']==8 and row['global_batch']==512 and row['GT_slots']==448 and row['teacher_slots']==64 for row in new)
assert not (r/'FAILURE.json').exists()
assert hashlib.sha256((r/'config.json').read_bytes()).hexdigest()=='b20fa0fed92dac5ca8ac4ddbbe0169a3da4eeb386a4eb690a00e0193839471c4'
first=new[0];updates=[row['future_projection_update_norm']['future_constraints_proj.weight'] for row in first['ranks']]
assert len(updates)==8 and min(updates)>0 and len(set(updates))==1
live=json.loads((r/'LIVE.json').read_text());status=json.loads((r/'STATUS.json').read_text())
assert time.time()-live['time']<60 and status['stage']=='train_1000_2500'
os.kill(status['pid'],0)
receipt=dict(time=time.time(),latest_step=new[-1]['step'],global_step=new[-1]['global_step'],resumed_updates=len(new),warm_updates_excluding_first2=len(new[2:]),warm_median_seconds=statistics.median(row['seconds'] for row in new[2:]),warm_min_seconds=min(row['seconds'] for row in new[2:]),warm_max_seconds=max(row['seconds'] for row in new[2:]),last10=[dict(step=row['step'],seconds=row['seconds'],loss=row['loss'],clip=row['clip']) for row in new[-10:]],first_future_projection_update_norm_by_rank=updates,recipe_sha256=hashlib.sha256((r/'config.json').read_bytes()).hexdigest(),world_size=8,global_batch=512,all_resumed_losses_and_gradient_norms_finite=True,first_update_projection_same_all8=True,status=status)
(r/'acceleration_v4/PRODUCTION_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['latest_step','resumed_updates','warm_median_seconds','warm_min_seconds','warm_max_seconds','all_resumed_losses_and_gradient_norms_finite','first_update_projection_same_all8']},indent=2))
