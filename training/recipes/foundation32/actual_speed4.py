from motion_mamba2.paths import artifact_path
import json, statistics, time
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
recent=[]
for line in (D/'train/train.jsonl').open():
    try: r=json.loads(line)
    except json.JSONDecodeError: continue
    if r['step']>1000:
        recent.append(r)
assert recent and all(r['world_size']==4 and r['batch_size']==32 and r['per_rank_batch']==8 for r in recent)
steady=[r for r in recent if r['step']>=1011]
assert len(steady)>=50, len(steady)
sample=steady[-100:]
times=[r['seconds'] for r in sample]
mean=statistics.mean(times); median=statistics.median(times)
result=dict(unix_time=time.time(),last_step=recent[-1]['step'],completed_four_rank_updates=len(recent),sample_steps=[sample[0]['step'],sample[-1]['step']],n=len(sample),seconds_mean=mean,seconds_median=median,seconds_min=min(times),seconds_max=max(times),physical_devices=[4,5,6,7],global_batch=32,per_rank_batch=8,accumulation=1,remaining_50k_training_hours=(50000-recent[-1]['step'])*mean/3600,remaining_100k_training_hours=(100000-recent[-1]['step'])*mean/3600,scope='Complete recorded optimizer update; excludes checkpoint save and stage evaluation wall time',topology_gate='topology4/GATE.json',replayed_unsaved_updates=127)
(D/'topology4/ACTUAL_SPEED.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
