"""Summarize actual admitted continuation updates, never probe timings."""
import statistics
from common import *
rows=[]
with (ROOT/'train/train.jsonl').open() as handle:
    for line in handle:
        r=json.loads(line)
        if r.get('per_rank_batch')==64 and r.get('step',0)>101:
            rows.append({k:r[k] for k in ('step','seconds','world_size','per_rank_batch','accumulation_steps')})
assert len(rows)>=10, f'Only {len(rows)} warmed real micro64 updates available'
ranks=rows[-1]['world_size']
recent=[r for r in rows if r['world_size']==ranks][-20:]
mean=statistics.mean(r['seconds'] for r in recent)
report=dict(global_batch=512,micro=64,ranks=ranks,accumulation_steps=512//ranks//64,
    observations=len(recent),global_step_first=75000+recent[0]['step'],global_step_last=75000+recent[-1]['step'],
    mean_seconds=mean,median_seconds=statistics.median(r['seconds'] for r in recent),
    samples_per_second=512/mean,remaining_training_hours=(25000-recent[-1]['step'])*mean/3600,
    projected_remaining_gpu_hours=(25000-recent[-1]['step'])*mean*ranks/3600,
    excludes_checkpoint_eval_and_startup=True,probe_timings_not_used=True)
if (ROOT/'ACTUAL_SPEED.json').exists() and not (ROOT/'micro64_switch/MICRO16_ACTUAL_SPEED.json').exists():
    write(ROOT/'micro64_switch/MICRO16_ACTUAL_SPEED.json',read(ROOT/'ACTUAL_SPEED.json'))
write(ROOT/'ACTUAL_SPEED.json',report)
print(json.dumps(report,ensure_ascii=False,indent=2))
