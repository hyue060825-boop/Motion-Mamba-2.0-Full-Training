from common import *
import torch
from motion_training.io import rows
cp=ROOT/'train/checkpoint5010.pt'
assert cp.exists()
ck=torch.load(cp,map_location='cpu',weights_only=False,mmap=True)
assert ck['step']==105010 and ck['framework']['stage_step']==5010
assert ck['framework']['scheduler']==dict(spec=dict(kind='constant',lr=1e-5),completed=5010)
assert len(ck['rng_by_rank'])==4 and len(set(ck['framework']['rank_state_hashes']))==1
assert ck['framework']['recipe_hash']==identity(read(ROOT/'config.json'))
transition=read(ROOT/'OPTIMIZER_CONTINUATION.json')
assert transition['optimizer_exact'] and transition['scheduler_exact']
audit=read(ROOT/'TRAINING_FILES_READY.json');assert audit['passed']
for name,h in audit['unchanged_protocol_source_hashes'].items():assert sha(ROOT/name)==h,name
updates=rows(ROOT/'train/train.jsonl')
assert [r['step'] for r in updates[:10]]==list(range(5001,5011))
assert all(r['global_batch']==512 and r['world_size']==4 and r['GT_slots']==448 and r['teacher_slots']==64
 and r['legacy_teacher_slots']==32 and r['future_teacher_slots']==32 for r in updates[:10])
assert read(ROOT/'CONTINUATION_REGISTRATION.json')['passed']
assert all(r['pool_generation']==3 for r in updates[:10])
write(ROOT/'STARTED_TRAINING.json',dict(time=time.time(),real_additional_updates_verified=10,
 global_step=105010,original_stage_step=5010,current_logged_step=updates[-1]['step'],target_global_step=110000,
 checkpoint=str(cp),checkpoint_sha256=sha(cp),actual_storage=str(cp.resolve()),
 Adam_states=len(ck['optimizer']['state']),full105k_Adam_FP32_masters_inherited=True,
 scheduler_continues_from5000=True,four_rank_RNG_saved=True,rank_weights_Adam_hashes_equal=True,
 sampling_continues_at5001=True,fresh_current_student_pool_generation=3,
 original_recipe_preserved=True,global_batch=512,devices=[0,1,2,3],GT=448,legacy=32,root_prepare=24,root_arrival=8,
 acceleration=read(ROOT/'gates/ACCELERATION_SELECTION.json'),quality_improvement_not_assumed=True))
print('VERIFIED_ADDITIONAL10_COMPLETE_105010')
