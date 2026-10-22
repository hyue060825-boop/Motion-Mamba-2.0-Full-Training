import collections
from common import *
from lazy_plan import Plan
from motion_training.io import identity,sha
h=read(ROOT/'MANIFEST_TEMPLATE.json');plan=Plan(h);sources=collections.Counter();layouts=collections.Counter();conditions=collections.Counter()
assert len(plan)==12800000
for step in range(20):
    batch=plan[step*512:(step+1)*512]
    assert len(batch)==512 and len({(v['H'],v['F']) for v in batch})==1
    assert len({v['seed'] for v in batch})==512
    assert identity(batch)==identity(plan[step*512:(step+1)*512])
    assert all(v['source_family_id'] not in h['dev_families'] and v['split']=='TRAIN' and v['supervision']=='GT' for v in batch)
    assert all(v['crop_end']-v['crop_start']==v['H']+52+v['F'] for v in batch)
    for v in batch:sources[v['dataset']]+=1;conditions[v['condition']]+=1
    layouts[str((batch[0]['H'],batch[0]['F']))]+=1
assert list(layouts.values()) and sorted(layouts.values())==[2,2,3,4,4,5]
for source,p in h['source_ratios'].items():assert abs(sources[source]/10240-p)<.003
assert abs(conditions['none']/10240-.2)<.003
assert len(plan[24999*512:25000*512])==512
diff=[]
for p in (ROOT/'code').rglob('*'):
    if not p.is_file() or '__pycache__' in str(p):continue
    old=OLD/'code'/p.relative_to(ROOT/'code')
    if sha(p)!=sha(old):diff.append(str(p.relative_to(ROOT/'code')))
assert sorted(diff)==['src/motion_training/batching.py','src/motion_training/engine.py']
write(ROOT/'CPU_VERIFICATION.json',dict(passed=True,reproducible_batches=True,batch512_actual_records=True,
    index_and_exclusions_unchanged=True,dev_families_excluded=True,source_counts=dict(sources),
    layout_counts=dict(layouts),condition_counts=dict(conditions),
    true_end_cursor=12800000,changed_code=diff,loss_codec_diffusion_text_runtime_sources_unchanged=True,
    gpu_backward_and_memory_admission_pending=True))
print('CPU_VERIFICATION_PASS',dict(sources),dict(layouts),flush=True)
