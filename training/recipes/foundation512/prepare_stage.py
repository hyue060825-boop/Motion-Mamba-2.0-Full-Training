import shutil, pickle, collections, hashlib, py_compile
from common import *
ROOT.mkdir(parents=True,exist_ok=True)
if not (ROOT/'code').exists():
    shutil.copytree(OLD/'code',ROOT/'code',ignore=shutil.ignore_patterns('__pycache__'))
engine_file=ROOT/'code/src/motion_training/engine.py'
original=(OLD/'code/src/motion_training/engine.py').read_text()
assert "c['batch_size'] in [10,32]" in original
original=original.replace("c['batch_size'] in [10,32]", "c['batch_size'] in [8,10,16,32,64,128,256,512]")
# Only this isolated stage gets cooperative stop requests, after a complete
# optimizer update and durable receipt. All ranks agree before saving/exiting.
needle="        if parallel.leader():write(dest/'STATE.json',dict(status='SEGMENT_COMPLETE'"
assert needle in original
stop_code="""            if cfg.get('boundary_request_path') and step%100==0:
                requests=parallel.gather(Path(cfg['boundary_request_path']).exists())
                if any(requests):
                    target=dest/f'checkpoint{step}.pt'
                    if not target.exists():save(target,m,ck,opt,cfg,step,manifest,pool)
                    if parallel.leader():write(dest/'STATE.json',dict(status='BOUNDARY_REQUEST',step=step,checkpoint=str(target),recipe_hash=recipe_hash(cfg),model_qualified=False))
                    parallel.barrier()
                    return str(target)
"""
original=original.replace(needle,stop_code+needle)
engine_file.write_text(original)
batch_file=ROOT/'code/src/motion_training/batching.py'
text=(OLD/'code/src/motion_training/batching.py').read_text()
replacements={
 "assert len(records)==cfg['batch_size']==32":"assert len(records)==cfg['batch_size']; global_batch=cfg['batch_size']; micro_size=cfg['per_rank_microbatch']",
 "assert parallel.world() in [2,4]":"assert parallel.world() in [1,2,4,8] and global_batch%parallel.world()==0",
 "width=32//parallel.world()":"width=global_batch//parallel.world(); assert width%micro_size==0",
 "microbatches=width//8":"microbatches=width//micro_size",
 "subset=local[micro*8:(micro+1)*8]":"subset=local[micro*micro_size:(micro+1)*micro_size]",
 "sample_slots=32":"sample_slots=global_batch",
 "for x in parts)/32":"for x in parts)/global_batch",
 "seconds=wall,batch_size=32,per_rank_batch=8":"seconds=wall,batch_size=global_batch,per_rank_batch=micro_size,accumulation_steps=microbatches",
}
for a,b in replacements.items():
    assert a in text,a; text=text.replace(a,b)
batch_file.write_text(text)
py_compile.compile(str(engine_file),doraise=True); py_compile.compile(str(batch_file),doraise=True)
from motion_training import data
from motion_training.io import rows,sha
import numpy as np
from motion_training._compat.fusion_index import excluded
cfg=read(OLD/'configs/train.json'); prior=read(OLD/'train/data/MANIFEST.json')
records=rows(cfg['data_index']); ex=read(cfg['exclusions']); dev=set(prior['dev_families'])
captions=set(read(OLD/'train/data/CAPTIONS.json')); groups={}; coverage=[]; missing=set(); all_texts=set()
available=set()
for folder in [str(ROOT/'text')]+cfg['text_roots']:
    if Path(folder).exists():available.update(p.stem for p in Path(folder).glob('*.npy'))
for source in ['bones','humanml3d','hiphi']:
    for h,f in data.BUCKETS:
        by=collections.defaultdict(list); eligible=covered=options=kept=0
        for r in records:
            if r['dataset']!=source or r['project_split']!='train' or excluded(r,ex) or r['source_family_id'] in dev: continue
            all_opts=data.intervals(r,h+52+f)
            if not all_opts:continue
            eligible+=1; options+=len(all_opts)
            valid=all_opts
            all_texts.update(v[2] for v in all_opts)
            missing.update(v[2] for v in all_opts if data.text_key(v[2]) not in available)
            if valid:by[r['source_family_id']].append((r,valid));covered+=1;kept+=len(valid)
        assert by,(source,h,f)
        # Do not silently narrow the dataset to fit an old text cache.
        assert covered==eligible and kept==options
        groups[source,h,f]=(sorted(by),dict(by))
        coverage.append(dict(source=source,H=h,F=f,families=len(by),records=covered,caption_intervals=kept))
with open(ROOT/'GROUPS.pkl.tmp','wb') as fp:pickle.dump(groups,fp,protocol=5)
(ROOT/'GROUPS.pkl.tmp').replace(ROOT/'GROUPS.pkl')
# The BONES bulk cache was already verified against the same official encoder.
# Recheck actual arrays before extending its use to newly drawn captions.
reuse=read(OLD/'TEXT_CACHE_REUSE.json'); bulk=Path(reuse['source'])
assert reuse['compatible'] and all(r['equal'] for r in reuse['verified'])
ix=read(bulk/'index.json'); features=np.load(bulk/'embeddings.npy',mmap_mode='r'); by={r['caption']:r['row'] for r in ix['samples']}
checked=0
for text in sorted(captions & set(by)):
    key=data.text_key(text)
    existing=next((Path(folder)/(key+'.npy') for folder in cfg['text_roots'] if (Path(folder)/(key+'.npy')).exists()),None)
    if existing:
        assert np.array_equal(np.load(existing).reshape(-1),np.array(features[by[text]],dtype=np.float32).reshape(-1))
        checked+=1
    if checked==128:break
assert checked==128
(ROOT/'text').mkdir(exist_ok=True);copied=0
for text in sorted(missing & set(by)):
    a=np.array(features[by[text]],dtype=np.float32);assert a.size==4096 and np.isfinite(a).all()
    p=ROOT/'text'/(data.text_key(text)+'.npy');tmp=p.with_suffix('.tmp.npy');np.save(tmp,a);tmp.replace(p);available.add(data.text_key(text));copied+=1
remaining=sorted(t for t in all_texts if data.text_key(t) not in available)
write(ROOT/'TEXT_REQUIRED.json',sorted(all_texts));write(ROOT/'TEXT_MISSING.json',remaining)
write(ROOT/'TEXT_REUSE.json',dict(bulk_source=str(bulk),checked_arrays=checked,arrays_bitwise_equal=True,copied=copied,still_missing=len(remaining)))
cfg['text_roots']=[str(ROOT/'text')]+cfg['text_roots']
cfg.update(parent=str(OLD/'train/checkpoint75000.pt'),name='fusion_75k_plus_batch512',max_updates=25000,
    batch_size=512,gt_slots=512,per_rank_microbatch=8,seed=202610070512,
    savepoints=[0,1,10,100,1000,5000,10000,15000,20000,25000],audit_every=1000,
    initialization_policy='carry_state_new_stage',optimizer_state_policy='inherit',
    budget_root=str(ROOT/'budget'),lazy_manifest_template=str(ROOT/'MANIFEST_TEMPLATE.json'),
    execution_authorization='USER_AFTER75K_LARGEST_STABLE_MICROBATCH_GLOBAL512_20261007',
    batching='equal DDP mean over ranks and microbatches; global512; max admitted microbatch',
    resource_policy='never stop foreign jobs; four ranks until eight available and same-batch topology check passes',
    boundary_request_path=str(ROOT/'BOUNDARY_REQUEST.json'))
write(ROOT/'config.json',cfg)
manifest=dict(format='lazy_family_crops_v1',schema=prior['schema'],recipe=cfg['name'],batch_size=512,
    seed=cfg['seed'],planned_exposures=25000*512,groups_path=str(ROOT/'GROUPS.pkl'),groups_sha256=sha(ROOT/'GROUPS.pkl'),
    index_sha256=sha(cfg['data_index']),exclusions_sha256=sha(cfg['exclusions']),
    additional_dev=prior['additional_dev'],dev_families=prior['dev_families'],
    text_caption_sha256=sha(OLD/'train/data/CAPTIONS.json'),coverage=coverage,
    source_ratios=cfg['source_ratios'],HGF=cfg['HGF'],conditions=[5,4,4,4,4,4],
    exact_text_cache_reused=True,text_missing=len(remaining),fresh_random_crops=True,fresh_noise_per_exposure=True,
    future_predictions_supervised=False,source_family_uniform=True,batch_layout_shared=True,
    previous_exposures=75000*32,cumulative_planned_exposures=75000*32+25000*512)
write(ROOT/'MANIFEST_TEMPLATE.json',manifest)
write(ROOT/'budget/LIMITS.json',dict(caps=dict(B=25000,checks=200,C=0,teacher=0),total_updates=25200,gpu_hours=480))
write(ROOT/'probe_budget/LIMITS.json',dict(caps=dict(B=0,checks=200,C=0,teacher=0),total_updates=200,gpu_hours=4))
hashes={str(p):sha(p) for p in sorted((ROOT/'code').rglob('*')) if p.is_file() and '__pycache__' not in str(p)}
write(ROOT/'SOURCE_HASHES.json',hashes)
write(ROOT/'PREPARED.json',dict(complete=not remaining,coverage=coverage,missing_captions=len(remaining),
    changed_training_code=[str(engine_file),str(batch_file)],original_source_untouched=True,
    loss_precision_codec_text_mappings_unchanged=True,scheduler_inherited=True,
    exposure_change_is_recipe_change_not_equivalence_claim=True))
print('PREPARED_STAGE',manifest['cumulative_planned_exposures'],'MISSING_TEXT',len(remaining),flush=True)
