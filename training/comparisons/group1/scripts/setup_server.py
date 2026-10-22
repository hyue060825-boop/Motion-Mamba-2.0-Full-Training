from motion_mamba2.paths import artifact_path
from pathlib import Path
import shutil,json,hashlib,time
import torch
S=Path(artifact_path('g1_external_matched100k_20261019_f77433d79c'))
A=Path(artifact_path('g1_motion_mamba2_four_gpu_matched100k_20261009_a4a8fc8513'))
N=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
def read(p):return json.loads(Path(p).read_text())
def write(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
S.mkdir(parents=True,exist_ok=True)
if not (S/'code').exists():shutil.copytree(A/'code',S/'code',symlinks=False,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for n in ['pipeline_accel.py','codec_graph.py','lazy_plan.py']:
 shutil.copy2(A/n,S/'scripts'/n)
initial=torch.load(N/'initial/checkpoint0.pt',map_location='cpu',weights_only=False,mmap=True)['config']
reference32=read(A/'configs/batch32.json');reference512=read(A/'configs/batch512.json')
fields=['scheduler','betas','eps','weight_decay','clip','loss','source_ratios','HGF','text_dropout','constraint_dropout','empty_spatial','spatial_families','extensions','precision']
for method,name,cards in [('dip','DiP_G1',[0,1,2,3]),('dart','DART_MLP_G1',[4,5,6,7])]:
 R=S/name;R.mkdir(exist_ok=True)
 assert not (R/'STARTED_TRAINING.json').exists(),'Do not rerun setup over training'
 for folder in ['gates','train','initial','logs','configs']:(R/folder).mkdir(exist_ok=True)
 for batch,src in [(32,reference32),(512,reference512)]:
  cfg=dict(src);cfg.update(name=name+'_matched100k',max_updates=100000,batch_size=batch,gt_slots=batch,
    per_rank_microbatch=8 if batch==32 else 64,optimizer_state_policy='fresh' if batch==32 else 'inherit',
    parent='fresh_external_denoiser_seed123',stats_dir=initial['stats_dir'],training_codec_contract=initial['training_codec_contract'],
    generation_contract=initial['generation_contract'],text_roots=reference512['text_roots'],
    savepoints=[10,100,1000,10000,20000,30000,40000,50000,60000,70000,75000,80000,90000,100000])
  assert all(cfg[k]==src[k] for k in fields)
  write(R/f'configs/batch{batch}.json',cfg)
 plan=dict(method=method,name=name,variant='common_G1_codec_conditioning_adapter_v1',cards=cards,world_size=4,
   schedule=[[0,75000,32],[75000,100000,512]],sample_slots=15200000,initialization_seed=123,
   frozen_shared_assets=['G1 ARDY codec','G1 stats','cached LLM2Vec4096 original captions'],
   official_denoiser_weights_loaded=False,Native_weights_loaded=False,
   architecture='official DiP TransformerDecoder512x8' if method=='dip' else 'official DART DenoiserMLP512 residual2',
   official_source_pin='ef8edce6a53c6ab19e53b4d4dcf15bc0bc60a778' if method=='dip' else 'bb67ae1ed6ce051080468bf15bc6e54a6c3f8417',
   adaptations=['common G1 hybrid148 representation','common cached4096 text','common spatial observations/masks supplied through input dimensions',
     'currentG52 supervision and original four losses','common FP32 output/geometry, BF16 AMP, FP32 Adam',
     'native internal dropout retained; external text/condition dropout applied once',
     'DART dedicated primitive VAE replaced by common frozen G1 codec: architecture control, not full DartControl' if method=='dart' else 'DiP SMPL converter not instantiated; common G1 codec used'],
   unsupported_claims=['not official full-system reproduction','no newly claimed native40s-deadline or delayed-memory task support','no simulation deployment'],
   wall_budget_hours=96,gpu_budget_hours=384,checkpoint_policy='retain all this-run saved checkpoints; no historical checkpoints removed',
   quality_policy='complete frozen budget despite poor scores; stop only execution/numerical/resource/budget/user boundary',
   authorization='2026-10-19 user requests first two priority baselines, four GPU each,75k global32+25k global512; previously approved disclosed common-representation training adapters')
 write(R/'PLAN.json',plan)
 evidence=dict(passed=True,matched_fields=fields,reference_batch32=str(A/'configs/batch32.json'),reference_batch512=str(A/'configs/batch512.json'),
   shared_plan32=str(N/'train/data/MANIFEST.json'),shared_plan512=str(N/'batch512_after75k_20261007/MANIFEST_TEMPLATE.json'),
   exact_sampler_reference_reuse=True,global_batch_schedule=plan['schedule'],sample_slots=15200000,
   model_specific_adapter_disclosed=True,quality_results_not_admission_criteria=True)
 write(R/'gates/PROTOCOL.json',evidence)
 hashes={str(p.relative_to(S)):sha(p) for top in ['scripts','code'] for p in (S/top).rglob('*.py')}
 hashes.update({str(p.relative_to(S)):sha(p) for p in (R/'configs').glob('*.json')})
 for repo,file in [('DiP','model/mdm.py'),('DART','model/mld_denoiser.py')]:hashes[f'repos/{repo}/{file}']=sha(S/'repos'/repo/file)
 write(R/'SOURCE_HASHES.json',hashes);write(R/'STATUS.json',dict(stage='PREPARING_GPU_ADMISSION',step=0,time=time.time()))
write(S/'QUEUE.json',dict(first_batch=['DiP_G1','DART_MLP_G1'],remaining_batches=[['CMDM','FloodDiffusion2'],['OMG300M','ActionPlan'],['COMET','MotionStreamer']],
 remaining_status='AUTHORIZED_QUEUE_NOT_YET_ADAPTED_OR_LAUNCHED',not_all_baselines_started=True))
print(S)
