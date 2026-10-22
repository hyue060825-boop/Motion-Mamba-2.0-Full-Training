from motion_mamba2.paths import artifact_path
from pathlib import Path
import hashlib,json
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
p=D/'evaluate.py';before=p.read_text();backup=p.with_name('evaluate.before_auto50k.py')
assert not backup.exists(), 'Do not apply twice'
assert "for split in (['DEV','FINAL'] if a.step==100000 else ['DEV']):" in before
after=before.replace('from motion_training import evaluation,acceleration','from motion_training import evaluation,acceleration,runtime')
after=after.replace('from motion_training.io import read,write,sha','from motion_training.io import read,write,sha,identity')
after=after.replace("p.add_argument('--step',type=int,required=True);a=p.parse_args()", "p.add_argument('--step',type=int,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()")
load_receipt='''
if a.prepare_only:
 print('PREPARED',a.step,path,flush=True)
 raise SystemExit(0)
# The previous Old/New50k evaluation used these same in-memory contracts.
from g1_padding_contract import install_padding_contract
original_load=runtime.load
evaluation_phase='gate'
def common_load(*args,**kwargs):
 m,c,ck=original_load(*args,**kwargs)
 previous=dict(ck['config']);ck=dict(ck,config=dict(previous))
 ck['config']['joint_limit_spec']=cfg['common_joint_limit_spec']
 install_padding_contract(m,'g1_hybrid_attention_mask_v1')
 m.generation_contract='g1_committed_v3'
 ck['config'].update(padding_contract='g1_hybrid_attention_mask_v1',generation_contract='g1_committed_v3')
 raw=m.denoiser.model
 receipt=dict(checkpoint=cfg['parent'],checkpoint_sha256=sha(cfg['parent']),stored_step=ck.get('step'),
  original_config=previous,effective_config=ck['config'],overrides={k:dict(before=previous.get(k),after=v) for k,v in ck['config'].items() if previous.get(k)!=v},
  weights_hash=identity(raw.state_dict()),parameters=sum(p.numel() for p in raw.parameters()),
  root_stats_hash=identity([m.motion_rep.global_root_stats.mean,m.motion_rep.global_root_stats.std_eps]),
  codec_hash=identity(m.autoencoder.state_dict()),decoder_history_frames=m.decoder_history_frames,
  generation_frames=m.gen_horizon_len,inference_conditions=dict(cfg=[2.,2.],denoising_steps=10,fps=25),
  sequence_config=dict(mode='hybrid',window_size_chunks=4,num_sink_tokens=4),
  overrides_scope='same in-memory inference/measurement metadata as original Old/New50k evaluator; weights and training unchanged')
 assert receipt['stored_step']==a.step,(receipt['stored_step'],a.step)
 write(folder/('LOAD_'+evaluation_phase+'.json'),receipt)
 if evaluation_phase.startswith('evaluate'):write(folder/'LOAD_evaluate.json',receipt)
 print('LOAD',a.step,evaluation_phase,receipt['parameters'],receipt['overrides'],flush=True)
 return m,c,ck
runtime.load=common_load
'''
after=after.replace("try:acceleration.gate(str(path),str(folder/'kernel'))", load_receipt+"\ntry:acceleration.gate(str(path),str(folder/'kernel'))")
after=after.replace("for split in (['DEV','FINAL'] if a.step==100000 else ['DEV']):", "for split in (['DEV','FINAL'] if a.step in [50000,100000] else ['DEV']):\n evaluation_phase='evaluate_'+split")
compile(after,str(p),'exec')
backup.write_text(before)
tmp=p.with_suffix('.tmp');tmp.write_text(after);tmp.replace(p)
write=lambda name,obj:(D/'auto50k'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('PATCH_RECEIPT.json',dict(file=str(p),backup=str(backup),before_sha256=hashlib.sha256(before.encode()).hexdigest(),after_sha256=hashlib.sha256(after.encode()).hexdigest(),change='Add FINAL at50k and common load receipts; no metric/training edits'))
print('PATCHED',p)
