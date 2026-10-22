from motion_mamba2.paths import artifact_path
import os,sys,json,argparse,copy,time
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from motion_training import evaluation,acceleration,runtime
from motion_training.io import read,write,sha,identity
p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();folder=D/'evaluation'/str(a.step);folder.mkdir(parents=True,exist_ok=True)
# Reuse the exact prior Old/New evaluation panel and metrics, only candidate path changes.
cfg=read(artifact_path('old_json_9aee84d268'));cfg.update(parent=str(D/'train'/f'checkpoint{a.step}.pt'),budget_root=str(D/'budget'),text_roots=read(D/'configs/train.json')['text_roots'])
cfg['kernel_gate_path']=str(folder/'kernel/GATE.json');path=folder/'config.json';write(path,cfg)

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

try:acceleration.gate(str(path),str(folder/'kernel'))
except AssertionError as error:
 write(folder/'KERNEL_FALLBACK.json',dict(reason=repr(error),evaluation_backend='reference',training_changed=False))
for split in (['DEV','FINAL'] if a.step in [50000,100000] else ['DEV']):
 evaluation_phase='evaluate_'+split
 c=copy.deepcopy(cfg);c['evaluation_split']=split;c['evaluation_source_indices']=[10,11] if split=='FINAL' else [3,4];cf=folder/f'config_{split}.json';write(cf,c);r=evaluation.evaluate(cfg['parent'],str(folder/split),str(cf));print('EVAL_COMPLETE',a.step,split,r['foundation']['function'],r['foundation']['joint'],flush=True)
write(folder/'COMPLETE.json',dict(complete=True,checkpoint=cfg['parent'],checkpoint_sha256=sha(cfg['parent']),candidate_stage_step=a.step,old_reference=artifact_path('old_c1509953a8'),common_panel=True,physical_simulation_verified=False,do_not_deploy_without_actual_simulation=True))
