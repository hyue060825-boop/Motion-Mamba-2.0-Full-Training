from pathlib import Path
import os, gc, torch
from util import ROOT,read,write,sha
from official_setup import activate,SUITE

def load(args):
 activate()
 from motion_training import runtime
 from motion_training._compat import legacy_loss
 from external_model import ExternalDenoiser
 info=read(ROOT/'CHECKPOINTS.json')[os.environ['EVAL_LABEL']]
 ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
 assert ck['step']==100000 and ck['scheduler']['completed']==100000
 assert ck['method']==os.environ['EVAL_LABEL'] and len(ck['rng_by_rank'])==4
 assert len(set(ck['rank_state_hashes']))==1
 torch.set_num_threads(2)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=runtime.load_model('g1',device=args.device,text_encoder=False,fresh_denoiser=True)
 legacy_loss._replace_stats(m,args.stats_dir)
 raw=ExternalDenoiser(ck['method'],SUITE/'repos'/('DiP' if ck['method']=='dip' else 'DART'),m.motion_rep,m.hybrid).to(args.device)
 raw.load_state_dict(ck['model'],strict=True)
 assert set(raw.state_dict())==set(ck['model'])
 assert all(torch.equal(v.detach().cpu(),ck['model'][k]) for k,v in raw.state_dict().items())
 # Public sampler names differ from the canonical training forward names.
 # These aliases do not change the network, tensors, parameters or arithmetic.
 raw.latent_embedding_dim=128
 trained_forward=raw.forward
 def evaluation_forward(*a,**kw):
  if 'timesteps' in kw:kw['t']=kw.pop('timesteps')
  if 'text_feat_pad_mask' in kw:kw['text_pad_mask']=kw.pop('text_feat_pad_mask')
  return trained_forward(*a,**kw)
 raw.forward=evaluation_forward
 m.denoiser.model=raw
 m.generation_contract=ck['config']['generation_contract'];m.training_codec_contract=ck['config']['training_codec_contract']
 assert m.generation_contract=='g1_committed_v3' and m.decoder_history_frames==52
 assert Path(args.stats_dir).resolve()==Path(ck['config']['stats_dir']).resolve()
 m.eval();m.requires_grad_(False);gc.collect();torch.cuda.empty_cache()
 write(Path(args.output).parent/'LOAD.json',dict(architecture=ck['method']+'_common_G1_adapter',global_step=ck['step'],checkpoint=args.checkpoint,
  all_loaded_denoiser_weights_verified=True,parameter_count=sum(p.numel() for p in raw.parameters()),checkpoint_source_hash=info['sha256'],
  generation_contract=m.generation_contract,maximum_history_frames=196,maximum_total_input_frames=248,decoder_history_frames=52,
  future_policy='native fixed input capacity, no time compression, waypoint or trained changes',full_future_conditioning_equivalent=False,
  weights_or_training_changed=False,inference_precision='FP32, TF32 disabled'))
 from committed_rolling import CommittedMOTION_MAMBA2
 proxy=CommittedMOTION_MAMBA2(m,Path(args.output).parent/'INFERENCE_TRACE.json')
 return proxy,proxy,ck,None
