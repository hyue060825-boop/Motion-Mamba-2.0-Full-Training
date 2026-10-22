from pathlib import Path
import os,gc,torch
from util import ROOT,read,write,sha
from official_setup import activate,SUITE

def load(args):
 activate()
 from motion_training import runtime
 from motion_training._compat import legacy_loss
 from external_model import ExternalDenoiser
 from committed_rolling import CommittedMOTION_MAMBA2
 method=os.environ['EVAL_LABEL'];info=read(ROOT/'CHECKPOINTS.json')[method]
 ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
 assert ck['method']==method and ck['step']==ck['scheduler']['completed']==100000
 assert ck['exposures']==15200000 and len(ck['rng_by_rank'])==4 and len(set(ck['rank_state_hashes']))==1
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 m=runtime.load_model('g1',device=args.device,text_encoder=False,fresh_denoiser=True)
 legacy_loss._replace_stats(m,args.stats_dir)
 raw=ExternalDenoiser(method,SUITE/'repos'/('FloodDiffusion2' if method=='fd2' else 'OMG'),m.motion_rep,m.hybrid).to(args.device)
 raw.load_state_dict(ck['model'],strict=True)
 assert set(raw.state_dict())==set(ck['model'])
 assert all(torch.equal(v.detach().cpu(),ck['model'][k]) for k,v in raw.state_dict().items())
 m.denoiser.model=raw;m.generation_contract=ck['config']['generation_contract'];m.training_codec_contract=ck['config']['training_codec_contract']
 assert m.generation_contract=='g1_committed_v3' and m.decoder_history_frames==52
 assert Path(args.stats_dir).resolve()==Path(ck['config']['stats_dir']).resolve()
 m.eval();m.requires_grad_(False);gc.collect();torch.cuda.empty_cache()
 write(Path(args.output).parent/'LOAD.json',dict(architecture=method+'_common_G1_variable_length_adapter',
  global_step=ck['step'],checkpoint=args.checkpoint,checkpoint_source_hash=info['sha256'],all_loaded_denoiser_weights_verified=True,
  parameter_count=sum(p.numel() for p in raw.parameters()),
  generation_contract=m.generation_contract,maximum_history_frames=196,decoder_history_frames=52,
  full_future_conditioning_equivalent=True,future_policy='complete sparse future supplied at every generation window',
  weights_or_training_changed=False,inference_precision='FP32, TF32 disabled; trained FP32 geometry boundary retained',
  adaptation_scope=raw.source_edits,core_source=raw.source,core_source_sha256=raw.source_sha256,
  model_parameter_dtypes={str(dtype):sum(p.numel() for p in m.parameters() if p.dtype==dtype) for dtype in {p.dtype for p in m.parameters()}},
  numeric_eligibility='weight and input checks only; root/control health separately measured'))
 proxy=CommittedMOTION_MAMBA2(m,Path(args.output).parent/'INFERENCE_TRACE.json')
 return proxy,proxy,ck,None
