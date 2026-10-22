from pathlib import Path
import torch
from util import ROOT,write,sha
from official_setup import activate,TRAIN
def load(args):
 m,c,ck=activate()(args.checkpoint,False,device=args.device)
 assert ck['step']==100000 and sha(Path(args.checkpoint))=='723e251a145461d379fc69f315f976431667fc2d8092c9d4f5f9f35c5817dd98'
 raw=m.denoiser.model
 for key in ['root_model','body_model']:
  state=getattr(raw,key).state_dict();assert set(state)==set(ck[key])
  assert all(torch.equal(v.detach().cpu(),ck[key][k].to(v.dtype).cpu()) for k,v in state.items())
 assert all(torch.equal(raw.state_dict()[k].detach().cpu(),v.to(raw.state_dict()[k].dtype).cpu()) for k,v in ck['denoiser_aux'].items())
 assert m.generation_contract=='g1_committed_v3' and m.decoder_history_frames==52
 assert str(Path(args.stats_dir).resolve())==str(Path(ck['config']['stats_dir']).resolve())
 write(Path(args.output).parent/'LOAD.json',dict(architecture='retrained_ardy100k_transformer',global_step=100000,
  checkpoint=args.checkpoint,all_loaded_denoiser_weights_verified=True,parameter_count=sum(p.numel() for p in raw.parameters()),
  actual_call='CommittedARDY callable, explicit cropped hybrid history and committed52-frame decoder',training_contract=ck['config'],initialization=ck['initialization'],
  checkpoint_source_hash=sha(Path(args.checkpoint)),maximum_history_frames=196,decoder_history_frames=52,weights_or_training_changed=False))
 from committed_rolling import CommittedMOTION_MAMBA2
 proxy=CommittedMOTION_MAMBA2(m,Path(args.output).parent/'INFERENCE_TRACE.json')
 return proxy,proxy,ck,None
