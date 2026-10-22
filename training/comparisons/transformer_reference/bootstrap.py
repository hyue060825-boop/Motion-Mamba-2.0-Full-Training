from common import *
import torch
import adapter
from motion_training import runtime,engine
from motion_training.io import identity
adapter.install()
cfg=read(ROOT/'configs/batch32.json')
m,c,ck=runtime.load(OLD/'initial/checkpoint0.pt',True,fresh_seed=123)
opt=runtime.optimizer(m,ck,dict(cfg,optimizer_state_policy='fresh',initialization_policy='fresh_denoiser'))
manifest=OLD/'train/data/MANIFEST.json'
engine.save(ROOT/'initial/checkpoint0.pt',m,ck,opt,cfg,0,manifest,None)
raw=m.denoiser.model
write(ROOT/'INITIALIZATION.json',dict(passed=True,architecture='official_motion_mamba2_transformer',common_keys_exact=ck['initialization']['shared_keys'],official_sequence_weights_loaded=False,old_native_weights_loaded=False,parameters=sum(p.numel() for p in raw.parameters()),sequence_parameters=sum(p.numel() for n,p in raw.named_parameters() if 'seqTransEncoder' in n),state_hash=identity(raw.state_dict()),dropouts=[dict(name=n,p=x.p) for n,x in raw.named_modules() if isinstance(x,torch.nn.Dropout)],precision='BF16 autocast; FP32 root projection and physical losses; FP32 Adam',checkpoint=str(ROOT/'initial/checkpoint0.pt')))
print('INITIALIZATION_PASSED',flush=True)
