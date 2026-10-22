"""Reconstruct the original initialization; performs no optimizer update."""
from pathlib import Path
import argparse,json,sys,tempfile,os
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'training'),str(R/'data')]
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--metadata',default=str(R/'configs/initialization/metadata.json'));a=p.parse_args()
import torch
from motion_mamba2.legacy_compat import checkpoint_metadata
from motion_training import runtime,engine
from motion_training.io import write,identity
from g1_padding_contract import install_padding_contract
cfg=json.loads((R/'configs/training/foundation32.json').read_text())
from motion_mamba2.paths import resolve_tree
cfg=resolve_tree(cfg);metadata=checkpoint_metadata(json.loads(Path(a.metadata).read_text()));metadata.update(step=0)
out=Path(a.output);assert not out.exists(),'Never overwrite a saved initialization';out.parent.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'metadata.pt';torch.save(metadata,path)
 m,c,ck=runtime.load(path,True,device=a.device,fresh_seed=123,official_frontend=True)
 ck['step']=0;ck['_framework_global_start']=0
 ck['config']=dict(ck['config'],generation_contract='g1_committed_v3',padding_contract='g1_hybrid_attention_mask_v1',training_codec_contract='g1_history_codec_104_v1',content_memory_enabled=False)
 ck['initialization'].update(official_denoiser_loaded=True,reuse_scope='compatible pretrained frontend/projections; new native sequence modules',old_native_weights_loaded=False)
 install_padding_contract(m,ck['config']['padding_contract']);m.generation_contract='g1_committed_v3'
 opt=runtime.optimizer(m,ck,dict(cfg,optimizer_state_policy='fresh',initialization_policy='fresh_denoiser'))
 contract=out.parent/'INITIAL_DATA_CONTRACT.json';write(contract,dict(plan='prepare from the authorized training index',sample_slots=3200000))
 engine.save(out,m,ck,opt,cfg,0,contract,None)
 write(out.parent/'INITIALIZATION.json',dict(step=0,Adam_empty=not opt.state,seed=123,model_tensor_hash=identity(m.denoiser.model.state_dict()),official_compatible_frontend_reused=True,sequence_modules_new=True,optimizer_updates=0))
print('INITIALIZATION_SAVED',out)
