from motion_mamba2.paths import artifact_path
import os,sys,copy,json,time
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from motion_training import runtime,engine
from motion_training.io import identity,write
from g1_padding_contract import install_padding_contract
cfg=json.loads((D/'configs/train.json').read_text())
source=artifact_path('checkpoint_step_050000_pt_697f95b41d')
original=runtime.load_model;official={}
def capture(*a,**kw):
    model=original(*a,**kw)
    raw=model.denoiser.model
    for name,value in raw.state_dict().items():
        if 'seqTransEncoder' not in name:official[name]=value.detach().float().cpu().clone()
    return model
runtime.load_model=capture
m,c,ck=runtime.load(source,True,fresh_seed=cfg['initialization_seed'],official_frontend=True)
raw=m.denoiser.model
matches=[];mismatches=[]
for name,value in raw.state_dict().items():
    if name in official and value.shape==official[name].shape:
        (matches if torch.equal(value.detach().float().cpu(),official[name]) else mismatches).append(name)
assert not mismatches,mismatches
for prefix in ['root_model.embed_text.','body_model.embed_text.','root_model.output_linear.','body_model.output_linear.']:
    assert any(k.startswith(prefix) for k in matches),prefix
ck['config']=dict(ck['config'],generation_contract='g1_committed_v3',padding_contract='g1_hybrid_attention_mask_v1',training_codec_contract='g1_history_codec_104_v1',joint_limit_spec=artifact_path('joint_limits_json_6998748a01'),content_memory_enabled=False)
ck['initialization'].update(official_denoiser_loaded=True,reuse_scope='compatible frontend/projections only; Native sequence modules freshly constructed',old_native_weights_loaded=False,matched_official_keys=matches)
ck['step']=0;ck['_framework_global_start']=0
install_padding_contract(m,ck['config']['padding_contract']);m.generation_contract=ck['config']['generation_contract']
ocfg=dict(cfg,optimizer_state_policy='fresh',initialization_policy='fresh_denoiser')
opt=runtime.optimizer(m,ck,ocfg)
out=D/'initial';out.mkdir(exist_ok=True)
placeholder=D/'INITIAL_DATA_CONTRACT.json';write(placeholder,dict(plan='streaming, prepared after initialization',sample_slots=3200000))
engine.save(out/'checkpoint0.pt',m,ck,opt,cfg,0,placeholder,None)
write(D/'INITIALIZATION_VERIFIED.json',dict(official_compatible_keys=matches,mismatches=mismatches,native_sequence_modules_new=True,old_native_weights_loaded=False,trainable_parameters=sum(p.numel() for p in raw.parameters()),initial_weights_hash=identity(raw.state_dict()),dropouts=[dict(name=n,p=x.p) for n,x in raw.named_modules() if isinstance(x,torch.nn.Dropout)],official_frontend_seed=123,normalizer_dir=ck['config']['stats_dir'],checkpoint=str(out/'checkpoint0.pt')))
print('INITIALIZATION_PASS',len(matches),'compatible tensors',flush=True)
