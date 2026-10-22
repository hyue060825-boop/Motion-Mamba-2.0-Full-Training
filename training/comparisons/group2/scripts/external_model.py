"""Pinned FD2-Wan / OMG300M denoisers under the common G1 recipe.

Architecture adaptations, not full official-system reproductions. Retain FD2
partial-causal clean history and OMG RoPE/QK-normalized transformer. Use the
unchanged common x0/noise/four-loss protocol rather than FD2's own triangular
flow schedule/quadratic loss or OMG's original datasets/multimodal losses.
"""
from pathlib import Path
import ast,math,hashlib,importlib.util
import torch
from torch import nn
import torch.nn.functional as F

class FP32Linear(nn.Linear):
 def forward(self,x):
  with torch.autocast(device_type=x.device.type,enabled=False):return F.linear(x.float(),self.weight,self.bias)

def classes(path,env,wan=False):
 tree=ast.parse(Path(path).read_text());body=[]
 for n in tree.body:
  if not isinstance(n,(ast.ClassDef,ast.FunctionDef)):continue
  if wan and isinstance(n,ast.ClassDef) and n.name=='WanModel':
   n.bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())]
   for f in n.body:
    if isinstance(f,ast.FunctionDef):f.decorator_list=[d for d in f.decorator_list if not isinstance(d,ast.Name) or d.id!='register_to_config']
  body.append(n)
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),env)
 return env

class ExternalDenoiser(nn.Module):
 def __init__(self,method,repo,motion_rep,hybrid):
  super().__init__();self.method=method
  object.__setattr__(self,'motion_rep',motion_rep);object.__setattr__(self,'hybrid',hybrid)
  self.nframe_root_dim=20;self.latent_embedding_dim=128
  self.training_precision_contract='g1_root_output_geometry_fp32_v1'
  self.source_edits=[];self.inactive_prefixes=[]
  env=dict(torch=torch,nn=nn,F=F,math=math,BaseMotionDenoiser=nn.Module)
  if method=='fd2':
   source=Path(repo)/'models/tools/wan_model.py'
   att=Path(repo)/'models/tools/attention.py';spec=importlib.util.spec_from_file_location('pinned_fd2_attention',att);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
   env['attention']=mod.attention;env=classes(source,env,wan=True)
   self.core=env['WanModel'](patch_size=(1,1,1),text_len=1,text_dim=4096,cross_attn_norm=True,cross_rope=False,
    in_dim=3465,dim=1024,ffn_dim=2048,freq_dim=256,out_dim=148,num_heads=8,num_layers=8,
    window_size=(-1,-1),qk_norm=True,eps=1e-6,causal=False,rope_channel_split=(1,0,0))
   self.source_edits=['omit diffusers save/config mixins only; nn.Module retains exact official compute functions',
    'common G1 input/output/text widths; official partial mask retains causal immutable history',
    'common Native x0 noise and four losses replace FD2 flow/training schedule; FD2-Wan architecture adaptation']
   final=self.core.head.head
  elif method=='omg':
   source=Path(repo)/'src/omg/generation/denoisers/transformer.py';env=classes(source,env)
   self.core=env['MotionTransformerDenoiser'](input_dim=3465,hidden_dim=1152,num_layers=14,num_heads=18,
    mlp_ratio=4.,text_dim=4096,dropout=.1,rope_base=10000.,frame_cond_injection='per_layer_film',
    audio_local_attn_window=8,human_motion_local_attn_window=2,use_human_motion_local_attn=False,
    self_attention_qk_norm=True,cross_attention_qk_norm=True)
   self.core.output=nn.Linear(1152,148);final=self.core.output
   self.inactive_prefixes=['core.extra_proj.']+['core.layers.'+str(i)+'.'+n+'.' for i in range(14) for n in ['audio_film','human_motion_film']]
   self.source_edits=['common G1 input/output width and cached4096 text replace original representation/text frontend',
    'original300M hidden1152/layers14/heads18/FF4/RoPE/QK-normalization retained',
    'audio and human-reference FiLM modules retained but inactive/frozen in text+sparse-spatial-only task; no missing-modal zero fake supervision']
  else:raise ValueError(method)
  replacement=FP32Linear(final.in_features,final.out_features);replacement.load_state_dict(final.state_dict())
  if method=='fd2':self.core.head.head=replacement
  else:self.core.output=replacement
  self.source=str(source);self.source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
  self.apply_trainable_policy()
  # Admitted per-model exact arithmetic reuse; protocol and weights unchanged.
  import os,json
  run_root=os.environ.get('BASELINE_RUN_ROOT')
  receipt=Path(run_root)/'gates/CORE_ACCELERATION.json' if run_root else None
  if receipt is not None and receipt.exists():
   admission=json.loads(receipt.read_text())
   if admission.get('admit',False):
    from core_acceleration import install
    install(self)(True)

 def apply_trainable_policy(self):
  for n,p in self.named_parameters():p.requires_grad_(not any(n.startswith(k) for k in self.inactive_prefixes))

 def prepared(self,x,history_len,generation_len,future_len,history_mask,generation_mask,future_mask,
  history_token_mask,generation_token_mask,future_token_mask,text_feat,text_pad_mask,t,first_heading_angle,
  motion_mask=None,observed_motion=None):
  b,n,d=x.shape;assert d==148 and bool((generation_len==52).all())
  assert motion_mask.shape==observed_motion.shape==(b,n*4,414)
  mask=motion_mask.float();obs=observed_motion.float()*mask
  role=torch.stack([history_token_mask,generation_token_mask,future_token_mask],-1).float()
  valid=role.bool().any(-1)
  angle=torch.stack([first_heading_angle.sin(),first_heading_angle.cos()],-1)
  a=torch.cat([x.float(),torch.cat([obs,mask],-1).reshape(b,n,3312),role,angle[:,None].expand(-1,n,-1)],-1)
  text=text_feat.float()[:,0]*text_pad_mask[:,0,None]
  return a,text,valid

 def forward(self,*args,**kwargs):
  bound=dict(zip(['x','history_len','generation_len','future_len','history_mask','generation_mask','future_mask',
   'history_token_mask','generation_token_mask','future_token_mask','text_feat','text_pad_mask','t','first_heading_angle',
   'motion_mask','observed_motion'],args));bound.update(kwargs)
  if 'timesteps' in bound:bound['t']=bound.pop('timesteps')
  if 'text_feat_pad_mask' in bound:bound['text_pad_mask']=bound.pop('text_feat_pad_mask')
  a,text,valid=self.prepared(**bound);b,n,_=a.shape
  if self.method=='omg':
   conditions=dict(text_context=text[:,None],text_mask=bound['text_pad_mask'].bool())
   return self.core(a,bound['t'],conditions,valid_mask=valid)
  assert n<=1024,'Official FD2 RoPE supports at least4096 physical frames, no time compression'
  # Same official partial attention: causal finalized H, bidirectional active
  # G/observed-future, active queries can see all permitted history.
  idx=torch.arange(n,device=a.device);hm=bound['history_token_mask']
  allowed=torch.where(hm[:,:,None],hm[:,None,:]&(idx[None,:]<=idx[:,None])[None],valid[:,None,:])
  allowed=allowed&valid[:,:,None]&valid[:,None,:]
  time=bound['t'].float()[:,None].expand(-1,n).clone()
  time[hm]=0 # Clean history tokens have zero noise level, matching partial attention.
  out=self.core([z.transpose(0,1)[:,:,None,None] for z in a],list(time),list(text[:,None]),n,
   attn_mask=list(allowed),text_seq_len=1)
  return torch.stack([z[:,:,0,0].transpose(0,1) for z in out])
