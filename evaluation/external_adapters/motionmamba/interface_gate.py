from common import *
import traceback
import numpy as np
import torch
from motion_training import distributed as parallel,sampling
from g1_bounded_codec import encode_windows
from g1_motion_contract import decode_committed_window
import pipeline_accel as fast

def rollout(m,batch):
 seq=batch['motion'][:,:24].clone();T=batch['motion'].shape[1];B=len(seq)
 pad=torch.ones(B,24,device=m.device,dtype=torch.bool)
 previous,_=encode_windows(m.hybrid,seq,pad.sum(1),pad)
 history,translation,heading=m._encode_init_history(seq,B)
 text=batch['text'][:,None];tm=torch.ones(text.shape[:2],device=m.device,dtype=torch.bool);tr=[]
 while seq.shape[1]<T:
  end=seq.shape[1];h=history.shape[1]*4;start=end-h
  mask=batch['mask'][:,start:].clone();obs=batch['observed'][:,start:].clone();mask[:,:h]=0;obs[:,:h]=0
  latent=m._generate_window(history,translation,0,h,T-start,text,tm,heading,mask,obs,
   torch.tensor([10],device=m.device),(2.,2.),list(range(9,-1,-1)),progress_bar=lambda x:x,target_motion=None,cfg_type=None)
  latent,center,root=m._recenter_history(latent,torch.full((B,),latent.shape[1]*4-1,device=m.device),requantize=True)
  translation+=center
  headings=m.motion_rep.get_root_heading_angle(m.motion_rep.global_root_stats.unnormalize(root))
  heading=headings[:,max(0,root.shape[1]-196)];history=latent[:,-49:].clone()
  world=sampling.world_motion(m,latent[:,-13:],translation);decoded,previous=decode_committed_window(m,world,previous)
  seq=torch.cat((seq,decoded),1)
  tr.append(dict(window=len(tr)+1,history_frames=h,input_frames=T-start,goal_visible=bool(mask.any()),decoder_history_frames=min(end,52)))
 return seq,tr

def actual(m,raw):
 cfg=read(ROOT/'configs/batch32.json')
 from motion_training import streaming
 record=streaming.load(OLD/'train/data/MANIFEST.json')['plan'][0:32][0]
 prepared=fast.read_cpu(record,cfg)
 a=torch.tensor(prepared['a'][:24],device=m.device)[None]
 a[:,:,0]-=a[:,23:24,0].clone();a[:,:,2]-=a[:,23:24,2].clone()
 prefix=m.motion_rep.normalize(a);text=torch.tensor(prepared['text'],device=m.device)[None]
 B,T,H,G=1,1012,24,52;N=T//4;device=m.device
 mask=torch.zeros(B,T,414,device=device);observed=torch.zeros_like(mask)
 target=torch.tensor([1.5,-.5],device=device)
 physical=torch.zeros_like(mask);physical[:,-1,[0,2]]=target
 normalized=m.motion_rep.normalize(physical)
 mask[:,-1,[0,2]]=1;observed[:,-1,[0,2]]=normalized[:,-1,[0,2]]
 recovered=m.motion_rep.unnormalize(observed)[:,-1,[0,2]]
 assert torch.allclose(recovered,target[None],atol=2e-6,rtol=1e-6)
 hist,_,heading=m._encode_init_history(prefix,B)
 x=torch.zeros(B,N,148,device=device);x[:,:H//4]=hist;x[:,H//4:H//4+13]=torch.randn(B,13,148,device=device)
 ti=torch.arange(N,device=device)[None];fi=torch.arange(T,device=device)[None]
 args=dict(x=x,history_len=torch.tensor([H],device=device),generation_len=torch.tensor([G],device=device),future_len=torch.tensor([T-H-G],device=device),
  history_mask=fi<H,generation_mask=(fi>=H)&(fi<H+G),future_mask=fi>=H+G,
  history_token_mask=ti<H//4,generation_token_mask=(ti>=H//4)&(ti<(H+G)//4),future_token_mask=ti>=(H+G)//4,
  text_feat=text[:,None],text_pad_mask=torch.ones(B,1,device=device,dtype=torch.bool),t=torch.tensor([500],device=device),
  first_heading_angle=heading,motion_mask=mask,observed_motion=observed)
 obs=observed.detach().clone().requires_grad_()
 out=raw(**dict(args,observed_motion=obs));gen=out[:,H//4:H//4+13]
 derivative=torch.autograd.grad(gen.square().mean(),obs)[0]
 goal_gradient=float(derivative[:,-1,[0,2]].abs().max())
 assert torch.isfinite(out).all() and goal_gradient>0,'Long future root target has no computational path to G'
 del out,gen,derivative
 with torch.no_grad():
  embedding,_,_=raw.prepared(**args)
  # All414 physical columns retain the same coordinate order, including
  # root and selected joint conditions. Check the exact mask/observation pack.
  assert torch.equal(embedding[:,:,148:148+3312].reshape(B,T,828),torch.cat([observed*mask,mask],-1))
  baseline=raw(**args)
  # Poison ONLY the other batch member. No sample may use its neighbor.
  args2={k:v.repeat((2,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) else v for k,v in args.items()}
  together=raw(**args2)
  poisoned_args={k:v.clone() if torch.is_tensor(v) else v for k,v in args2.items()}
  poisoned_args['x'][1]+=19;poisoned_args['text_feat'][1]-=7
  poisoned_args['observed_motion'][1]+=999
  changed=raw(**poisoned_args)
  assert torch.equal(together[0],changed[0]),'Cross-sample leakage in HTM/BSM axes'
  taint=observed.clone();taint[:,H+1,100]=9999
  assert torch.equal(baseline,raw(**dict(args,observed_motion=taint)))
 batch=dict(motion=torch.cat([prefix,prefix.new_zeros(B,T-24,414)],1),mask=mask,observed=observed,text=text)
 torch.manual_seed(6137);torch.cuda.manual_seed_all(6137)
 with torch.inference_mode():first,trace=rollout(m,batch)
 poisoned=dict(batch,motion=batch['motion'].clone());poisoned['motion'][:,24:]=12345
 torch.manual_seed(6137);torch.cuda.manual_seed_all(6137)
 with torch.inference_mode():second,trace2=rollout(m,poisoned)
 assert torch.equal(first,second),'Unobserved future GT leaked into rollout'
 assert first.shape==(1,1012,414) and torch.isfinite(first).all() and len(trace)==19
 assert all(q['goal_visible'] and q['history_frames']<=196 for q in trace)
 return dict(passed=True,input_frames=T,physical_fps=25,prefix_frames=24,actual_goal_frame=1011,goal_time_sec=40.48,
  target_root_XZ_m=target.tolist(),long_future_goal_gradient=goal_gradient,root_normalizer_roundtrip=True,
  all414_coordinate_order_exact=True,no_time_compression_or_target_relocation=True,goal_visible_in_every_window=True,
  completed_generation_windows=19,actual_committed52_decoder=True,unobserved_future_GT_taint_delta=0.,
  masked_unobserved_values_no_effect=True,batch_neighbor_poison_exact=True,all_256_capacity_no_goal_crop=True,traces=trace,temporary_check_weights_only=True,
  performance_not_gate_criterion=True,full_system_reproduction=False)

def run(m,raw,opt):
 result=None
 if parallel.leader():
  flags=[p.requires_grad for p in raw.parameters()];training=raw.training
  try:
   raw.eval();raw.requires_grad_(False)
   with torch.enable_grad():result=actual(m,raw)
   write(ROOT/'gates/INTERFACE.json',result)
  except BaseException:
   result=dict(passed=False,error=traceback.format_exc());write(ROOT/'gates/INTERFACE.json',result)
  finally:
   for p,flag in zip(raw.parameters(),flags):p.requires_grad_(flag)
   raw.train(training);opt.zero_grad(set_to_none=True);torch.cuda.empty_cache()
 results=parallel.gather(result)
 assert results[0]['passed'],results[0]
