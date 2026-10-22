from pathlib import Path
import torch
from g1_bounded_codec import encode_windows
from g1_motion_contract import decode_committed_window
from motion_training import sampling
def contract(m,batch):
 seq=batch['motion'][:,:24].clone();T=batch['motion'].shape[1];B=seq.shape[0]
 pad=torch.ones(B,24,device=m.device,dtype=torch.bool)
 previous,_=encode_windows(m.hybrid,seq,pad.sum(1),pad)
 history,translation,heading=m._encode_init_history(seq,B)
 text=batch['text_feature'][:,None];tm=torch.ones(text.shape[:2],device=m.device,dtype=torch.bool)
 traces=[]
 while seq.shape[1]<T:
  end=seq.shape[1];h=history.shape[1]*4;start=end-h
  capacity_end=min(T,start+248)
  mm=batch['motion_mask'][:,start:capacity_end].clone();obs=batch['observed_motion'][:,start:capacity_end].clone()
  mm[:,:h]=0;obs[:,:h]=0
  m.diffusion.calc_diffusion_vars(m.diffusion.space_timesteps(10)[0])
  latent=m._generate_window(history,translation,0,h,capacity_end-start,text,tm,heading,mm,obs,
   torch.tensor([10],device=m.device),(2.,2.),list(range(9,-1,-1)),progress_bar=lambda x:x,target_motion=None,cfg_type=None)
  latent,center,root=m._recenter_history(latent,torch.full((B,),latent.shape[1]*4-1,device=m.device),requantize=True)
  translation=translation+center
  headings=m.motion_rep.get_root_heading_angle(m.motion_rep.global_root_stats.unnormalize(root))
  heading=headings[:,max(0,root.shape[1]-196)]
  history=latent[:,-49:].clone()
  world=sampling.world_motion(m,latent[:,-13:],translation)
  decoded,previous=decode_committed_window(m,world,previous)
  seq=torch.cat((seq,decoded),1)
  traces.append(dict(window=len(traces)+1,history=h,total=T-start,actual_input_frames=capacity_end-start,future_capacity_frames=capacity_end-start-h-52,goal_visible=bool(mm.count_nonzero()),decoder_context_frames=min(end,52)))
 return seq,traces

class CommittedMOTION_MAMBA2:
 def __init__(self,model,trace_path):self.model=model;self.trace_path=trace_path;self.calls=[]
 def __getattr__(self,name):return getattr(self.model,name)
 def __call__(self,texts,num_frames,num_denoising_steps,pad_mask,first_heading_angle,motion_mask,observed_motion,target_motion=None,cfg_weight=2.,text_feat=None,text_pad_mask=None,cfg_type=None,progress_bar=None,return_text_embeddings_x_dict=False,crop_history_length=None,init_history_sequence=None):
  assert num_denoising_steps==10 and target_motion is None and crop_history_length is None
  assert init_history_sequence.shape[1]==24 and bool(pad_mask.all()) and tuple(cfg_weight)==(2.,2.)
  dummy=torch.cat((init_history_sequence,init_history_sequence.new_zeros(init_history_sequence.shape[0],num_frames-24,414)),1)
  batch=dict(motion=dummy,motion_mask=motion_mask,observed_motion=observed_motion,text_feature=text_feat[:,0])
  out,trace=contract(self.model,batch)
  assert out.shape[1]==num_frames and all(x['history']<=196 and x['decoder_context_frames']<=52 for x in trace)
  self.calls.append(dict(batch_size=out.shape[0],windows=trace))
  from util import write
  write(self.trace_path,dict(status='ACTUAL_FIXED_CAPACITY_248_HISTORY_196_COMMITTED_DECODER_52',calls=self.calls,selection='checkpoint contract, not quality-score selection'))
  return out
