"""Preserve B1 geometric GEMM/backward shapes; batch only denoiser forward."""
import torch
import pipeline_accel as fast
from motion_training import losses,batching
from g1_training_precision import prediction_loss_context
def components(raw,f,pred):
 values=[];diagnostics=[]
 for i in range(pred.shape[0]):
  value,info,_=fast.components_group(raw,batching.split_field(f,i),pred[i:i+1])
  values.append(value);diagnostics.append(info['diagnostics'])
 return torch.stack(values).mean(),dict(diagnostics=torch.stack(diagnostics).mean(0))
class LossModule(torch.nn.Module):
 def __init__(self,raw,c):super().__init__();self.raw=raw;self.client=c
 def forward(self,f):
  self.client.reset_memory();pred=self.raw(*(f[k] for k in losses.FORWARD_KEYS))
  with prediction_loss_context(self.raw,pred):return components(self.raw,f,pred)
def install():
 update=fast.install('serial_codec',4)
 batching.LossModule=LossModule
 return update
