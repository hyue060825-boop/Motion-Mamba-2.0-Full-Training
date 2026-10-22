"""Admitted Native batch geometry, keeping frozen codec B1 Graph and recipe."""
from common import ROOT, read
from motion_training.io import sha
import pipeline_accel as fast
import accelerated_loss
from motion_training import batching, losses
from g1_training_precision import prediction_loss_context
import torch
class LossModule(torch.nn.Module):
 def __init__(self,raw,client):
  super().__init__();self.raw=raw;self.client=client
 def forward(self,f):
  self.client.reset_memory();pred=self.raw(*(f[k] for k in losses.FORWARD_KEYS))
  with prediction_loss_context(self.raw,pred):
   if pred.shape[0]==8:return accelerated_loss.components(self.raw,f,pred)
   assert pred.shape[0]==64
   value,info,_=fast.components_group(self.raw,f,pred)
   return value,info
VERSION='native_batched_geometry_admitted_20261009_v1'
def install():
 proof=read(ROOT/'gates/NATIVE_GEOMETRY_ADMISSION.json')
 assert proof['passed'] and proof['architecture']==read(ROOT/'PLAN.json')['architecture']
 for name,digest in proof['source_hashes'].items():
  assert sha(ROOT/name)==digest,(name,'Changed accelerated source')
 for name in ['GEOMETRY_BATCH64_ADMISSION.json','GEOMETRY_FULL_UPDATE.json']:
  assert read(ROOT/'gates'/name)['passed']
 update=fast.install('serial_codec',4)
 batching.LossModule=LossModule
 return update
