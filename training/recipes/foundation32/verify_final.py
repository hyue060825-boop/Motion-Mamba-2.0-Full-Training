from motion_mamba2.paths import artifact_path
import os,sys,json,gc
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from motion_training import runtime,engine,data,losses,batching
from motion_training.io import write
cfg=engine.config(D/'configs/train.json');m,c,ck=runtime.load(cfg['parent'],True);raw=m.denoiser.model;raw.eval()
plan=json.loads(Path(artifact_path('manifest_json_d6a24ce639')).read_text())['plan'];results=[]
thresholds=dict(loss_absolute=2e-5,derivative_relative_l2=.0001,derivative_cosine=.999999,prediction_relative=.03,hybrid_derivative_cosine=.995,hybrid_derivative_relative_l2=.05)
write(D/'DECOMPOSED_EQUIVALENCE_THRESHOLDS.json',thresholds)
for h,f in data.BUCKETS:
 records=[r for r in plan if r['H']==h and r['F']==f][:8];fields=[batching.corrected_build(m,c,r,cfg,audit=False)[0] for r in records]
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  serial=[]
  for item in fields:
   c.reset_memory();serial.append(raw(*(item[k] for k in losses.FORWARD_KEYS)).detach())
  c.reset_memory();bf=batching.collate(fields);bp=raw(*(bf[k] for k in losses.FORWARD_KEYS)).detach()
 pred=torch.cat(serial).float();pr=float((pred-bp).norm()/pred.norm())
 # Use IDENTICAL predictions to independently test batch reductions and codec derivatives.
 reference=[];sv=[];losses.decode_G=batching.ORIGINAL_DECODE
 with torch.autocast('cuda',enabled=False):
  for p,item in zip(pred,fields):
   leaf=p[None].detach().clone().requires_grad_();value,_,_=losses.components(raw,item,leaf);sv.append(float(value));reference.append(torch.autograd.grad(value/8,leaf)[0])
  leaf=pred.detach().clone().requires_grad_();batching.install();value,_=batching.independent_components(raw,bf,leaf);gradient=torch.autograd.grad(value,leaf)[0];ref=torch.cat(reference)
 rel=float((ref-gradient).norm()/ref.norm());cos=float(torch.nn.functional.cosine_similarity(ref.flatten(),gradient.flatten(),dim=0));ld=abs(float(value)-sum(sv)/8)
 # Denoiser gradients from the smooth latent objective, isolate geometry conditioning.
 raw.zero_grad(set_to_none=True)
 for item in fields:
  with torch.autocast('cuda',dtype=torch.bfloat16):_,_,terms=losses.compute(raw,c,item,cfg['loss'])
  (terms['hybrid']/8).backward()
 grads={n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None};raw.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):_,_,terms=losses.compute(raw,c,bf,cfg['loss'])
 terms['hybrid'].backward();a=b=dot=err=0.
 for n,p in raw.named_parameters():
  if n not in grads:continue
  x=grads.pop(n).double();y=p.grad.detach().float().cpu().double() if p.grad is not None else torch.zeros_like(x);a+=float(x.square().sum());b+=float(y.square().sum());dot+=float((x*y).sum());err+=float((x-y).square().sum())
 hr=(err/a)**.5;hc=dot/(a*b)**.5
 result=dict(H=h,F=f,loss_absolute=ld,derivative_relative_l2=rel,derivative_cosine=cos,prediction_relative=pr,hybrid_derivative_cosine=hc,hybrid_derivative_relative_l2=hr)
 result['pass']=ld<thresholds['loss_absolute'] and rel<thresholds['derivative_relative_l2'] and cos>thresholds['derivative_cosine'] and pr<thresholds['prediction_relative'] and hc>thresholds['hybrid_derivative_cosine'] and hr<thresholds['hybrid_derivative_relative_l2']
 results.append(result);write(D/'DECOMPOSED_EQUIVALENCE.json',dict(results=results,all_pass=all(r['pass'] for r in results),whole_weight_geometry_gradient_bitwise_equivalence=False,interpretation='BF16 batch GEMM reduction changes predictions; physical decoder Jacobian can amplify prediction changes. Fixed-prediction loss and gradient comparison isolates semantic equivalence; do not claim identical optimizer trajectory.',thresholds=thresholds));print('DECOMPOSED',result,flush=True)
 assert result['pass'],result
 raw.zero_grad(set_to_none=True);del fields,bf,grads;gc.collect();torch.cuda.empty_cache()
