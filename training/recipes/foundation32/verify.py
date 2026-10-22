from motion_mamba2.paths import artifact_path
import os,sys,json,time,copy,gc
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from motion_training import runtime,engine,data,losses,batching
from motion_training.io import write
cfg=engine.config(D/'configs/train.json')
m,c,ck=runtime.load(cfg['parent'],True);raw=m.denoiser.model
raw.train();m.autoencoder.eval()
manifest=json.loads(Path(artifact_path('manifest_json_d6a24ce639')).read_text())
plan=manifest['plan'];results=[]
# Preregister before computing. BF16 batched kernels have different reduction order.
thresholds=dict(objective_rel=0.01,gradient_cosine=0.995,gradient_relative_l2=0.05,largest_normalized_component_delta=0.02)
write(D/'BATCH_EQUIVALENCE_THRESHOLDS.json',thresholds)
for h,f in data.BUCKETS:
 records=[r for r in plan if r['H']==h and r['F']==f][:8]
 fields=[batching.corrected_build(m,c,r,cfg,audit=False)[0] for r in records]
 opt=None;raw.zero_grad(set_to_none=True);losses.decode_G=batching.ORIGINAL_DECODE
 vals=[];started=time.perf_counter()
 for item in fields:
  with torch.autocast('cuda',dtype=torch.bfloat16):value,_,_=losses.compute(raw,c,item,cfg['loss'])
  vals.append(float(value));(value/8).backward()
 torch.cuda.synchronize();serial_seconds=time.perf_counter()-started
 reference={n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
 raw.zero_grad(set_to_none=True);batching.install();fields=batching.collate(fields)
 started=time.perf_counter()
 with torch.autocast('cuda',dtype=torch.bfloat16):value,_,_=losses.compute(raw,c,fields,cfg['loss'])
 value.backward();torch.cuda.synchronize();batch_seconds=time.perf_counter()-started
 dot=aa=bb=err=0.;maxdiff=0.;same_unused=True
 for n,p in raw.named_parameters():
  if n not in reference:
   if p.grad is not None and p.grad.count_nonzero():same_unused=False
   continue
  a=reference.pop(n);b=p.grad.detach().float().cpu() if p.grad is not None else torch.zeros_like(a)
  dot+=float((a.double()*b).sum());aa+=float(a.double().square().sum());bb+=float(b.double().square().sum());err+=float((a.double()-b).square().sum())
  maxdiff=max(maxdiff,float((a-b).abs().max())/max(float(a.abs().max()),1e-6));del a,b
 cosine=dot/(aa*bb)**.5;rel=(err/max(aa,1e-30))**.5;obj=abs(float(value)-sum(vals)/8)/max(abs(sum(vals)/8),1e-9)
 result=dict(H=h,F=f,objective_relative=obj,gradient_cosine=cosine,gradient_relative_l2=rel,largest_normalized_component_delta=maxdiff,same_unused=same_unused,serial_seconds=serial_seconds,batch_seconds=batch_seconds)
 result['pass']=obj<thresholds['objective_rel'] and cosine>thresholds['gradient_cosine'] and rel<thresholds['gradient_relative_l2'] and same_unused
 results.append(result);print('EQUIVALENCE',result,flush=True);write(D/'BATCH_EQUIVALENCE.json',dict(thresholds=thresholds,results=results,all_pass=all(r['pass'] for r in results),dropout_zero=True))
 assert result['pass'],result
 raw.zero_grad(set_to_none=True);del fields,reference;gc.collect();torch.cuda.empty_cache()
print('ALL_LAYOUTS_PASS',flush=True)
