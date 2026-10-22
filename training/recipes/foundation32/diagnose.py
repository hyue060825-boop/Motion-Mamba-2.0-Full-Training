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
for term in ['hybrid','decoded_body','goal','fk_consistency']:
 h=f=0
 records=[r for r in plan if r['H']==h and r['F']==f][:8]
 fields=[batching.corrected_build(m,c,r,cfg,audit=False)[0] for r in records]
 raw.zero_grad(set_to_none=True);losses.decode_G=batching.ORIGINAL_DECODE;vals=[];preds=[]
 for item in fields:
  with torch.autocast('cuda',dtype=torch.bfloat16):value,info,terms=losses.compute(raw,c,item,cfg['loss'],return_graph=True)
  preds.append(info['prediction'].detach());vals.append(float(terms[term]));(terms[term]/8).backward()
 reference={n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
 raw.zero_grad(set_to_none=True)
 # Keep the original serial decoder to isolate denoiser batching.
 fields=batching.collate(fields)
 with torch.autocast('cuda',dtype=torch.bfloat16):value,info,terms=losses.compute(raw,c,fields,cfg['loss'],return_graph=True)
 terms[term].backward();aa=bb=dot=err=0.;groups={}
 for n,p in raw.named_parameters():
  if n not in reference:continue
  a=reference.pop(n);b=p.grad.detach().float().cpu() if p.grad is not None else torch.zeros_like(a)
  ad=a.double();bd=b.double();stats=[float((ad*bd).sum()),float(ad.square().sum()),float(bd.square().sum()),float((ad-bd).square().sum())]
  key='.'.join(n.split('.')[:3]);acc=groups.setdefault(key,[0.,0.,0.,0.]);groups[key]=[v+w for v,w in zip(acc,stats)]
  dot+=stats[0];aa+=stats[1];bb+=stats[2];err+=stats[3];del a,b,ad,bd
 predref=torch.cat(preds);prednow=info['prediction'];pr=float((predref-prednow).float().norm()/predref.float().norm())
 groups={k:dict(cosine=v[0]/max((v[1]*v[2])**.5,1e-30),relative=(v[3]/max(v[1],1e-30))**.5,reference_norm=v[1]**.5) for k,v in groups.items()}
 r=dict(term=term,serial_value=sum(vals)/8,batch_value=float(terms[term]),cosine=dot/max((aa*bb)**.5,1e-30),relative=(err/max(aa,1e-30))**.5,prediction_relative=pr,groups=groups)
 results.append(r);write(D/'BATCH_DIAGNOSIS.json',results);print('DIAGNOSIS',term,r['cosine'],r['relative'],pr,flush=True)
 raw.zero_grad(set_to_none=True);del fields,reference,preds;gc.collect();torch.cuda.empty_cache()
