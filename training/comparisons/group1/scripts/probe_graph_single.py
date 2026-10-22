from common import *
import argparse,time
import torch
from model_io import load
from motion_training import losses
import pipeline_accel as fast,lazy_plan
import serial_geometry_graph as graph

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.cuda.set_device(0)
torch.distributed.init_process_group('nccl',init_method='file://'+str(Path(a.output).parent/'single_nccl_init'),rank=0,world_size=1)
m,c,opt,ck=load(a.checkpoint);raw=m.denoiser.model;cfg=read(ROOT/'configs/batch512.json')
late=lazy_plan.Plan(read(LATE/'MANIFEST_TEMPLATE.json'));cache=graph.GeometryGraphs(raw,m.hybrid);rows=[]
for hf in [(0,0),(52,0)]:
 print('SINGLE_START',hf,flush=True)
 rs=late.records_at(ck['step']-75000,64,layout=hf)
 f=fast.build_group(m,c,fast.materialize(fast.submit_reads(c,rs,cfg,2)),cfg,False,serial_codec=True)[0]
 raw.eval()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p0=raw(*(f[k] for k in losses.FORWARD_KEYS))
 pred=p0.detach().clone().requires_grad_();h=hf[0];lo=h//4
 with torch.autocast('cuda',enabled=False):
  ref=torch.stack([fast.components_group(raw,{k:v[i:i+1] if torch.is_tensor(v) else v for k,v in f.items()},pred[i:i+1])[0] for i in range(64)]).mean()
 ref.backward();g0=pred.grad[:,lo:lo+13].detach().clone()
 fixed=dict(target=f['target'][:,lo:lo+13],supervised=f['supervised_motion'],observed=f['observed'][:,h:h+52],mask=f['motion_mask'][:,h:h+52].bool(),prefix=f.get('decoder_history_hybrid'))
 value,diag,g=cache.run(p0[:,lo:lo+13],fixed)
 rel=float((g-g0).double().square().sum().sqrt()/g0.double().square().sum().sqrt())
 print('GRAPH_RESULT',hf,float(value),float(ref),rel,'max',float((g-g0).abs().max()),flush=True)
 ts=[]
 for j in range(5):
  t=time.perf_counter();cache.run(p0[:,lo:lo+13],fixed);torch.cuda.synchronize();ts.append(time.perf_counter()-t)
 rows.append(dict(layout=hf,loss_delta=abs(float(ref)-float(value)),prediction_gradient_relative=rel,gradient_max=float((g-g0).abs().max()),times=ts,constants=len(cache.constants.values),peak_GiB=torch.cuda.max_memory_reserved()/2**30))
 write(a.output,dict(rows=rows,passed=all(r['loss_delta']<=1e-5 and r['prediction_gradient_relative']<=1e-6 for r in rows),production_updates=0))
torch.distributed.destroy_process_group()
