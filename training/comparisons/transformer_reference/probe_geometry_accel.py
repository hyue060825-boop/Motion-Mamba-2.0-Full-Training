from common import *
import torch, copy, statistics, argparse, time
import adapter, pipeline_accel as fast, accelerated_loss, serial_geometry_graph as graphloss
from motion_training import runtime, batching, losses, streaming
from motion_training.io import identity
import lazy_plan

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--batch-only',action='store_true');p.add_argument('--micro',type=int,default=8);a=p.parse_args()
adapter.install();cfg=read(ROOT/'configs/batch512.json')
m,c,ck=runtime.load(a.checkpoint,True);raw=m.denoiser.model
late=lazy_plan.Plan(read(LATE/'MANIFEST_TEMPLATE.json'))
results=[]
def backward(fields, kind):
    raw.zero_grad(set_to_none=True);raw.train();c.reset_memory()
    torch.manual_seed(222);torch.cuda.manual_seed_all(222)
    if kind=='graph':
        with torch.autocast('cuda',dtype=torch.bfloat16):value,info=module(fields)
    else:
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=raw(*(fields[k] for k in losses.FORWARD_KEYS))
        from g1_training_precision import prediction_loss_context
        with prediction_loss_context(raw,pred):
            value,info=accelerated_loss.components(raw,fields,pred) if kind=='serial' else fast.components_group(raw,fields,pred)[:2]
    (value/2).backward();torch.cuda.synchronize()
    return float(value),{n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
def compare(a,b):
    assert a.keys()==b.keys()
    norm=sum(float(x.square().sum()) for x in a.values());err=sum(float((a[n]-b[n]).square().sum()) for n in a)
    return (err/max(norm,1e-30))**.5
module=graphloss.LossModule(raw,c)
for i,layout in enumerate([(0,0),(52,0),(196,0),(0,196),(52,144),(144,52)]):
    records=late.records_at(321+i,a.micro,layout=layout)
    cpu=fast.materialize(fast.submit_reads(c,records,cfg,4));fields,_=fast.build_group(m,c,cpu,cfg,False,serial_codec=True)
    v0,g0=backward(fields,'serial');vb,gb=backward(fields,'batch');batchrel=compare(g0,gb);del gb
    print(dict(layout=layout,batch_gradient_relative=batchrel,batch_loss_relative=abs(vb-v0)/max(abs(v0),1e-9)),flush=True)
    if a.batch_only:
        results.append(dict(layout=layout,batch_gradient_relative=batchrel,batch_loss_relative=abs(vb-v0)/max(abs(v0),1e-9)));del g0;continue
    vg,gg=backward(fields,'graph');rel=compare(g0,gg);del g0,gg
    row=dict(layout=layout,serial=v0,batch=vb,graph=vg,batch_gradient_relative=batchrel,graph_gradient_relative=rel,graph_loss_relative=abs(vg-v0)/max(abs(v0),1e-9),replays=module.cache.replays,peak_GiB=torch.cuda.max_memory_reserved()/2**30)
    print(row,flush=True);results.append(row)
    write(ROOT/'gates/GEOMETRY_GRAPH_PROBE_PARTIAL.json',results)
    assert rel<=1e-3 and row['graph_loss_relative']<=1e-5,row
write(ROOT/('gates/GEOMETRY_BATCH_PROBE.json' if a.batch_only else 'gates/GEOMETRY_GRAPH_PROBE.json'),dict(passed=True,cases=results,checkpoint=a.checkpoint,optimizer_updates=0))
print('GEOMETRY_GRAPH_PROBE_PASSED',flush=True)
