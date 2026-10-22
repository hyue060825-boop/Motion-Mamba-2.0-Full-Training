from common import *
import torch,time,copy
import adapter,pipeline_accel as fast,accelerated_loss
from motion_training import runtime,batching,losses,streaming
from motion_training.io import identity
adapter.install();cfg=read(ROOT/'configs/batch32.json')
m,c,ck=runtime.load(Path(read(ROOT/'PLAN.json')['start_checkpoint']),True);raw=m.denoiser.model
plan=streaming.load(OLD/'train/data/MANIFEST.json')['plan']
selected={}
for step in range(200):
 rs=plan[step*32:(step+1)*32];selected.setdefault((rs[0]['H'],rs[0]['F']),rs[:8])
 if len(selected)==6:break
assert len(selected)==6
results=[]
def fields(rs):
 fs=[batching.corrected_build(m,c,r,cfg,audit=False)[0] for r in rs]
 return {k:torch.cat([x[k] for x in fs],0) if torch.is_tensor(v) else v for k,v in fs[0].items()}
def backward(f,group):
 raw.zero_grad(set_to_none=True);raw.train();torch.manual_seed(222);torch.cuda.manual_seed_all(222)
 with torch.autocast('cuda',dtype=torch.bfloat16):pred=raw(*(f[k] for k in losses.FORWARD_KEYS))
 from g1_training_precision import prediction_loss_context
 with prediction_loss_context(raw,pred):
  value=accelerated_loss.components(raw,f,pred)[0] if group else batching.independent_components(raw,f,pred)[0]
 value.backward();torch.cuda.synchronize()
 return float(value),{n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
for layout,rs in selected.items():
 ref=fields(rs);prepared=fast.materialize(fast.submit_reads(c,rs,cfg,4));opt,_=fast.build_group(m,c,prepared,cfg,False,serial_codec=True)
 exact=all(torch.equal(v,opt[k]) for k,v in ref.items() if torch.is_tensor(v))
 assert exact,('input_difference',layout,[k for k,v in ref.items() if torch.is_tensor(v) and not torch.equal(v,opt[k])])
 a,ga=backward(ref,False);b,gb=backward(opt,True)
 assert ga.keys()==gb.keys()
 err=sum(float((ga[n]-gb[n]).square().sum()) for n in ga);norm=sum(float(ga[n].square().sum()) for n in ga)
 rel=(err/max(norm,1e-30))**.5;lossrel=abs(a-b)/max(abs(a),1e-9)
 passed=rel<=1e-3 and lossrel<=1e-5
 results.append(dict(layout=layout,inputs_exact=exact,loss_reference=a,loss_fast=b,gradient_relative_l2=rel,passed=passed,peak_GiB=torch.cuda.max_memory_reserved()/2**30))
 assert passed,results[-1]
 del ga,gb,ref,opt
# Exact graph/non-graph optimizer update comparison at the SAME grouped reductions.
rs=selected[(52,0)];cpu=fast.materialize(fast.submit_reads(c,rs,cfg,4))
initial={n:v.detach().clone() for n,v in raw.state_dict().items()};states=[]
for graph in [False,True]:
 raw.load_state_dict(initial);o=runtime.optimizer(m,ck,cfg)
 f=fields(rs) if not graph else fast.build_group(m,c,cpu,cfg,False,True)[0]
 val,g=backward(f,True);torch.nn.utils.clip_grad_norm_(raw.parameters(),1,error_if_nonfinite=True);o.step()
 states.append(identity([raw.state_dict(),o.state_dict()]));del o,g,f
assert states[0]==states[1],'Adam update changed under graph'
write(ROOT/'gates/ACCELERATION.json',dict(passed=True,cases=results,graph_adam_states_exact=True,discarded_optimizer_updates=2,thresholds=dict(gradient_relative_l2=1e-3,loss_relative=1e-5),training_weights_untouched=True))
print('ACCELERATION_PASSED',flush=True)
