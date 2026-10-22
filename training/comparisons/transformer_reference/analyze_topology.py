from common import *
import math,torch
from motion_training.io import identity
from motion_training.resource_switch import values
torch.set_num_threads(2);T=ROOT/'gates';thresholds=read(T/'TOPOLOGY_THRESHOLDS.json')
restore=[read(T/x/'RESTORED.json') for x in ['two','four']];assert restore[0]==restore[1]
updates=[read(T/x/'UPDATE.json') for x in ['two','four']]
receipts=[{r['id']:{k:r[k] for k in ['seed','input_hash','target_hash','noise_hash','physical_G_hash']} for rank in u['receipts'] for r in rank} for u in updates]
assert receipts[0]==receipts[1] and len(receipts[0])==32
parent=torch.load(read(ROOT/'PLAN.json')['start_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
aa=torch.load(T/'two/checkpoint.pt',map_location='cpu',mmap=True,weights_only=False);bb=torch.load(T/'four/checkpoint.pt',map_location='cpu',mmap=True,weights_only=False)
vp,va,vb=values(parent),values(aa),values(bb);a=b=dot=err=maximum=0.
for k,p in vp.items():
 x=(va[k].float()-p.float()).double().flatten();y=(vb[k].float()-p.float()).double().flatten();a+=float(x@x);b+=float(y@y);dot+=float(x@y);err+=float((x-y).square().sum());maximum=max(maximum,float((va[k].float()-vb[k].float()).abs().max()))
update_cos=dot/max((a*b)**.5,1e-30);update_rel=(err/max(a,1e-30))**.5
assert aa['framework']['scheduler']==bb['framework']['scheduler'];assert len(set(bb['framework']['rank_state_hashes']))==1
ga=torch.load(T/'two/gradients.pt',map_location='cpu',mmap=True,weights_only=False);gb=torch.load(T/'four/gradients.pt',map_location='cpu',mmap=True,weights_only=False);a=b=dot=err=0.
assert ga.keys()==gb.keys()
for k,x in ga.items():
 x=x.double().flatten();y=gb[k].double().flatten();a+=float(x@x);b+=float(y@y);dot+=float(x@y);err+=float((x-y).square().sum())
grad_cos=dot/max((a*b)**.5,1e-30);grad_rel=(err/max(a,1e-30))**.5;lossdiff=abs(updates[0]['loss']-updates[1]['loss'])
passed=grad_cos>=thresholds['gradient_cosine_min'] and grad_rel<=thresholds['gradient_relative_max'] and update_cos>=thresholds['master_update_cosine_min'] and update_rel<=thresholds['master_update_relative_max'] and maximum<=thresholds['master_max_absolute'] and lossdiff<=thresholds['loss_absolute_max']
r=dict(passed=passed,exact_parent_and_Adam_restore=True,sample_hashes_exact=True,four_ranks_agree=True,scheduler_exact=True,gradient_cosine=grad_cos,gradient_relative_l2=grad_rel,update_cosine=update_cos,update_relative_l2=update_rel,master_max_absolute=maximum,loss_difference=lossdiff,thresholds=thresholds,probe_weights_discarded=True)
write(T/'TOPOLOGY.json',r);print(r,flush=True);assert passed,r
