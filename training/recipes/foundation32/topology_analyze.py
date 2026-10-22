from motion_mamba2.paths import artifact_path
import sys,json,math,statistics,hashlib
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'));T=D/'topology4';sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src')]
import torch
from motion_training.io import read,write,rows,sha,identity
from motion_training import engine
from motion_training.resource_switch import values

torch.set_num_threads(2)
thresholds=read(T/'THRESHOLDS.json');cfg=engine.config(D/'configs/train.json');parent=read(T/'PARENT.json');rr={k:rows(T/k/'updates.jsonl') for k in ['two','four']}
restored={k:read(T/k/'RESTORED.json') for k in ['two','four']};same_restore=restored['two']==restored['four'];assert same_restore
receipts={}
for arm in ['two','four']:
 receipts[arm]={r['id']:{k:r[k] for k in ['seed','input_hash','target_hash','noise_hash','physical_G_hash']} for rank in rr[arm][0]['receipts'] for r in rank}
same_inputs=receipts['two']==receipts['four'] and len(receipts['two'])==32
pa=torch.load(parent['checkpoint'],map_location='cpu',mmap=True,weights_only=False);ca=torch.load(T/'two/checkpoint1.pt',map_location='cpu',mmap=True,weights_only=False);cb=torch.load(T/'four/checkpoint1.pt',map_location='cpu',mmap=True,weights_only=False)
vp,va,vb=values(pa),values(ca),values(cb);a=b=dot=err=0.;maximum=0.
for k,p in vp.items():
 x=(va[k].float()-p.float()).flatten().double();y=(vb[k].float()-p.float()).flatten().double();a+=float(x@x);b+=float(y@y);dot+=float(x@y);err+=float((x-y).square().sum());maximum=max(maximum,float((va[k].float()-vb[k].float()).abs().max()))
update_cos=dot/max((a*b)**.5,1e-30);update_rel=(err/max(a,1e-30))**.5
ranks_equal=len(cb['framework']['rank_state_hashes'])==4 and len(set(cb['framework']['rank_state_hashes']))==1
assert ca['framework']['scheduler']==cb['framework']['scheduler']
assert ca['step']==cb['step']==parent['step']+1
del pa,ca,cb,vp,va,vb
# Compare gradient direction and preclip scale, not only resulting Adam changes.
ga=torch.load(T/'two/first_gradients.pt',map_location='cpu',mmap=True,weights_only=False);gb=torch.load(T/'four/first_gradients.pt',map_location='cpu',mmap=True,weights_only=False);a=b=dot=err=0.
for k,x in ga.items():
 x=x.double().flatten();y=gb[k].double().flatten();a+=float(x@x);b+=float(y@y);dot+=float(x@y);err+=float((x-y).square().sum())
grad_cos=dot/max((a*b)**.5,1e-30);grad_rel=(err/max(a,1e-30))**.5
clipa=rr['two'][0]['phases_by_rank'][0]['clip'];clipb=rr['four'][0]['phases_by_rank'][0]['clip'];cliprel=abs(clipa-clipb)/max(clipa,1e-20);lossdiff=abs(rr['two'][0]['loss']-rr['four'][0]['loss'])
time2=statistics.median(r['seconds'] for r in rr['two'][1:]);time4=statistics.median(r['seconds'] for r in rr['four'][1:]);mean4=statistics.mean(r['seconds'] for r in rr['four'][1:])
passed=same_restore and same_inputs and ranks_equal and grad_cos>=thresholds['gradient_cosine_min'] and grad_rel<=thresholds['gradient_relative_max'] and cliprel<=thresholds['preclip_norm_relative_max'] and update_cos>=thresholds['master_update_cosine_min'] and update_rel<=thresholds['master_update_relative_max'] and maximum<=thresholds['master_max_absolute'] and lossdiff<=thresholds['loss_absolute_max'] and time4<time2*thresholds['speed_ratio_max']
result=dict(passed=passed,kind='batch32_resource_transition',from_ranks=2,to_ranks=4,effective_batch=32,resume_checkpoint=parent['checkpoint'],resume_sha256=sha(parent['checkpoint']),recipe_identity=identity(cfg),engine_sha256=sha(D/'code/src/motion_training/engine.py'),batching_sha256=sha(D/'code/src/motion_training/batching.py'),parent_state_preserved_exact=same_restore,sample_hashes_exact=same_inputs,four_ranks_equal=ranks_equal,gradient_cosine=grad_cos,gradient_relative_l2=grad_rel,preclip_norm_relative=cliprel,master_update_cosine=update_cos,master_update_relative_l2=update_rel,master_max_absolute=maximum,loss_absolute_difference=lossdiff,two_seconds_median=time2,four_seconds_median=time4,four_seconds_mean=mean4,speedup=time2/time4,thresholds=thresholds,probe_weights_discarded=True,remaining_training_hours=(100000-parent['step'])*mean4/3600)
write(T/'GATE.json',result);print('TOPOLOGY_GATE',result,flush=True);assert passed,result
