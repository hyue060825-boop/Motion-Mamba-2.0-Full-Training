from motion_mamba2.paths import artifact_path
import json,sys,time,gc
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'pipeline_accel_20261007'
sys.path.insert(0,str(R));from common import *
import torch
from motion_training import runtime,batching,losses
from lazy_plan import Plan
import pipeline_accel as acc
cp=read(S/'BOUNDARY.json')['checkpoint'];m,c,ck=runtime.load(cp,True);raw=m.denoiser.model;raw.train();m.autoencoder.eval()
cfg=read(R/'config.json');plan=Plan(read(R/'MANIFEST_TEMPLATE.json'));results=[]
def stats(x,y):
    d=(x.detach().float()-y.detach().float()).flatten();xx=x.detach().float().flatten()
    return dict(exact=torch.equal(x,y),max_abs=float(d.abs().max()) if d.numel() else 0,
        relative_l2=float(d.norm()/xx.norm().clamp_min(1e-20)))
layouts=[(0,0),(52,0),(196,0),(0,196),(52,144),(144,52)]
for hf in layouts:
    records=plan.records_at(63000+len(results),64,layout=hf)
    items=[]
    for r in records:
        f,_=batching.corrected_build(m,c,r,cfg,audit=False);items.append(f)
    reference=batching.collate(items)
    prepared=acc.materialize(acc.submit_reads(c,records,cfg,4));candidate,_=acc.build_group(m,c,prepared,cfg,audit=False)
    fields={};passed=True
    for k,x in reference.items():
        if not torch.is_tensor(x):continue
        y=candidate[k];assert x.dtype==y.dtype and x.shape==y.shape,(k,x.shape,y.shape)
        fields[k]=stats(x,y)
        # Targets, noise, conditioning and masks cannot change for speed.
        if not fields[k]['exact']:passed=False
    if reference.get('decoder_history_hybrid') is not None:
        fields['decoder_history_hybrid']=stats(reference['decoder_history_hybrid'],candidate['decoder_history_hybrid'])
        passed=passed and fields['decoder_history_hybrid']['exact']
    print('FIELDS',hf,passed,{k:v for k,v in fields.items() if not v['exact']},flush=True)
    # Keep inputs identical here so decoder/loss equivalence is isolated.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        c.reset_memory();prediction=raw(*(reference[k] for k in losses.FORWARD_KEYS))
    prediction=prediction.detach().requires_grad_(True)
    from g1_training_precision import prediction_loss_context
    with prediction_loss_context(raw,prediction):
        v0,info0=batching.independent_components(raw,reference,prediction)
        g0=torch.autograd.grad(v0,prediction)[0];v0=float(v0.detach());del info0
        v1,info1,terms=acc.components_group(raw,reference,prediction)
        g1=torch.autograd.grad(v1,prediction)[0];v1=float(v1.detach())
    grad=stats(g0,g1);lossrel=abs(v1-v0)/max(abs(v0),1e-8)
    row=dict(H=hf[0],F=hf[1],batch=64,exact_inputs=passed,fields=fields,loss_reference=v0,loss_candidate=v1,
        loss_relative_difference=lossrel,prediction_gradient=grad,
        loss_gradient_pass=lossrel<1e-4 and grad['relative_l2']<.005,
        future_visible=sum(p['visible'] for p in prepared),text_drop=sum(p['td'] for p in prepared),constraint_drop=sum(p['cd'] for p in prepared))
    results.append(row);write(S/'INPUT_LOSS_GATE.json',dict(passed=all(x['exact_inputs'] and x['loss_gradient_pass'] for x in results),rows=results,checkpoint=cp,complete=False))
    print('LOSS',hf,v0,v1,grad,flush=True)
    del items,reference,candidate,prediction,g0,g1,v1,info1,terms;gc.collect();torch.cuda.empty_cache()
write(S/'INPUT_LOSS_GATE.json',dict(passed=all(x['exact_inputs'] and x['loss_gradient_pass'] for x in results),rows=results,checkpoint=cp,complete=True))
print('COMPLETE',flush=True)
