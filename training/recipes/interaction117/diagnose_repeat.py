"""Zero-update repeated backwards; separates reference variability from history changes."""
from common import *
import ast,math,argparse,traceback
import torch
from motion_training import runtime,losses,history,distributed as parallel
from motion_training.acceleration import configure,state_tree
import speed_backend as speed
sys.path.insert(0,str(ROOT/'acceleration_v3'))
import history_v3
import future_goal as future
import pipeline_accel as fast
from g1_training_precision import prediction_loss_context
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--pool',required=True);a=p.parse_args()
parallel.setup();cfg=read(ROOT/'config.json');pool=read(a.pool)
from budget_process import start_budget
if parallel.leader():start_budget("v4_repeated_backward_diagnostic",devices=parallel.world())
m,c,ck=runtime.load(a.checkpoint,True);raw=m.denoiser.model;raw.train();m.autoencoder.eval()
refs=[r for r in pool['labels'] if r.get('pool_kind')=='future' and r['F']>0]
r=dict(refs[parallel.rank()%len(refs)],seed=cfg['seed']+parallel.rank()+73)
initial=runtime.rng();original=speed.ORIGINAL_PRIME;previous_det=torch.are_deterministic_algorithms_enabled();results=[]
def execute(mode,det):
    runtime.restore_rng(initial);raw.zero_grad(set_to_none=True)
    torch.use_deterministic_algorithms(det)
    if mode=='reference':
        speed.ORIGINAL_PRIME=original;speed.install('reference',audit=True)
    else:
        history_v3.install('canonical',audit=True);speed.ORIGINAL_PRIME=history_v3.prime
    configure(c,'reference')
    f,_=future.teacher_batch(m,c,r,dict(cfg,text_dropout=0.,constraint_dropout=0.),ck['config'])
    ih=identity({k:f[k] for k in losses.FORWARD_KEYS}|dict(target=f['target'],prefix=f['decoder_history_hybrid'],supervised=f['supervised_motion']))
    sh=identity(state_tree([c.controller.root_block._sequence_state,c.controller.body_block._sequence_state]))
    torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pred=raw(*(f[k] for k in losses.FORWARD_KEYS))
        with prediction_loss_context(raw,pred):v,_,_=fast.components_group(raw,f,pred)
    v.backward();torch.cuda.synchronize()
    grads={n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
    out=dict(input=ih,state=sh,prediction=identity(pred),loss=float(v),rng=identity(runtime.rng()),grads=grads)
    del f,pred,v;return out

def compare(x,y):
    assert set(x['grads'])==set(y['grads'])
    errors={n:float((x['grads'][n]-y['grads'][n]).square().sum()) for n in x['grads']}
    den=sum(float(g.square().sum()) for g in x['grads'].values())
    return dict(input_exact=x['input']==y['input'],state_exact=x['state']==y['state'],prediction_exact=x['prediction']==y['prediction'],rng_exact=x['rng']==y['rng'],loss_delta=abs(x['loss']-y['loss']),gradient_relative_l2=math.sqrt(sum(errors.values())/max(den,1e-30)),largest_gradient_contributors=sorted(errors.items(),key=lambda z:z[1],reverse=True)[:8])
try:
 for det in [False,True]:
    try:
        execute('reference',det) # shape/library warm-up, gradients discarded
        runs=[execute('reference',det),execute('reference',det),execute('canonical',det)]
        row=dict(rank=parallel.rank(),F=r['F'],id=r['id'],deterministic=det,reference_repeat=compare(runs[0],runs[1]),canonical_compare=compare(runs[1],runs[2]))
        del runs
    except Exception:
        row=dict(rank=parallel.rank(),F=r['F'],id=r['id'],deterministic=det,error=traceback.format_exc())
    results.extend(parallel.gather(row));print('REPEAT_DIAG',row,flush=True)
finally:
 speed.ORIGINAL_PRIME=original;torch.use_deterministic_algorithms(previous_det);raw.zero_grad(set_to_none=True)
if parallel.leader():write(ROOT/'acceleration_v4/REPEAT_DIAGNOSTIC.json',dict(rows=results,checkpoint=a.checkpoint,pool=a.pool,optimizer_updates=0,production_modified=False))
parallel.barrier()
