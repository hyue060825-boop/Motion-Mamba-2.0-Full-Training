from common import *
import ast,math,torch
from motion_training import runtime,losses
from motion_training.acceleration import configure
import speed_backend as speed
sys.path.insert(0,str(ROOT/'acceleration_v3'))
import history_v3,future_goal as future
nodes=[n for n in ast.parse((ROOT/'train_future.py').read_text()).body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'train_definitions','exec'),globals())
cfg=read(ROOT/'config.json');pool=read(ROOT/'pools/g0/POOL.json')
m,c,ck=runtime.load(cfg['parent'],True);raw=m.denoiser.model;raw.train();m.autoencoder.eval()
opts=[r for r in pool['labels'] if r.get('pool_kind')=='future' and r['F']>0]
r=dict(opts[7],seed=cfg['seed']+7+73);initial=runtime.rng();runs=[];refgrad=None
for mode in ['reference','reference','canonical','canonical','reference']:
 runtime.restore_rng(initial);raw.zero_grad(set_to_none=True)
 speed.install('reference',audit=True) if mode=='reference' else history_v3.install('canonical',audit=True)
 configure(c,'reference');f,_=future.teacher_batch(m,c,r,dict(cfg,text_dropout=0.,constraint_dropout=0.),ck['config'])
 torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
 with torch.autocast('cuda',dtype=torch.bfloat16):
  pred=raw(*(f[k] for k in losses.FORWARD_KEYS))
 from g1_training_precision import prediction_loss_context
 with prediction_loss_context(raw,pred):v,_,_=fast.components_group(raw,f,pred)
 phash=identity(pred);v.backward()
 grads={n:p.grad.float().detach().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
 if refgrad is None:refgrad=grads;rel=0.;bad=[]
 else:
  num=sum(float((grads[n]-refgrad[n]).square().sum()) for n in grads);den=sum(float(x.square().sum()) for x in refgrad.values());rel=math.sqrt(num/max(den,1e-30))
  bad=sorted([(float((grads[n]-refgrad[n]).abs().max()),n) for n in grads],reverse=True)[:6]
 row=dict(mode=mode,loss=float(v),prediction_hash=phash,relative_gradient_l2=rel,worst_parameters=bad)
 runs.append(row);print(row,flush=True)
write(ROOT/'gates/BACKWARD_REPEAT_DIAGNOSTIC.json',dict(runs=runs,record=r['id'],F=r['F'],optimizer_updates=0))
