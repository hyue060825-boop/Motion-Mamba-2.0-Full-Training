from common import *
import ast,math,statistics
import torch
from motion_training import runtime,history
from motion_training.acceleration import configure,state_tree
import history_v3 as candidate
tree=ast.parse((ROOT/'train.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'train.py'),'exec'),globals())
cfg=read(ROOT/'config.json');pool=read(ROOT/'pools/g1/POOL.json')
checkpoint=Path(read(ROOT/'LATEST.json')['checkpoint'])
m,c,ck=runtime.load(checkpoint,True);raw=m.denoiser.model;raw.train();m.autoencoder.eval()
opt=runtime.optimizer(m,ck,cfg,resume=True)
initial=copy_state(raw.state_dict());initial_opt=copy_state(opt.state_dict());initial_rng=runtime.rng()
dest=ROOT/'acceleration_v3';checks=[]
def run(mode,r):
 candidate.install(mode,audit=False);configure(c,'reference');raw.zero_grad(set_to_none=True)
 torch.cuda.synchronize();start=time.perf_counter()
 f,_=history.teacher_batch(m,c,r,cfg,ck['config'])
 torch.cuda.synchronize();prime=time.perf_counter()-start
 state=identity(state_tree([c.controller.root_block._sequence_state,c.controller.body_block._sequence_state]))
 fields=identity({k:f[k] for k in losses.FORWARD_KEYS}|dict(target=f['target'],prefix=f['decoder_history_hybrid'],supervised=f['supervised_motion']))
 torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
 with torch.autocast('cuda',dtype=torch.bfloat16):v,_=MixedLoss(raw,c)(f)
 v.backward();torch.cuda.synchronize()
 return dict(prime=prime,state=state,fields=fields,value=float(v),rng=identity(runtime.rng()),grads={n:p.grad.detach().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None})
samples=[]
for depth in (1,6,12,18,24,29):
 r=dict(min(pool['labels'],key=lambda r:abs(r['prime_chunks']-depth)),seed=cfg['seed']+depth);samples.append(r)
for i,r in enumerate(samples):
 ref=run('dense',r);other=run('canonical',r)
 keys=ref['grads'].keys()==other['grads'].keys()
 exact=keys and all(torch.equal(ref['grads'][n],other['grads'][n]) for n in ref['grads'])
 row=dict(depth=r['prime_chunks'],label=r.get('base_id',r.get('id')),state_exact=ref['state']==other['state'],input_exact=ref['fields']==other['fields'],rng_exact=ref['rng']==other['rng'],gradient_exact=exact,loss_abs=abs(ref['value']-other['value']),dense_prime_seconds=ref['prime'],canonical_prime_seconds=other['prime'],post_update_checks=min(i,2))
 row['passed']=row['state_exact'] and row['input_exact'] and row['rng_exact'] and exact and row['loss_abs']==0
 checks.append(row);print('V3_CHECK',row,flush=True)
 write(dest/'SINGLE_GATE.json',dict(passed=all(x['passed'] for x in checks),checks=checks,complete=False))
 assert row['passed'],row
 if i<2:
  torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['clip'],error_if_nonfinite=True);opt.step()
raw.load_state_dict(initial);opt.load_state_dict(initial_opt);runtime.restore_rng(initial_rng)
# An exception must clear this fresh training prefix, without touching Adam.
candidate.install('canonical');r=samples[-1]
item=candidate.packed_item(r['label'],m.device)
ctx=history.HistoryContext(item['chunks'],item['physical'],r['collector_checkpoint_sha256'])
original=raw.body_model.sequence_stack.forward;calls=[0]
def fail(*args,**kwargs):
 calls[0]+=1
 if calls[0]==2:raise RuntimeError('V3 intentional second-chunk fault')
 return original(*args,**kwargs)
before=identity(opt.state_dict());raw.body_model.sequence_stack.forward=fail
try:
 try:ctx.prime(m,c,item['target']['text']);raise AssertionError('fault not raised')
 except RuntimeError as e:assert 'intentional' in str(e)
finally:raw.body_model.sequence_stack.forward=original
fault=dict(state_cleared=all(b._sequence_state is None for b in (c.controller.root_block,c.controller.body_block)),optimizer_unchanged=before==identity(opt.state_dict()),training_mode_restored=raw.training)
assert all(fault.values()),fault
timings=[]
for rep in range(4):
 for mode in (('dense','canonical') if rep%2==0 else ('canonical','dense')):
  r=samples[-1];candidate.install(mode);torch.cuda.synchronize();start=time.perf_counter()
  history.teacher_batch(m,c,r,cfg,ck['config']);torch.cuda.synchronize()
  timings.append(dict(mode=mode,repeat=rep,seconds=time.perf_counter()-start))
write(dest/'SINGLE_GATE.json',dict(passed=True,complete=True,checks=checks,fault=fault,prime_timings=timings,cache_stats=candidate.stats(),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_sha256=sha(dest/'history_v3.py'),discarded_updates=2,scope='B1 actual teacher-history state, inputs, loss and gradients; not full training speed'))
print('V3_SINGLE_PASS',fault,timings,flush=True)
