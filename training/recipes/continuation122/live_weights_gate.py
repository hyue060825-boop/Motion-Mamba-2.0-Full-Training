from common import *
import argparse,ast,collections
import numpy as np
import torch
from motion_training import runtime,losses,history,distributed as parallel
from motion_training.acceleration import configure,state_tree
import speed_backend as speed
sys.path.insert(0,str(ROOT/'acceleration_v3'));import history_v3
sys.path.insert(0,str(ROOT/'acceleration_v5'));import history_v5 as v5
import future_goal as future
from budget_process import start_budget
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--pool',required=True);p.add_argument('--mode',required=True);p.add_argument('--extra',action='store_true');p.add_argument('--encoder',action='store_true');p.add_argument('--conditioning',action='store_true');a=p.parse_args()
parallel.setup();torch.use_deterministic_algorithms(True)
if parallel.leader():start_budget('v5_no_grad_live_weight_gate',devices=8)
cfg=read(ROOT/'config.json');pool=read(a.pool);m,c,ck=runtime.load(a.checkpoint,True);raw=m.denoiser.model
node=next(n for n in ast.parse((ROOT/'train_future.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='choose')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<choose>','exec'))
rs=choose(pool,1181,parallel.rank(),cfg)
v5.install();param=c.controller.root_block.sequence_stack.layers[0].mixer._mamba_wrapper.mamba3.in_proj.weight
params=[param]
extra=None
if a.extra:
 import extra_graph as extra
 extra.install(c)
 layer=c.controller.root_block.sequence_stack.layers[0]
 params.extend([layer.mlp[0].weight,layer.condition_modulation[-1].weight])
encoder=None
if a.encoder:
 import encode_graph as encoder
 encoder.install(c)
 params.append(c.controller.root_block.embed_text.weight)
cond=None
if a.conditioning:
 import condition_cache as cond
 cond.install(c)
before=[p.detach().clone() for p in params];pointers=[p.data_ptr() for p in params];checks=[];hashes=[]
def digest():
 tree=state_tree([c.controller.root_block._sequence_state,c.controller.body_block._sequence_state])
 for bi,block in enumerate([c.controller.root_block,c.controller.body_block]):
  for i,(s,layer) in enumerate(zip(block._sequence_state.layers,block.sequence_stack.layers)):
   assert s.attention_cache.projection_version==layer.mixer.attention._projection_version()
   assert s.attention_cache.source_kernel_id==id(layer.mixer.attention)
   tree[bi]['layers'][i]['attention_cache']=dict(tree[bi]['layers'][i]['attention_cache'])
   tree[bi]['layers'][i]['attention_cache']['projection_version']=tuple(0 for _ in s.attention_cache.projection_version)
 return identity(tree)
try:
 for epoch in range(2):
  if epoch:
   with torch.no_grad():
    for p in params:p.add_(0.001)
   assert [p.data_ptr() for p in params]==pointers
   assert all(not torch.equal(p,b) for p,b in zip(params,before))
  run=[]
  for mode in ['current',a.mode]:
   history_v3.install('canonical',audit=True);configure(c,'reference');v5.begin_update(m,c,rs,mode)
   if extra is not None:extra.enable(mode!='current')
   if encoder is not None:encoder.enable(mode!='current')
   if cond is not None:cond.begin(rs,mode!='current',raw)
   states=[]
   for r in rs:
    f,_=future.teacher_batch(m,c,r,cfg,ck['config'])
    states.append(digest());del f
   run.append((states,v5.stats(),extra.stats() if extra is not None else {},encoder.stats() if encoder is not None else {}));v5.end_update()
   if cond is not None:cond.end()
  assert run[0][0]==run[1][0],(epoch,run[0][0],run[1][0])
  hashes.append(run[0][0]);checks.append(dict(epoch=epoch,states_exact=True,stats=run[1][1],extra_stats=run[1][2],encoder_stats=run[1][3],parameter_storage_unchanged=True))
 assert hashes[0]!=hashes[1],'Probe mutation did not alter computed history state'
 if a.mode in ['graph','prefix_graph']:
  assert checks[1]['stats'].get('graph_replays',0)>0
  assert checks[1]['stats'].get('graph_captures',0)==0,'Updated values unexpectedly required graph recapture'
 if extra is not None:
  assert checks[1]['extra_stats'].get('replays',0)>0
  assert checks[1]['extra_stats'].get('captures',0)==0
 if encoder is not None:
  assert checks[1]['encoder_stats'].get('replays',0)>0
  assert checks[1]['encoder_stats'].get('captures',0)==0
finally:
 with torch.no_grad():
  for p,b in zip(params,before):p.copy_(b)
 v5.end_update()
 if cond is not None:cond.end()
 assert all(torch.equal(p,b) for p,b in zip(params,before))
rows=parallel.gather(dict(rank=parallel.rank(),passed=True,checks=checks,new_weight_values_change_history=True,no_optimizer_updates=True))
if parallel.leader():write(ROOT/'acceleration_v5/LIVE_WEIGHT_GATE.json',dict(passed=all(x['passed'] for x in rows),rows=rows,mode=a.mode,extra=a.extra,encoder=a.encoder,conditioning=a.conditioning,checkpoint=a.checkpoint,checkpoint_sha256=sha(a.checkpoint),backend_sha256=sha(ROOT/'acceleration_v5/history_v5.py')))
parallel.barrier()
