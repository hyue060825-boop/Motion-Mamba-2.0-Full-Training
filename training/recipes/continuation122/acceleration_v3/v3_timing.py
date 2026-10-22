from common import *
import ast,statistics
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from motion_training import runtime,history,distributed as parallel
from motion_training.acceleration import configure
import history_v3 as candidate
for path in (ROOT/'train.py',ROOT/'acceleration_v3/v3_full_gate.py'):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())
parallel.setup();assert parallel.world()==8
cfg=read(ROOT/'config.json');pool=read(ROOT/'pools/g1/POOL.json');dest=ROOT/'acceleration_v3'
assert read(dest/'EIGHT_RANK_GATE.json')['passed'] and read(dest/'EIGHT_RANK_GATE.json')['complete']
m,c,ck=runtime.load(ROOT/'train/checkpoint3490.pt',True);raw=m.denoiser.model;raw.train();m.autoencoder.eval()
opt=runtime.optimizer(m,ck,cfg,resume=True);before=copy_state(raw.state_dict());before_opt=copy_state(opt.state_dict());before_schedule=copy_state(opt.training_schedule);rng=runtime.rng()
module=DDP(MixedLoss(raw,c),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
plan=lazy_plan.load(ROOT/'train/data/MANIFEST.json')['plan']
for mode in ('dense','canonical'):restore();update(mode,3491,False)
rows=[]
for rep in range(2):
 for mode in (('dense','canonical') if rep%2==0 else ('canonical','dense')):
  restore();r=update(mode,3491,False);r['repeat']=rep;allrows=parallel.gather(r)
  if parallel.leader():rows.extend(allrows);print('V3_WARM_TIMING',rep,mode,max(x['seconds'] for x in allrows),flush=True)
if parallel.leader():
 dense=[max(r['seconds'] for r in rows if r['repeat']==i and r['mode']=='dense') for i in range(2)]
 canonical=[max(r['seconds'] for r in rows if r['repeat']==i and r['mode']=='canonical') for i in range(2)]
 ratios=[a/b for a,b in zip(dense,canonical)]
 write(dest/'WARM_TIMING.json',dict(rows=rows,dense_median_seconds=statistics.median(dense),canonical_median_seconds=statistics.median(canonical),paired_median_speedup=statistics.median(ratios),pairs=ratios,scope='Eight-rank real global512 update including Adam. Both paths warm on the same actual update3491; fixed inputs and restored weights/Adam; two alternating timing pairs; excludes diagnostic hashing; limited scope, production continuation remains to be measured.',discarded_updates=6))
 budget=__import__('motion_training.io',fromlist=['Budget']).Budget(cfg['budget_root'])
 for i in range(6):batching.fast_ticket(budget,'checks',discarded_optimizer_probe='v3_warmed_timing_'+str(i),checkpoint_written=False)
 print('V3_WARM_DONE',dense,canonical,ratios,flush=True)
parallel.barrier()
