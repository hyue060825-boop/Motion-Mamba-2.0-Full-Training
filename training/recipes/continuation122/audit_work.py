from common import *
import ast,collections,statistics
import torch
import future_goal as future
import speed_backend as speed
from motion_training import runtime,losses,distributed as parallel,engine,batching
import numpy as np
cfg=read(ROOT/'config.json');pool=read(ROOT/'pools/g1/POOL.json')
tree=ast.parse((ROOT/'train_future.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<choose>','exec'))
cache={};labels={}
for r in pool['labels']:
 path=r['label']
 if path not in cache:
  item=torch.load(path,map_location='cpu',weights_only=False)
  cache[path]=[identity(chunk) for chunk in item['chunks']]
  labels[path]=dict(depth=len(item['chunks']),keys=[list(chunk[1]) for chunk in item['chunks'][:1]])
rows=[]
for step in range(1201,1251):
 per_rank=[choose(pool,step,rank,cfg) for rank in range(8)]
 depths=[sum(r['prime_chunks'] for r in records) for records in per_rank]
 hits=[]
 for records in per_rank:
  seen=set();hit=0
  for r in records:
   chain=cache[r['label']]
   for i in range(1,len(chain)+1):
    key=tuple(chain[:i])
    if key in seen:hit+=1
    seen.add(key)
  hits.append(hit)
 # Balance within legacy/far/arrival slots, retain exact64 records and seeds.
 balanced=[[] for _ in range(8)];costs=[0]*8
 for kind,count in [('far',3),('legacy',4),('arrival',1)]:
  candidates=[r for records in per_rank for r in records if (r.get('pool_kind','legacy')=='legacy' if kind=='legacy' else r.get('pool_kind')=='future' and (r['F']>0 if kind=='far' else r['F']==0))]
  slots=[0]*8
  for r in sorted(candidates,key=lambda r:r['prime_chunks'],reverse=True):
   rank=min((k for k in range(8) if slots[k]<count),key=lambda k:(costs[k],k))
   balanced[rank].append(r);costs[rank]+=r['prime_chunks'];slots[rank]+=1
 rows.append(dict(step=step,depths=depths,within_rank_reusable_chunks=hits,balanced_depths=costs))
result=dict(pool=str(ROOT/'pools/g1/POOL.json'),rows=rows,median_max_history_chunks=statistics.median(max(x['depths']) for x in rows),median_balanced_max_chunks=statistics.median(max(x['balanced_depths']) for x in rows),total_reusable_chunks=sum(sum(x['within_rank_reusable_chunks']) for x in rows),total_history_chunks=sum(sum(x['depths']) for x in rows),sample_structure=next(iter(labels.values())))
write(ROOT/'acceleration_v5/WORK_AUDIT.json',result);print({k:v for k,v in result.items() if k!='rows'})
