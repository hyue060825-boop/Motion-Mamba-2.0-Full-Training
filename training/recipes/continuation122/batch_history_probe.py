from common import *
import argparse,collections,copy,statistics
import torch
from motion_training import runtime,history
from motion_training.acceleration import configure,state_tree
sys.path.insert(0,str(ROOT/'acceleration_v3'));import history_v3
from budget_process import start_budget
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--pool',required=True);a=p.parse_args()
start_budget('v5_equal_depth_batch_history_probe',devices=1)
torch.use_deterministic_algorithms(True)
m,c,ck=runtime.load(a.checkpoint,True);raw=m.denoiser.model
pool=read(a.pool);buckets=collections.defaultdict(list)
for r in pool['labels']:buckets[r['prime_chunks']].append(r)
depth=max(k for k,v in buckets.items() if len(v)>=2)
records=buckets[depth][:2];items=[torch.load(r['label'],map_location=m.device,weights_only=False) for r in records]
history_v3.install('canonical',audit=True);trees=[]
for r,item in zip(records,items):
 context=history.HistoryContext(item['chunks'],item['physical'],r['collector_checkpoint_sha256'])
 history_v3.prime(context,m,c,item['target']['text'])
 trees.append(copy.deepcopy(state_tree([c.controller.root_block._sequence_state,c.controller.body_block._sequence_state])))
chunks=[]
for chunks_i in zip(*(item['chunks'] for item in items)):
 tokens=torch.cat([x[0] for x in chunks_i],dim=0);kw={}
 for key in chunks_i[0][1]:
  values=[x[1][key] for x in chunks_i]
  if torch.is_tensor(values[0]):kw[key]=torch.cat(values,dim=0)
  else:assert all(v==values[0] for v in values);kw[key]=values[0]
 chunks.append((tokens,kw))
physical=torch.cat([item['physical'] for item in items],dim=0)
context=history.HistoryContext(chunks,physical,records[0]['collector_checkpoint_sha256'])
text=torch.cat([item['target']['text'] for item in items],dim=0)
history_v3.prime(context,m,c,text)
batched=state_tree([c.controller.root_block._sequence_state,c.controller.body_block._sequence_state])
def slice_row(x,row):
 if torch.is_tensor(x):return x[row:row+1].clone() if x.ndim and x.shape[0]==2 else x.clone()
 if isinstance(x,dict):return {k:(1 if k=='batch_size' else slice_row(v,row)) for k,v in x.items()}
 if isinstance(x,list):return [slice_row(v,row) for v in x]
 if isinstance(x,tuple):return tuple(slice_row(v,row) for v in x)
 return x
def errors(a,b,path=''):
 if torch.is_tensor(a):
  assert torch.is_tensor(b) and a.shape==b.shape,(path,a.shape,b.shape)
  if a.dtype in [torch.bool,torch.int32,torch.int64]:return [] if torch.equal(a,b) else [(path,'nonfloat_mismatch')]
  delta=float((a.float()-b.float()).abs().max())
  return [] if torch.equal(a,b) else [(path,delta)]
 if isinstance(a,dict):return [e for k in a for e in errors(a[k],b[k],path+'/'+str(k))]
 if isinstance(a,(tuple,list)):return [e for i,(x,y) in enumerate(zip(a,b)) for e in errors(x,y,path+'/'+str(i))]
 return [] if a==b else [(path,'metadata_mismatch')]
rows=[]
for i,ref in enumerate(trees):
 candidate=slice_row(batched,i);diff=errors(ref,candidate)
 rows.append(dict(row=i,exact=identity(ref)==identity(candidate),differences=diff[:20],difference_count=len(diff)))
result=dict(passed=all(x['exact'] for x in rows),rows=rows,depth=depth,batch=2,checkpoint=a.checkpoint,scope='Equal-depth canonical history only; no padding, no optimizer; state equality required before considering full-update admission')
write(ROOT/'acceleration_v5/BATCH_HISTORY_FEASIBILITY.json',result);print(result)
