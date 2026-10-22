import os,sys,time,json,random,copy
from pathlib import Path
import numpy as np,torch
CACHE=None;META=None;ROWS=[];PENDING={};ACTIVE=False;CURRENT=None

def counts(client):
 if not hasattr(client,'controller'):return None
 from motion_training import acceleration
 v=acceleration.stats(client)
 return dict(replays=sum(x['replays'] for x in v.values()),fallbacks=sum(x['fallbacks'] for x in v.values()),builds=sum(len(x['build_events']) for x in v.values()))
def before(client):
 global ACTIVE
 v=counts(client);ACTIVE=True;return v

def after(client,prediction,prior,elapsed,size,start,rep,warmup):
 global ACTIVE
 ACTIVE=False
 for path,data in list(PENDING.items()):_WRITE(path,data)
 PENDING.clear()
 row=dict(start=start,batch=size,repetition=rep,warmup=warmup,elapsed_seconds=elapsed,seconds_per_sample=elapsed/size)
 if prior is not None:
  now=counts(client);row['graph_delta']={k:now[k]-prior[k] for k in prior}
 if os.environ.get('RUN_PHASE','').startswith('gate'):
  from motion_training import acceleration,runtime
  from motion_training.io import identity
  ctl=client.controller
  state=dict(root=ctl.root_block._sequence_state,body=ctl.body_block._sequence_state,accepted=ctl._accepted_commits,completed=ctl._completed_windows,next_chunk=ctl._next_chunk_id)
  row['hashes']=dict(prediction=identity(prediction),state=identity(acceleration.state_tree(state)),RNG=identity(runtime.rng()))
 ROWS.append(row)

def buffered_write(p,d):
 if ACTIVE:PENDING[str(p)]=d
 else:_WRITE(p,d)
_WRITE=None

def prepare_entry(entry):
 global _WRITE
 import util
 if _WRITE is None:_WRITE=util.write;util.write=buffered_write
 import checkpoint_loader as loader_module
 if not hasattr(loader_module,'_kernel_original_load'):loader_module._kernel_original_load=loader_module.load
 original=loader_module._kernel_original_load
 def load(args):
  global CACHE,META,CURRENT
  CURRENT=Path(args.output).parent
  if CACHE is None:
   CACHE=original(args)
   META=json.loads((CURRENT/'LOAD.json').read_text())
  model,client,ck,arch=CACHE
  native=os.environ['EVAL_MODEL'].startswith('native')
  backend=os.environ.get('EVAL_BACKEND','reference')
  if native:
   from motion_training import acceleration
   acceleration.configure(client,backend)
  elif backend!='reference':raise ValueError('No Native-specific kernel may be attached to another architecture')
  random.seed(2026101952);np.random.seed(2026101952)
  metadata=dict(META,actual_backend=backend,kernel219_method_enabled=native and backend=='optimized',inference_backend='SHARED serial-CFG Graph and exact fusion' if native and backend=='optimized' else META.get('inference_backend','native architecture reference'),timing_batch=int(os.environ.get('EVAL_BATCH_SIZE','2')),timing_phase=os.environ['RUN_PHASE'],source_weights_unchanged=True)
  _WRITE(CURRENT/'LOAD.json',metadata)
  return CACHE
 loader_module.load=load
 entry.load=load

def flush(dest):
 _WRITE(Path(dest)/'TIMING_DETAIL.json',dict(rows=ROWS,IO_writes_deferred_outside_timer=True,whole_sequence_timing=True,torch_threads=torch.get_num_threads(),GPU=torch.cuda.get_device_name(),backend=os.environ['EVAL_BACKEND']))
 ROWS.clear()
