"""Frozen easy-v3 paired inference. No training and no future-GT trajectory."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,hashlib,inspect,importlib.util,traceback,argparse,fcntl
import numpy as np
R=Path(artifact_path('unified_interactive_easy_v3_hiphi_20261021_32d9db1045'))
H=Path(artifact_path('humanml3d_section1_20261020_c1c76adb30'))
REPAIR=Path(artifact_path('section1_masked_kernel_repair_20261020_47cd64f873'))
MODELS=['native100', 'native110', 'native117', 'native122', 'minimal', 'ardy', 'dip', 'dart', 'fd2', 'omg', 'cmdm', 'motionstreamer', 'motion_mamba2_official']
def rd(p):return json.loads(Path(p).read_text())
def wr(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def verify():
 for n,h in rd(R/'FROZEN.json')['hashes'].items():assert sha(R/n)==h,('frozen_changed',n)
 for k in MODELS:
  v=rd(R/'CHECKPOINTS.json')[k];s=Path(v['checkpoint']).stat();f=v.get('frozen_file_stat')
  if f:assert s.st_size==f['size'] and s.st_mtime_ns==f['mtime_ns'],('checkpoint_changed',k)
def setup(method,out):
 info=rd(R/'CHECKPOINTS.json')[method];d=(R/'adapters'/method) if method=='minimal' else H/'models'/({'ardy':'transformer_reference','ardy_official':'upstream_transformer'}.get(method,method))
 os.environ.update(EVAL_TRAINING_ROOT=info['training_root'],EVAL_MODEL=method,EVAL_LABEL=method,
  EVAL_ARCH='native' if method.startswith('native') else method,HF_HOME=artifact_path('huggingface_ed9ba7d805'),
  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',RUN_PHASE='easyv3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 if not method.startswith('native'):os.environ['MOTION_MAMBA2_SEQUENCE_MODEL']='transformer'
 sys.path.insert(0,info['training_root'])
 if (Path(info['training_root'])/'common.py').exists():import common
 sys.path[:0]=[str(d/'sampler'),str(d)]
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out.mkdir(parents=True,exist_ok=True)
 from checkpoint_loader import load
 model,client,ck,arch=load(argparse.Namespace(checkpoint=info['checkpoint'],device='cuda:0',stats_dir=info['stats_dir'],output=str(out/'result.json'),mode='minimal' if method=='minimal' else 'hybrid' if method.startswith('native') else 'original',diffusion_steps=10,cfg_weight_text=2.,cfg_weight_constraint=2.))
 # The new event ledger supersedes old proxy disk tracing, including in timing.
 import util
 oldwrite=util.write
 def quiet_trace(p,x):
  if Path(p).name=='INFERENCE_TRACE.json':return
  return oldwrite(p,x)
 util.write=quiet_trace
 m=model.model if hasattr(model,'model') and hasattr(model.model,'_generate_window') else model
 if hasattr(model,'trace_path'):model.trace_path=out/'INFERENCE_TRACE.json'
 assert rd(out/'LOAD.json')['all_loaded_denoiser_weights_verified']
 if method.startswith('native') or method=='minimal':
  ctl=client.controller
  ctl.configure_rollout_diagnostics(enabled=False,periodic_reset_windows=0)
  ctl.configure_content_memory_tensor_trace(enabled=False)
  ctl.configure_mamba_output_decay(start_windows=0,half_life_windows=0.,min_scale=0.)
  ctl.configure_state_intervention('carry');ctl.configure_content_memory_intervention('normal',permutation=None)
 return model,m,client
def backend(method,m,client,enabled):
 if method.startswith('native'):
  from motion_training import acceleration
  acceleration.configure(client,'optimized' if enabled else 'reference')
  ctl=client.controller
  if not hasattr(ctl,'_easy_mask_module'):
   package=type(ctl.root_block.sequence_stack).__module__.rsplit('.',1)[0]
   spec=importlib.util.spec_from_file_location(package+'.easy_masked_graph',REPAIR/'masked_graph_candidate.py')
   module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   ctl._easy_mask_module=module;ctl._easy_original_masks={};ctl._easy_graphs={}
  for name,b in [('root',ctl.root_block),('body',ctl.body_block)]:
   stack=b.sequence_stack;stack.repair_crop_suffix=False
   if enabled:
    if name not in ctl._easy_graphs:ctl._easy_graphs[name]=ctl._easy_mask_module.ReadOnlyStackGraphs()
    stack._inference_graphs=ctl._easy_graphs[name]
   for layer in stack.layers:
    att=layer.mixer.attention
    if id(att) not in ctl._easy_original_masks:ctl._easy_original_masks[id(att)]=att._build_allowed_mask
    att._build_allowed_mask=ctl._easy_mask_module.captured_mask if enabled else ctl._easy_original_masks[id(att)]
  return 'mask_aware_exact_graph' if enabled else 'reference'
 raw=m.denoiser.model
 if hasattr(raw,'_set_exact_core_acceleration'):
  raw._set_exact_core_acceleration(enabled)
  return 'exact_external_core' if enabled else 'reference'
 return 'reference'
def graph_stats(method,client):
 if not method.startswith('native'):return {}
 from motion_training.acceleration import stats
 v=stats(client)
 return {k:dict(builds=len(x['build_events']),replays=x['replays'],fallbacks=x['fallbacks']) for k,x in v.items()}
def schedule(task):
 p=rd(R/'PLAN.json');names=[s for s,n in p['tasks'][task]['stages'] for _ in range(n)]
 return names,[p['prompts'][s] for s in names]
def run(method,model,m,client,case,task,seed,assets,features,gate=False,timing=False):
 import torch,random
 names,texts=schedule(task);N=len(names);T=24+52*N
 idx=assets['case_ids'].tolist().index(case['case_id']);initial=assets['initial_physical'][idx]
 hist=torch.tensor(initial,device='cuda',dtype=torch.float32)[None]
 prefix=m.motion_rep.normalize(hist)
 # Construct no full target pose and no future GT. Only eight authorized values.
 coords=assets['goal_coordinate_indices'][idx];vals=assets['goal_physical_values'][idx]
 sparse=hist.new_zeros(1,1,414);sparse[:,:,coords]=torch.as_tensor(vals,device='cuda')
 norm=m.motion_rep.normalize(sparse)
 obs=hist.new_zeros(1,T,414);mask=obs.clone();mask[:,-1,coords]=1;obs[:,-1,coords]=norm[:,0,coords]
 assert int(mask.count_nonzero())==8 and torch.count_nonzero(obs.masked_select(mask==0))==0
 assert np.array_equal(coords,np.asarray(case['goal_input_coordinate_indices']))
 cache={p:torch.tensor(v,device='cuda',dtype=torch.float32)[None,None] for p,v in features.items()}
 tm=torch.ones(1,1,device='cuda',dtype=torch.bool);first=cache[texts[0]]
 orig=m._generate_window;sig=inspect.signature(orig);events=[];states=[]
 def state():
  from motion_training import acceleration,runtime
  from motion_training.io import identity
  c=client.controller
  return dict(state=identity(acceleration.state_tree(dict(root=c.root_block._sequence_state,body=c.body_block._sequence_state,accepted=c._accepted_commits,completed=c._completed_windows,next_chunk=c._next_chunk_id))),RNG=identity(runtime.rng()))
 def dynamic(*a,**kw):
  b=sig.bind(*a,**kw);b.apply_defaults();i=len(events);assert i<N
  assert b.arguments.get('target_motion') is None
  b.arguments['text_feat']=cache[texts[i]];b.arguments['text_pad_mask']=tm
  mm=b.arguments['motion_mask'];oo=b.arguments['observed_motion']
  # Host transfers, hashing and file I/O are excluded from warmed latency.
  if not timing:
   count=int(mm.count_nonzero());assert count in [0,8],('goal_lost_or_duplicated',count)
   assert torch.count_nonzero(oo.masked_select(mm==0))==0
   ev=dict(window=i+1,stage=names[i],prompt=texts[i],observed_scalars=count,timeline_frames=mm.shape[1],target_motion_is_None=True)
  else:ev=dict(window=i+1,stage=names[i])
  events.append(ev)
  return orig(*b.args,**b.kwargs)
 m._generate_window=dynamic
 finish=None
 if (method.startswith('native') or method=='minimal') and gate:
  finish=client.controller._finish_generated_commit
  def completed(*a,**kw):
   v=finish(*a,**kw);states.append(state());return v
  client.controller._finish_generated_commit=completed
 random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 before=graph_stats(method,client);torch.cuda.synchronize();t=time.perf_counter()
 try:
  with torch.inference_mode():
   if method.startswith('native') or method=='minimal':
    client.reset_memory();hp=torch.ones(1,24,device='cuda',dtype=torch.bool)
    hybrid,_=m.hybrid.get_hybrid_motion_from_explicit(prefix,hp.sum(1),hp)
    heading=torch.atan2(hist[:,-1,4],hist[:,-1,3])
    client.controller.prime_accepted_history(hybrid,text_feat=first,text_pad_mask=tm,first_heading_angle=heading)
    from g1_motion_contract import generate_with_committed_decode
    pred=generate_with_committed_decode(m,[texts[0]],N*52,decoder_prefix=hybrid[:,-52//m.num_frames_per_token:],num_denoising_steps=10,pad_mask=torch.ones(1,N*52,device='cuda',dtype=torch.bool),first_heading_angle=heading,motion_mask=mask[:,24:],observed_motion=obs[:,24:],cfg_weight=(2.,2.),crop_history_length=0,text_feat=first,text_pad_mask=tm,progress_bar=lambda x:x)
   else:
    if hasattr(model,'calls'):model.calls=[]
    pred=model([texts[0]],T,num_denoising_steps=10,pad_mask=torch.ones(1,T,device='cuda',dtype=torch.bool),first_heading_angle=None,motion_mask=mask,observed_motion=obs,cfg_weight=(2.,2.),crop_history_length=None,init_history_sequence=prefix,text_feat=first,text_pad_mask=tm,progress_bar=lambda x:x)[:,24:]
   # Include accepted physical decode for all methods, exclude FK scoring.
   physical=m.motion_rep.unnormalize(pred)
   torch.cuda.synchronize();elapsed=time.perf_counter()-t
   geom=m.motion_rep.inverse(pred,is_normalized=True)
 except BaseException:
  raise
 finally:
  m._generate_window=orig
  if finish is not None:client.controller._finish_generated_commit=finish
 assert len(events)==N
 if not timing and method not in ['dip','dart']:assert all(e['observed_scalars']==8 for e in events),events
 a=physical.float().cpu().numpy()[0];j=geom['posed_joints'].float().cpu().numpy()[0];root=geom['root_positions'].float().cpu().numpy()[0]
 assert np.isfinite(a).all() and np.isfinite(j).all() and a.shape==(N*52,414)
 rng=hashlib.sha256(repr(random.getstate()).encode()+repr(np.random.get_state()).encode()+torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
 audit=dict(initial_sha256=digest(initial),sparse_physical_goal_sha256=digest(vals),indices=coords.tolist(),noise_seed=seed,mask_sha256=digest(mask.cpu().numpy()),normalized_observed_sha256=digest(obs.cpu().numpy()),no_hidden_GT=True,final_RNG=rng,per_window_state_RNG=states,events=events,physical_sha256=digest(a),joint_sha256=digest(j),root_sha256=digest(root),elapsed_s=elapsed,graph_before=before,graph_after=graph_stats(method,client))
 return dict(physical=a,joints=j,root=root),audit
def save_trial(dest,geo,audit,case,task,seed):
 dest.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest/'geometry.npz',**geo)
 wr(dest/'INPUT_AND_EVENTS.json',audit)
 wr(dest/'COMPLETE.json',dict(case_id=case['case_id'],source_family=case['source_id'],task=task,seed=seed,geometry_sha256=sha(dest/'geometry.npz'),audit_sha256=sha(dest/'INPUT_AND_EVENTS.json'),time=time.time()))
def worker(method,task):
 import torch
 verify();out=R/'models'/method/task;out.mkdir(parents=True,exist_ok=True)
 lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wr(out/'WORKER_PID.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),cmdline=sys.argv,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),time=time.time()))
 model,m,client=setup(method,out)
 assets=dict(np.load(R/'INPUT_ASSETS.npz'));allcases=[json.loads(s) for s in (R/'CASES.jsonl').open()]
 cases=[c for c in allcases if c['split']=='locked_test'];dev=[c for c in allcases if c['split']=='development'][:4]
 names=rd(R/'TEXT_INDEX.json')['prompts'];features=dict(zip(names,np.load(R/'TEXT_FEATURES.npy')))
 gates=[];gatefile=out/'ACCELERATION_GATE.json'
 if not gatefile.exists():
  for c in dev:
   backend(method,m,client,False);a,aa=run(method,model,m,client,c,task,c['noise_seeds'][0],assets,features,gate=True)
   candidate=backend(method,m,client,True);b,bb=run(method,model,m,client,c,task,c['noise_seeds'][0],assets,features,gate=True)
   checks={k:bool(np.array_equal(a[k],b[k])) for k in a}
   checks.update(RNG=aa['final_RNG']==bb['final_RNG'],state=aa['per_window_state_RNG']==bb['per_window_state_RNG'],events=aa['events']==bb['events'])
   gates.append(dict(end=c['selected_end_effector'],case_id=c['case_id'],candidate=candidate,checks=checks,max_delta={k:float(np.max(np.abs(a[k]-b[k]))) for k in a}))
   wr(out/'GATE_PROGRESS.json',dict(rows=gates,time=time.time()))
   assert all(checks.values()),('exact_acceleration_gate_failed',method,task,gates[-1])
  wr(gatefile,dict(passed=True,rows=gates,source_sha256=sha(__file__),mask_graph_sha256=sha(REPAIR/'masked_graph_candidate.py'),time=time.time()))
 else:
  gate=rd(gatefile);assert gate['passed'] and gate['source_sha256']==sha(__file__),'Changed source requires a new gate'
 active_backend=backend(method,m,client,True)
 # Two complete independent warmups per duration before B1 measurement.
 latencyfile=out/'LATENCY.json'
 if not latencyfile.exists():
  for c in cases[:2]:run(method,model,m,client,c,task,c['noise_seeds'][0],assets,features,timing=True)
  times=[]
  for c in cases[:32]:
   _,a=run(method,model,m,client,c,task,c['noise_seeds'][0],assets,features,timing=True)
   if method.startswith('native'):
    assert all(a['graph_after'][k]['builds']==a['graph_before'][k]['builds'] for k in a['graph_before']),('timed_graph_build',c['case_id'])
   times.append(dict(case_id=c['case_id'],seed=c['noise_seeds'][0],elapsed_s=a['elapsed_s'],graph_before=a['graph_before'],graph_after=a['graph_after']))
  W=len(schedule(task)[0]);v=np.asarray([x['elapsed_s'] for x in times])*1000/W
  wr(latencyfile,dict(backend=active_backend,batch=1,warmup_sequences=2,timed_sequences=32,windows=W,mean_ms_window=float(v.mean()),median_ms_window=float(np.median(v)),p95_ms_window=float(np.quantile(v,.95)),scope='current prompt embeddings cached; complete rollout model+physical decode+commit, no file hashing or FK scoring; amortized per52frame window',rows=times,time=time.time()))
 total=len(cases)*3;done=0
 for c in cases:
  for si,seed in enumerate(c['noise_seeds']):
   dest=out/'trials'/f'{cases.index(c):03d}_s{si}'
   if (dest/'COMPLETE.json').exists():
    comp=rd(dest/'COMPLETE.json');assert comp['seed']==seed and sha(dest/'geometry.npz')==comp['geometry_sha256']
   else:
    geo,audit=run(method,model,m,client,c,task,seed,assets,features)
    save_trial(dest,geo,audit,c,task,seed)
   done+=1
   wr(out/'LIVE.json',dict(stage='ACTUAL_FIXED_TEST_GENERATION',completed=done,total=total,case=c['case_id'],time=time.time()))
   if done%24==0:print(method,task,done,total,flush=True)
 wr(out/'GENERATION_COMPLETE.json',dict(passed=True,rollouts=done,independent_families=128,backend=active_backend,task=task,checkpoint=rd(R/'CHECKPOINTS.json')[method],time=time.time()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--task',required=True);a=ap.parse_args()
 try:worker(a.model,a.task)
 except BaseException:
  wr(R/'models'/a.model/a.task/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
