"""Eight-rank accelerated112k continuation with real timed-goal teacher inputs."""
from common import *
import argparse,collections,contextlib,math,functools
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from motion_training import runtime,losses,distributed as parallel,engine,batching
from motion_training.acceleration import configure,state_tree
from g1_training_precision import prediction_loss_context
import pipeline_accel as fast
import lazy_plan, speed_backend as speed
sys.path.insert(0,str(ROOT/'acceleration_v3'))
import history_v3
import future_goal as future

class MixedLoss(nn.Module):
    def __init__(self,raw,c):super().__init__();self.raw=raw;self.client=c
    def forward(self,f):
        if not f.get('generated_history_state_ready',False):self.client.reset_memory()
        pred=self.raw(*(f[k] for k in losses.FORWARD_KEYS))
        with prediction_loss_context(self.raw,pred):value,info,_=fast.components_group(self.raw,f,pred)
        return value,info

def choose(pool,step,rank,cfg):
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],step,rank,pool['generation']]))
    families=[('legacy',4),('far',3),('arrival',1)]
    records=[]
    for kind,count in families:
        opts=[r for r in pool['labels'] if (r.get('pool_kind','legacy')=='legacy' if kind=='legacy' else
              r.get('pool_kind')=='future' and ((r['F']>0) if kind=='far' else r['F']==0))]
        assert opts,kind
        buckets=collections.defaultdict(list)
        for r in opts:buckets[r['batch_bucket']].append(r)
        keys=sorted(buckets)
        for j in range(count):
            key=keys[(step*8+rank*count+j)%len(keys)];rs=buckets[key]
            r=dict(rs[int(rng.integers(len(rs)))])
            r.update(base_id=r['id'],seed=cfg['seed']+step*512+rank*64+56+len(records),
                     id=f'{r["id"]}:s{step}:r{rank}:j{len(records)}')
            records.append(r)
    # End on a visible future sample so its projection also participates in the
    # final DDP synchronization. All preceding gradients remain accumulated.
    return records[:4]+records[7:]+records[4:7]

def gate(m,c,ck,cfg,pool,checkpoint_path):
    from motion_training import history
    raw=m.denoiser.model;raw.train();m.autoencoder.eval()
    options=[]
    for key,condition in [('legacy',lambda r:r.get('pool_kind','legacy')=='legacy'),
                          ('far',lambda r:r.get('pool_kind')=='future' and r['F']>0),
                          ('arrival',lambda r:r.get('pool_kind')=='future' and r['F']==0)]:
        candidates=[r for r in pool['labels'] if condition(r)]
        # Cover every available future extent across ranks. Only the equivalence
        # gate disables dropout, ensuring the future projection receives gradient.
        if key=='far':
            extents=sorted({r['F'] for r in candidates})
            extent=extents[parallel.rank()%len(extents)]
            candidates=[r for r in candidates if r['F']==extent]
        r=dict(candidates[parallel.rank()%len(candidates)],seed=cfg['seed']+parallel.rank()+73)
        options.append((key,r))
    checks=[]
    for key,r in options:
        runs=[];initial_rng=runtime.rng()
        for mode in ['reference','canonical']:
            runtime.restore_rng(initial_rng);raw.zero_grad(set_to_none=True)
            if mode=='reference':
                speed.install('reference',audit=True)
                if v5 is not None:v5.begin_update(m,c,[r],'current')
                if extra_graph is not None:extra_graph.enable(False)
                if encode_graph is not None:encode_graph.enable(False)
                if cond_cache is not None:cond_cache.begin([r],False,raw)
            else:
                history_v3.install('canonical',audit=True)
                if v5 is not None:v5.begin_update(m,c,[r],history_mode)
                if extra_graph is not None:extra_graph.enable(True)
                if encode_graph is not None:encode_graph.enable(True)
                if cond_cache is not None:cond_cache.begin([r],True,raw)
            configure(c,'reference')
            gatecfg=dict(cfg,text_dropout=0.,constraint_dropout=0.)
            tick=time.perf_counter();f,receipt=future.teacher_batch(m,c,r,gatecfg,ck['config'])
            torch.cuda.synchronize();prime=time.perf_counter()-tick
            fields=identity({k:f[k] for k in losses.FORWARD_KEYS}|dict(target=f['target'],prefix=f['decoder_history_hybrid'],supervised=f['supervised_motion']))
            state=identity(state_tree([c.controller.root_block._sequence_state,c.controller.body_block._sequence_state]))
            torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
            with torch.autocast('cuda',dtype=torch.bfloat16):v,_=MixedLoss(raw,c)(f)
            assert torch.isfinite(v);v.backward()
            grads={n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
            future_grad=sum(float(x.square().sum()) for n,x in grads.items() if 'future_constraints_proj' in n)
            runs.append(dict(loss=float(v),fields=fields,state=state,rng=identity(runtime.rng()),
                             grads=grads,future_grad_sq=future_grad,prime_seconds=prime,receipt=receipt))
            del v,f
            if cond_cache is not None:cond_cache.end()
            if v5 is not None:v5.end_update()
        ref,acc=runs;keys=set(ref['grads'])==set(acc['grads'])
        denominator=sum(float(x.square().sum()) for x in ref['grads'].values())
        delta=sum(float((ref['grads'][n]-acc['grads'][n]).square().sum()) for n in ref['grads']) if keys else math.inf
        row=dict(kind=key,rank=parallel.rank(),depth=r['prime_chunks'],F=r['F'],input_exact=ref['fields']==acc['fields'],
            history_state_exact=ref['state']==acc['state'],rng_exact=ref['rng']==acc['rng'],
            loss_delta=abs(ref['loss']-acc['loss']),gradient_relative_l2=math.sqrt(delta/max(denominator,1e-30)),
            future_projection_gradient_sq=acc['future_grad_sq'],
            reference_prime_seconds=ref['prime_seconds'],accelerated_prime_seconds=acc['prime_seconds'])
        row['passed']=bool(keys and row['input_exact'] and row['history_state_exact'] and row['rng_exact']
            and row['loss_delta']<=1e-7 and row['gradient_relative_l2']<=1e-6
            and (key!='far' or row['future_projection_gradient_sq']>0))
        checks.append(row);print('ACCEL_GATE',row,flush=True)
    raw.zero_grad(set_to_none=True)
    parts=parallel.gather(checks)
    result=dict(deterministic_comparison=torch.are_deterministic_algorithms_enabled(),passed=all(x['passed'] for rs in parts for x in rs),checks=[x for rs in parts for x in rs],
        checkpoint=checkpoint_path,checkpoint_sha256=sha(checkpoint_path),pool_sha256=sha(a.pool),world_size=parallel.world(),
        extra_graph_sha256=sha(ROOT/'acceleration_v5/extra_graph.py') if extra_graph is not None else None,
        condition_cache_sha256=sha(ROOT/'acceleration_v5/condition_cache.py') if cond_cache is not None else None,
        encode_graph_sha256=sha(ROOT/'acceleration_v5/encode_graph.py') if encode_graph is not None else None,
        history_v5_mode=history_mode,history_v5_sha256=sha(ROOT/'acceleration_v5/history_v5.py') if v5 is not None else None,
        current_gradient_path='reference PyTorch and existing Mamba; no new training kernel',
        fast_gt='unchanged frozen_serial_codec_graph_v2 batched GT with admitted original reductions',
        backend_sha256=sha(ROOT/'speed_backend.py'),history_v3_sha256=sha(ROOT/'acceleration_v3/history_v3.py'),
        teacher_builder_sha256=sha(ROOT/'future_goal.py'))
    if parallel.leader():write(ROOT/'gates/ACCELERATION_GATE.json',result)
    assert result['passed'],result
    history_v3.install('canonical',audit=False)

p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--pool',required=True)
p.add_argument('--start',type=int,default=0);p.add_argument('--until',type=int,required=True);p.add_argument('--gate-only',action='store_true')
a=p.parse_args()
if a.gate_only:
    # Identical unfused CUDA backward can differ between repeats. Compare both
    # paths deterministically; do not loosen tolerances or alter production.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
parallel.setup();cfg=read(ROOT/'config.json');pool=read(a.pool)
assert parallel.world()==8
# Exact parent lineage is verified after loading the complete checkpoint.
assert pool['current_student_history'] and pool['teacher_does_not_commit']
assert pool['qualified_future_far']>=16 and pool['qualified_future_arrival']>=4 and pool['qualified_legacy']>=32
manifest=ROOT/'train/data/MANIFEST.json'
if parallel.leader():lazy_plan.prepare(cfg,manifest.parent)
parallel.barrier();plan=lazy_plan.load(manifest)['plan']
m,c,ck=runtime.load(a.parent,True);raw=m.denoiser.model;configure(c,'reference')
if pool['checkpoint_sha256']!=sha(a.parent):
    # Saved maintenance boundaries reuse the unchanged active pool, exactly as
    # continuation within a generation does. Reject unrelated or edited pools.
    saved=ck['framework']
    assert a.start>0 and saved['stage_step']==a.start and saved['recipe_hash']==identity(cfg)
    assert saved['pool_path']==a.pool and saved['pool_hash']==sha(a.pool)
    assert saved['manifest_hash']==sha(manifest) and saved['world_size']==8
    assert len(set(saved['rank_state_hashes']))==1
v5=None;extra_graph=None;encode_graph=None;cond_cache=None;history_mode='current'
if (ROOT/'acceleration_v5/ADMISSION.json').exists():
    extra=read(ROOT/'acceleration_v5/ADMISSION.json')
    assert extra['passed'] and extra['backend_sha256']==sha(ROOT/'acceleration_v5/history_v5.py')
    history_mode=extra['mode'];assert history_mode in ['prefix','graph','prefix_graph']
    sys.path.insert(0,str(ROOT/'acceleration_v5'));import history_v5 as v5
    v5.install()
    if extra.get('extra_graph',False):
        assert extra['extra_graph_sha256']==sha(ROOT/'acceleration_v5/extra_graph.py')
        import extra_graph
        extra_graph.install(c)
    if extra.get('encode_graph',False):
        assert extra['encode_graph_sha256']==sha(ROOT/'acceleration_v5/encode_graph.py')
        import encode_graph
        encode_graph.install(c)
    if extra.get('condition_cache',False):
        assert extra['condition_cache_sha256']==sha(ROOT/'acceleration_v5/condition_cache.py')
        import condition_cache as cond_cache
        cond_cache.install(c)
if a.gate_only:
    from motion_training.io import Budget
    with parallel.timed(Budget(cfg['budget_root']), 'acceleration_gate:'+str(a.start)):
        gate(m,c,ck,cfg,pool,a.parent)
    parallel.barrier();raise SystemExit(0)
admission=read(ROOT/'gates/ACCELERATION_GATE.json')
assert admission['passed'] and admission['world_size']==8
assert admission['checkpoint_sha256']==sha(a.parent)
assert admission['pool_sha256']==sha(a.pool)
assert admission['history_v5_mode']==history_mode
if v5 is not None:assert admission['history_v5_sha256']==sha(ROOT/'acceleration_v5/history_v5.py')
if extra_graph is not None:assert admission['extra_graph_sha256']==sha(ROOT/'acceleration_v5/extra_graph.py')
if encode_graph is not None:assert admission['encode_graph_sha256']==sha(ROOT/'acceleration_v5/encode_graph.py')
if cond_cache is not None:assert admission['condition_cache_sha256']==sha(ROOT/'acceleration_v5/condition_cache.py')
assert admission['checkpoint']==a.parent
for key,name in [('backend_sha256','speed_backend.py'),('history_v3_sha256','acceleration_v3/history_v3.py'),('teacher_builder_sha256','future_goal.py')]:
    assert admission[key]==sha(ROOT/name)
history_v3.install('canonical',audit=False)
opt=runtime.optimizer(m,ck,cfg,resume=a.start>0)
if a.start==0:
    assert ck['step']==112000
    previous=copy_state(opt.training_schedule);opt.training_schedule=dict(spec=cfg['scheduler'],completed=0)
    if parallel.leader():write(ROOT/'OPTIMIZER_TRANSITION.json',dict(inherit_weights=True,inherit_FP32_masters=True,
        inherit_Adam=True,previous_schedule=previous,new_schedule=opt.training_schedule,parent=a.parent,parent_sha256=sha(a.parent)))
else:
    assert ck['framework']['stage_step']==a.start and ck['framework']['recipe_hash']==identity(cfg)
ck['_framework_global_start']=112000
ck['_history_acceleration']=dict(version=history_v3.VERSION+':v5:'+history_mode,gate=str(ROOT/'gates/ACCELERATION_GATE.json'))
ck['_resource_transition']=dict(physical_world_size=8,global_batch=512,GT_per_rank=56,teacher_per_rank=8,
    GT_batched=True,teacher_history_rebuilt_with_current_weights=True)
runtime.restore_rng(ck['rng_by_rank'][parallel.rank()%len(ck['rng_by_rank'])])
raw.train();m.autoencoder.eval()
module=DDP(MixedLoss(raw,c),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,
    find_unused_parameters=True,gradient_as_bucket_view=True)
budget=__import__('motion_training.io',fromlist=['Budget']).Budget(cfg['budget_root'])
if parallel.leader():write(ROOT/'train/EFFECTIVE_CONFIG.json',dict(recipe=cfg,parent=a.parent,parent_sha256=sha(a.parent),
    imports=runtime.import_sources(),stage_start=a.start,world_size=8,optimizer_inherited=True))
with parallel.timed(budget,'mixed_train:'+str(a.start)):
 for step in range(a.start+1,a.until+1):
    tick=time.perf_counter();opt.zero_grad(set_to_none=True);lr=runtime.set_update_lr(opt)
    audit=step in [1,10,a.start+1] or step%500==0
    tracked={n:p.detach().clone() for n,p in raw.named_parameters() if 'future_constraints_proj.weight' in n} if step==a.start+1 else {}
    rs=choose(pool,step,parallel.rank(),cfg)
    if v5 is not None:v5.begin_update(m,c,rs,history_mode)
    if extra_graph is not None:extra_graph.enable(True)
    if encode_graph is not None:encode_graph.enable(True)
    if cond_cache is not None:cond_cache.begin(rs,True,raw)
    records=plan.records_at(step-1,448);subset=records[parallel.rank()*56:(parallel.rank()+1)*56]
    pending=fast.submit_reads(c,subset,cfg,4);prepared=fast.materialize(pending)
    f,gt_receipts=fast.build_group(m,c,prepared,cfg,audit=audit,serial_codec=True)
    torch.manual_seed(subset[0]['seed']);torch.cuda.manual_seed_all(subset[0]['seed'])
    with module.no_sync():
        with torch.autocast('cuda',dtype=torch.bfloat16):v,info=module(f)
        assert torch.isfinite(v);(v*(8*56/512)).backward()
    gt_loss=float(v.detach());del v,f,prepared
    values=[];teacher_receipts=[]
    speed.set_audit(audit)
    for j,r in enumerate(rs):
        f,receipt=future.teacher_batch(m,c,r,cfg,ck['config'])
        torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
        sync=module.no_sync() if j<len(rs)-1 else contextlib.nullcontext()
        with sync:
            with torch.autocast('cuda',dtype=torch.bfloat16):v,info=module(f)
            assert torch.isfinite(v);(v*(8/512)).backward()
        values.append(float(v.detach()))
        if audit:teacher_receipts.append({k:receipt.get(k) for k in ['id','pool_kind','F','relative_goal_frame','F_visible_after_drop','F_before_drop','observed_scalars','text_dropped','condition_dropped','input_hash','noise_hash']})
        del v,f
    history_stats=v5.stats() if v5 is not None else {}
    history_stats['extra']=extra_graph.stats() if extra_graph is not None else {}
    history_stats['encoder']=encode_graph.stats() if encode_graph is not None else {}
    history_stats['conditioning']=cond_cache.stats() if cond_cache is not None else {}
    if cond_cache is not None:cond_cache.end()
    if v5 is not None:v5.end_update()
    norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['clip'],error_if_nonfinite=True)
    opt.step();opt.training_schedule['completed']+=1;torch.cuda.synchronize()
    delta={n:float((dict(raw.named_parameters())[n].detach()-p).float().norm()) for n,p in tracked.items()}
    elapsed=time.perf_counter()-tick;peak=torch.cuda.max_memory_allocated()/2**30
    part=dict(rank=parallel.rank(),GT_loss=gt_loss,teacher_loss=sum(values)/len(values),seconds=elapsed,
        peak_GiB=peak,history_stats=history_stats,GT_slots=56,teacher_slots=8,future_slots=4,legacy_slots=4,
        future_projection_update_norm=delta,teacher_receipts=teacher_receipts)
    allparts=parallel.gather(part)
    if parallel.leader():
        batching.fast_ticket(budget,'B',step=step,world_size=8,sample_slots=512)
        rec=dict(step=step,global_step=112000+step,world_size=8,global_batch=512,GT_slots=448,teacher_slots=64,
            future_teacher_slots=32,legacy_teacher_slots=32,pool=str(a.pool),pool_generation=pool['generation'],
            speed_backend=history_v3.VERSION+':v5:'+history_mode,lr=lr,loss=sum(x['GT_loss']*.875+x['teacher_loss']*.125 for x in allparts)/8,
            clip=float(norm),seconds=max(x['seconds'] for x in allparts),ranks=allparts)
        append(ROOT/'train/train.jsonl',rec);print('UPDATE',step,rec['loss'],rec['seconds'],flush=True)
        write(ROOT/'LIVE.json',dict(step=step,global_step=112000+step,until=a.until,global_batch=512,world_size=8,
            speed_backend=history_v3.VERSION+':v5:'+history_mode,seconds=rec['seconds'],peaks_GiB=[x['peak_GiB'] for x in allparts],time=time.time()))
    stop=False
    if parallel.leader() and step%10==0:
        used=sum(x['seconds']*x.get('devices',1) for x in __import__('motion_training.io',fromlist=['rows']).rows(ROOT/'budget/GPU_TIME.jsonl'))
        active=read(ROOT/'budget/ACTIVE.json')
        used+=sum((time.time()-x['start'])*x.get('devices',1) for x in active.values())
        stop=used>=read(ROOT/'budget/LIMITS.json')['gpu_hours']*.75*3600 or Path(cfg['boundary_request_path']).exists()
    stop=any(parallel.gather(stop)) if step%10==0 else False
    if step in cfg['savepoints'] or step==a.until or stop:
        engine.save(ROOT/f'train/checkpoint{step}.pt',m,ck,opt,cfg,step,manifest,a.pool)
        parallel.barrier()
        if parallel.leader():write(ROOT/'LATEST.json',dict(global_step=112000+step,stage_step=step,
            checkpoint=str(ROOT/f'train/checkpoint{step}.pt'),pool=str(a.pool),rank_state_hashes_verified=True))
    if stop:break
 parallel.barrier()
