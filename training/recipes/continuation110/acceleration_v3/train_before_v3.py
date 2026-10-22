from common import *
import argparse,contextlib,collections,functools,time,math
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from motion_training import runtime,losses,batching,engine,distributed as parallel
from motion_training.history import HistoryContext
from motion_training.io import rows
from motion_training.acceleration import configure
from g1_online_controls import conditions
from motion_mamba2.models.motion_model import translate_normalized_root_motion
from g1_motion_contract import CONTRACT,decode_training_prediction
from g1_training_precision import prediction_loss_context
import pipeline_accel as fast
import lazy_plan

@functools.lru_cache(maxsize=64)
def load_label(path):return torch.load(path,map_location='cpu',weights_only=False)

def merge(values):
    if torch.is_tensor(values[0]):
        return torch.stack(values) if values[0].ndim==0 else torch.cat(values,0)
    assert all(v==values[0] for v in values), 'Nonuniform history metadata'
    return values[0]

def teacher_group(m,c,records,cfg):
    items=[copy_state(load_label(r['label']),m.device) for r in records]
    depth=len(items[0]['chunks']);assert all(len(i['chunks'])==depth for i in items)
    chunks=[]
    for j in range(depth):
        tokens=torch.cat([i['chunks'][j][0] for i in items])
        kw={k:merge([i['chunks'][j][1][k] for i in items]) for k in items[0]['chunks'][j][1]}
        chunks.append((tokens,kw))
    physical=torch.cat([i['physical'] for i in items]);text=torch.cat([i['target']['text'] for i in items])
    HistoryContext(chunks,physical,records[0]['collector_checkpoint_sha256']).prime(m,c,text)
    target=torch.cat([i['target']['target_latent'] for i in items]);prefix=torch.cat([i['target']['decoder_prefix'] for i in items]);device=m.device
    b=len(items);valid=torch.ones(b,52,device=device,dtype=torch.bool);tokens=torch.ones(b,13,device=device,dtype=torch.bool);z=torch.zeros(b,device=device,dtype=torch.long)
    obs=[];masks=[];td=[];cd=[];ts=[];noises=[]
    for r,i in zip(records,items):
        o,mask=conditions(m,i['target']['control'],i['translation'])
        root=translate_normalized_root_motion(m.motion_rep.extract_root(o),-i['translation'],m.motion_rep)
        o=m.motion_rep.concat_root_body(root,m.motion_rep.extract_body(o))*mask
        rng=np.random.default_rng(r['seed']);text_drop=bool(rng.random()<cfg['text_dropout']);condition_drop=bool(rng.random()<cfg['constraint_dropout'])
        if condition_drop:o.zero_();mask.zero_()
        obs.append(o);masks.append(mask);td.append(text_drop);cd.append(condition_drop)
        gen=torch.Generator(device=device).manual_seed(r['seed'])
        ts.append(torch.randint(m.diffusion.num_base_steps,(1,),generator=gen,device=device))
        noises.append(torch.randn((1,13,148),generator=gen,device=device,dtype=target.dtype))
    dropped=torch.tensor(td,device=device,dtype=torch.bool);text.masked_fill_(dropped[:,None,None],0)
    t=torch.cat(ts);noise=torch.cat(noises)
    m.diffusion.calc_diffusion_vars(torch.arange(m.diffusion.num_base_steps,device=device))
    f=dict(x=m.diffusion.q_sample(target,t,noise),target=target,decoder_history_hybrid=prefix,history_len=z,generation_len=z+52,future_len=z,
        history_mask=~valid,generation_mask=valid,future_mask=~valid,history_token_mask=~tokens,generation_token_mask=tokens,future_token_mask=~tokens,
        text=text,text_mask=(~dropped)[:,None],heading=torch.cat([i['heading'] for i in items]),motion_mask=torch.cat(masks).float(),observed=torch.cat(obs),
        pad_mask=valid,lengths=z+52,timestep=t,noise=noise,generated_history_state_ready=True,sequential_cache=True,motion_contract=CONTRACT,hybrid_converter=m.hybrid)
    with torch.no_grad():f['supervised_motion']=decode_training_prediction(target,f).detach()
    return f

class MixedLoss(nn.Module):
    def __init__(self,raw,c):super().__init__();self.raw=raw;self.client=c
    def forward(self,f):
        if not f.get('generated_history_state_ready',False):self.client.reset_memory()
        pred=self.raw(*(f[k] for k in losses.FORWARD_KEYS))
        with prediction_loss_context(self.raw,pred):value,details,_=fast.components_group(self.raw,f,pred)
        return value,details

def choose(pool,step,rank,count,cfg):
    groups=collections.defaultdict(list)
    for r in pool['labels']:groups[r['batch_bucket']].append(r)
    # Equal stage/depth buckets; all examples in a group have compatible state shapes.
    # Eight logical sample lanes stay fixed even when four physical ranks each
    # accumulate two lanes. Topology does not change the chosen history labels.
    keys=sorted(groups);key=keys[(step*8+rank)%len(keys)];options=groups[key]
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],step,rank,pool['generation']]))
    order=[]
    while len(order)<count:order.extend(rng.permutation(len(options)).tolist())
    return [dict(options[i],base_id=options[i]['id'],id=f'{options[i]["id"]}:s{step}:r{rank}:l{j}',seed=cfg['seed']+step*512+rank*64+j) for j,i in enumerate(order[:count])]

def gate(m,c,ck,cfg,pool):
    from motion_training.history import teacher_batch
    raw=m.denoiser.model;raw.eval();configure(c,'reference')
    rs=choose(pool,1,parallel.rank(),2,cfg)
    # Test same primed memory/input targets and all trainable parameter gradients.
    def run_group():
        raw.zero_grad(set_to_none=True);value=0.
        for r in rs:
            f,_=teacher_batch(m,c,r,cfg,ck['config'])
            with torch.autocast('cuda',dtype=torch.bfloat16):v,_=MixedLoss(raw,c)(f)
            (v/len(rs)).backward();value+=float(v)/len(rs)
        return value,{n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
    value,grads=run_group();raw.zero_grad(set_to_none=True);reference=0.
    for r in rs:
        f,_=teacher_batch(m,c,r,cfg,ck['config'])
        with torch.autocast('cuda',dtype=torch.bfloat16):v,_,_=losses.compute(raw,c,f,cfg['loss'])
        (v/len(rs)).backward();reference+=float(v)/len(rs)
    errors=[];squares=0.;delta=0.
    for n,p in raw.named_parameters():
        if p.grad is None:assert n not in grads;continue
        assert n in grads
        x=p.grad.detach().float().cpu();squares+=float(x.square().sum());delta+=float((x-grads[n]).square().sum());errors.append(float((x-grads[n]).abs().max()))
    rel=math.sqrt(delta/max(squares,1e-30));loss_delta=abs(value-reference)
    passed=loss_delta<=.003 and rel<=.02
    result=dict(passed=passed,rank=parallel.rank(),loss_group=value,loss_serial=reference,loss_abs_delta=loss_delta,gradient_relative_l2=rel,gradient_max_abs=max(errors),scope='Two independent history samples accumulated serially; loss wrapper vs original losses.compute, BF16; merged-history batching rejected',tolerances=dict(loss_abs=.003,gradient_relative_l2=.02))
    write(ROOT/'gates'/f'teacher_rank{parallel.rank()}.json',result)
    allchecks=parallel.gather(result)
    if parallel.leader():write(ROOT/'gates/TEACHER_BATCH_GATE.json',dict(passed=all(x['passed'] for x in allchecks),checks=allchecks))
    assert all(x['passed'] for x in allchecks),allchecks
    raw.zero_grad(set_to_none=True);raw.train()

p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--pool',required=True);p.add_argument('--until',type=int,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--gate-only',action='store_true');a=p.parse_args()
parallel.setup();cfg=read(ROOT/'config.json');pool=read(a.pool)
parent_matches_pool=pool['checkpoint_sha256']==sha(a.parent)
assert pool['current_student_history'] and pool['teacher_does_not_commit'] and pool['labels']
manifest=ROOT/'train/data/MANIFEST.json'
if parallel.leader():lazy_plan.prepare(cfg,manifest.parent)
parallel.barrier();plan=lazy_plan.load(manifest)['plan']
m,c,ck=runtime.load(a.parent,True);raw=m.denoiser.model;configure(c,'reference')
if not parent_matches_pool:
    assert a.start>0 and ck['framework']['stage_step']==a.start
    assert ck['framework']['pool_hash']==sha(a.pool),'Within-generation resume must keep exact same history pool'
speed_mode='reference'
if (ROOT/'SPEED_ADMISSION.json').exists() and not a.gate_only:
    import speed_backend
    admission=read(ROOT/'SPEED_ADMISSION.json')
    assert admission['passed'] and admission['cpu_inputs_exact']
    assert admission['backend_sha256']==sha(ROOT/'speed_backend.py')
    speed_mode=admission['mode'];speed_backend.install(speed_mode,audit=False)
if a.gate_only:
    gate(m,c,ck,cfg,pool);parallel.barrier();raise SystemExit(0)
assert read(ROOT/'gates/TEACHER_BATCH_GATE.json')['passed']
if a.start==0:
    previous_cfg=read(OLD/'config.json');opt=runtime.optimizer(m,ck,previous_cfg)
    # Preserve Adam moments and exact FP32 master weights; register only a new schedule.
    old_schedule=copy_state(opt.training_schedule);opt.training_schedule=dict(spec=cfg['scheduler'],completed=0)
    if parallel.leader():write(ROOT/'OPTIMIZER_TRANSITION.json',dict(inherited=True,masters_and_Adam_moments_unchanged=True,old_schedule=old_schedule,new_schedule=opt.training_schedule,new_stage_not_exact_resume=True))
else:
    assert ck['framework']['stage_step']==a.start and ck['framework']['recipe_hash']==identity(cfg)
    opt=runtime.optimizer(m,ck,cfg,resume=True)
ck['_framework_global_start']=100000
assert parallel.world() in (4,8) or a.gate_only
runtime.restore_rng(ck['rng_by_rank'][parallel.rank()%len(ck['rng_by_rank'])])
ck['_resource_transition']=dict(physical_world_size=parallel.world(),logical_sample_lanes=8,global_batch=512,logical_microbatch=64,gradient_scaling='physical_world_size /512 times each sample sum',sample_and_dropout_seeds_unchanged=True)
raw.train();m.autoencoder.eval();module=DDP(MixedLoss(raw,c),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
budget=__import__('motion_training.io',fromlist=['Budget']).Budget(cfg['budget_root'])
if a.start==0:engine.save(ROOT/'train/checkpoint0.pt',m,ck,opt,cfg,0,manifest,a.pool)
with parallel.timed(budget,'mixed_train:'+str(a.start)):
 for step in range(a.start+1,a.until+1):
    tick=time.perf_counter();opt.zero_grad(set_to_none=True);lr=runtime.set_update_lr(opt)
    stage=next(s for s in cfg['interaction_schedule'] if s['from']<=step<=s['to'])
    cap=read(ROOT/'RATIO_CAP.json')
    teacher=min(stage['teacher'],cap['teacher_slots']);gt=512-teacher
    assert gt+teacher==512 and gt%8==teacher%8==0
    records=plan.records_at(step-1,gt);width=gt//8
    lanes=list(range(parallel.rank()*8//parallel.world(),(parallel.rank()+1)*8//parallel.world()))
    local=[r for lane in lanes for r in records[lane*width:(lane+1)*width]]
    audit=step%1000==0 or step==1 or step==a.start+1
    pending=fast.submit_reads(c,local,cfg,4 if speed_mode!='reference' else 0);gt_values=[]
    for micro,lane in enumerate(lanes):
        subset=local[micro*width:(micro+1)*width]
        prepared=fast.materialize(pending[micro*width:(micro+1)*width]);f,receipts=fast.build_group(m,c,prepared,cfg,audit=audit,serial_codec=True)
        torch.manual_seed(subset[0]['seed']);torch.cuda.manual_seed_all(subset[0]['seed'])
        with module.no_sync():
            with torch.autocast('cuda',dtype=torch.bfloat16):v,info=module(f)
            assert torch.isfinite(v);(v*(parallel.world()*width/512)).backward()
        gt_values.append(float(v.detach()));del v,f,prepared
    gt_loss=sum(gt_values)/len(gt_values)
    rs=[r for lane in lanes for r in choose(pool,step,lane,teacher//8,cfg)];teacher_values=[]
    from motion_training.history import teacher_batch
    if speed_mode!='reference':speed_backend.set_audit(audit)
    for j,r in enumerate(rs):
        f,_=teacher_batch(m,c,r,cfg,ck['config'])
        torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
        sync=module.no_sync() if j<len(rs)-1 else contextlib.nullcontext()
        with sync:
            with torch.autocast('cuda',dtype=torch.bfloat16):v,info=module(f)
            assert torch.isfinite(v);(v*(parallel.world()/512)).backward()
        teacher_values.append(float(v.detach()));del v,f
    teacher_loss=sum(teacher_values)/len(teacher_values)
    clip=torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['clip'],error_if_nonfinite=True)
    opt.step();opt.training_schedule['completed']+=1;torch.cuda.synchronize()
    elapsed=time.perf_counter()-tick
    peak=torch.cuda.max_memory_allocated()/2**30
    part=dict(rank=parallel.rank(),logical_lanes=lanes,GT_loss=gt_loss,teacher_loss=teacher_loss,clip=float(clip),seconds=elapsed,peak_GiB=peak,GT_slots=len(local),teacher_slots=len(rs),teacher_ids=[r['base_id'] for r in rs])
    allparts=parallel.gather(part)
    if parallel.leader():
        batching.fast_ticket(budget,'B',step=step,world_size=parallel.world(),sample_slots=512)
        rec=dict(step=step,global_step=100000+step,world_size=parallel.world(),global_batch=512,GT_slots=gt,teacher_slots=teacher,ratio_admission=cap,pool=str(a.pool),pool_generation=pool['generation'],speed_backend=speed_mode,lr=lr,loss=sum(x['GT_loss']*gt/512+x['teacher_loss']*teacher/512 for x in allparts)/parallel.world(),seconds=max(x['seconds'] for x in allparts),ranks=allparts)
        append(ROOT/'train/train.jsonl',rec)
        print('UPDATE',step,rec['loss'],rec['seconds'],'GT',gt,'teacher',teacher,flush=True)
        write(ROOT/'LIVE.json',dict(step=step,global_step=100000+step,until=a.until,global_batch=512,world_size=parallel.world(),speed_backend=speed_mode,seconds=rec['seconds'],peaks_GiB=[x['peak_GiB'] for x in allparts],GT_slots=gt,teacher_slots=teacher,time=time.time()))
    # Reserve one quarter of the total GPU budget for collection and independent
    # acceptance. All ranks stop only after a complete update, then save/evaluate.
    stop=False
    if parallel.leader() and step%10==0:
        used=sum(r['seconds']*r.get('devices',1) for r in rows(budget.root/'GPU_TIME.jsonl'))
        active=read(budget.root/'ACTIVE.json')
        used+=sum((time.time()-r['start'])*r.get('devices',1) for r in active.values())
        stop=used+parallel.world()*120>=read(budget.root/'LIMITS.json')['gpu_hours']*.75*3600
        if Path(cfg['boundary_request_path']).exists():
            stop=True
            write(ROOT/'BOUNDARY_SAVE.json',dict(step=step,request=read(cfg['boundary_request_path']),full_update=True))
        if stop:write(ROOT/'BUDGET_STOP.json',dict(step=step,gpu_hours_used=used/3600,cap_gpu_hours=read(budget.root/'LIMITS.json')['gpu_hours'],full_update=True))
    stop=any(parallel.gather(stop)) if step%10==0 else False
    if step%1000==0 or step==a.until or step in (1,10,a.start+1,a.start+10) or stop:
        engine.save(ROOT/f'train/checkpoint{step}.pt',m,ck,opt,cfg,step,manifest,a.pool)
        parallel.barrier()
        if parallel.leader():write(ROOT/'LATEST.json',dict(global_step=100000+step,stage_step=step,checkpoint=str(ROOT/f'train/checkpoint{step}.pt'),pool=str(a.pool)))
    if stop:break
 parallel.barrier()
