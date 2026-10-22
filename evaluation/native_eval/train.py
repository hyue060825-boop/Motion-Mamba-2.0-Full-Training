import collections
import gc
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .common import *
from .schema import canonicalize
from .model import DualEncoder,contrastive,reconstruction_loss
from .text import TextCache
from .metrics import candidate_pools,retrieval

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def panel(domain,split):
    p=ROOT/'panels'/f'{domain}_{split}.json'
    if p.exists():
        cached=read(p)
        if 'candidate_pools' not in cached:
            chosen=cached['records'];cfg=read(CODE/'config.json')
            pools={str(s):candidate_pools(chosen,s) for s in cfg['panel_seeds']}
            try:matched=candidate_pools(chosen,cfg['panel_seeds'][0],True)
            except ValueError:matched=None
            cached.update(candidate_pools=pools,length_matched_pools=matched,fid_window='52 frames at fixed center for diagnostic only',panel_sha256=identity([chosen,pools,matched]))
            write(p,cached)
        return cached
    cfg=read(CODE/'config.json');rr=list(rows(ROOT/'data'/f'{domain}_{split}.jsonl'))
    chosen=[];fam=set();caption=set()
    for r in sorted(rr,key=lambda r:identity(['panel-v1',r['sample_id']])):
        options=[c for c in r['captions'] if c not in caption]
        if r['source_family_id'] in fam or not options:continue
        x=dict(r,caption=options[0]);chosen.append(x);fam.add(r['source_family_id']);caption.add(options[0])
        if len(chosen)>=cfg['validation_families']:break
    chosen=sorted(chosen,key=lambda r:r['sample_id'])
    pools={str(s):candidate_pools(chosen,s) for s in cfg['panel_seeds']}
    try:matched=candidate_pools(chosen,cfg['panel_seeds'][0],True)
    except ValueError:matched=None
    result=dict(records=chosen,candidate_pools=pools,length_matched_pools=matched,
                full_sequence_scope='complete source-provided semantic segments, not full-sequence FID release',
                fid_window='52 frames at floor((frames-52)/2), one window per family',
                role=split,version='v2',panel_sha256=identity([chosen,pools,matched]))
    write(p,result);return result

class MotionDataset(Dataset):
    def __init__(self,records,text,stats,epoch=0,seed=0,training=False):
        self.records=records;self.text_cache=text;self.mean=stats['mean'];self.std=stats['std']
        self.epoch=epoch;self.seed=seed;self.training=training;self.variant=read(CODE/'config.json').get('variant','R0')
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=dict(self.records[i]);a=np.load(r['feature_path'],mmap_mode='r',allow_pickle=False)
        rng=np.random.default_rng(int(identity([self.seed,self.epoch,r['sample_id']])[:16],16))
        caption=r.get('caption') or r['captions'][int(rng.integers(len(r['captions'])))]
        r['caption']=caption;a=canonicalize(a)
        return (a-self.mean)/self.std,self.text_cache[caption],r

def collate(items):
    lengths=torch.tensor([len(v[0]) for v in items]);x=torch.zeros(len(items),int(lengths.max()),414)
    for i,(a,_,_) in enumerate(items):x[i,:len(a)]=torch.from_numpy(a)
    return x,lengths,torch.tensor(np.stack([v[1] for v in items])),[v[2] for v in items]

def epoch_records(rr,seed,epoch):
    rng=np.random.default_rng(seed+epoch*100003);families=collections.defaultdict(list)
    for r in rr:families[r['source_family_id']].append(r)
    selected=[items[int(rng.integers(len(items)))] for _,items in sorted(families.items())]
    rng.shuffle(selected)
    # Local length buckets reduce padding, without fixed ordering of entire dataset.
    result=[]
    for start in range(0,len(selected),1024):
        bucket=sorted(selected[start:start+1024],key=lambda r:r['frames'])
        chunks=[bucket[i:i+32] for i in range(0,len(bucket),32)];rng.shuffle(chunks)
        for c in chunks:result.extend(c)
    return result

@torch.inference_mode()
def encode(model,records,text,stats,mode='full',perturb=None):
    model.eval();ordered=sorted(records,key=lambda r:(r['frames'],r['sample_id']))
    features={};motion={};texts={}
    for start in range(0,len(ordered),32):
        batch=ordered[start:start+32];items=[]
        for r in batch:
            a=np.array(np.load(r['feature_path'],mmap_mode='r'),dtype='float32')
            if mode=='52':
                s=(len(a)-52)//2;a=a[s:s+52]
            if perturb:a=perturb(a,r)
            a=canonicalize(a);items.append(((a-stats['mean'])/stats['std'],text[r['caption']],r))
        x,l,t,rr=collate(items)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h,m=model.motion(x.cuda(),l.cuda());tt=model.text(t.cuda())
        for i,r in enumerate(rr):
            k=r['sample_id'];features[k]=h[i].float().cpu().numpy();motion[k]=m[i].float().cpu().numpy();texts[k]=tt[i].float().cpu().numpy()
    ids=[r['sample_id'] for r in records]
    return np.stack([features[k] for k in ids]),np.stack([motion[k] for k in ids]),np.stack([texts[k] for k in ids])

def save_checkpoint(path,model,optimizer,scheduler,epoch,batch,step,best,bad,contract,losses):
    value=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        epoch=epoch,batch=batch,step=step,best=best,bad_epochs=bad,contract=contract,epoch_losses=losses,
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    temp=path.with_suffix('.tmp');torch.save(value,temp);os.replace(temp,path)

def train(domain,seed,pilot=False):
    raise RuntimeError('Legacy v1-compatible trainer disabled in v2; use native_eval.v2_train via python -m native_eval train')
    cfg=read(CODE/'config.json');variant=cfg.get('variant','R0');dest=ROOT/'runs'/variant/domain/(f'pilot_seed{seed}' if pilot else f'seed{seed}');dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():return read(dest/'DONE.json')
    text=TextCache();rr=list(rows(ROOT/'data'/f'{domain}_train.jsonl'));stats=dict(np.load(ROOT/'data'/f'{domain}_stats.npz'))
    dev=panel(domain,'development');seed_all(seed);model=DualEncoder(cfg,variant=cfg.get('variant','R0')).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    families=len({r['source_family_id'] for r in rr});epoch_steps=math.ceil(families/cfg['batch_size'])
    max_steps=cfg['pilot_updates'] if pilot else cfg['max_epochs']*epoch_steps
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1,(s+1)/max(1,epoch_steps))*max(.1,.5*(1+math.cos(math.pi*s/max_steps))))
    contract=dict(config=cfg,domain=domain,seed=seed,data_sha256=sha(ROOT/'data'/f'{domain}_train.jsonl'),
        dev_panel_sha256=dev['panel_sha256'],statistics_sha256=sha(ROOT/'data'/f'{domain}_stats.npz'),text_cache=text.identities,
        architecture_code_sha256=sha(CODE/'native_eval/model.py'),training_code_sha256=sha(CODE/'native_eval/train.py'),pilot=pilot)
    write(dest/'CONTRACT.json',contract)
    state=dict(epoch=0,batch=0,step=0,best=-1.,bad_epochs=0,epoch_losses=[])
    if (dest/'last.pt').exists():
        state=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
        if state['contract']!=contract:raise ValueError('Resume contract changed; cannot silently reuse optimizer')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
    mean=torch.tensor(stats['mean'],device='cuda');std=torch.tensor(stats['std'],device='cuda')
    step=state['step'];best=state['best'];bad=state['bad_epochs'];started=time.time();start_step=step;losses=state['epoch_losses']
    status('PILOT' if pilot else 'TRAINING',domain=domain,seed=seed,directory=str(dest))
    with gpu_budget(('pilot:' if pilot else 'train:')+domain+':'+str(seed)):
        for epoch in range(state['epoch'],cfg['max_epochs']):
            chosen=epoch_records(rr,seed,epoch)
            if pilot:chosen=chosen[:32*cfg['pilot_updates']]
            loader=DataLoader(MotionDataset(chosen,text,stats,epoch,seed,True),batch_size=cfg['batch_size'],
                num_workers=cfg['workers'],collate_fn=collate,pin_memory=True,prefetch_factor=2)
            for bi,(x,l,t,records) in enumerate(loader):
                if epoch==state['epoch'] and bi<state['batch']:continue
                model.train();x=x.cuda(non_blocking=True);l=l.cuda();t=t.cuda();optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    h,m=model.motion(x,l);tt=model.text(t)
                    ctl=contrastive(m,tt,t,records,cfg['temperature'])
                    rec=reconstruction_loss(model.reconstruct(h,l,x.shape[1]),x,l,mean,std)
                    loss=ctl+cfg['reconstruction_weight']*rec
                    # Text is never assigned to these cropped motion-only examples.
                    if step%4==0:
                        # R0 keeps the old leading-window exposure. R1/R2 use
                        # a pre-registered deterministic start/middle/end window.
                        if cfg.get('variant','R0')=='R0': starts=torch.zeros(len(x),dtype=torch.long,device=x.device)
                        else:
                            starts=torch.minimum(torch.clamp((l-52)//2,min=0),torch.tensor(52,device=x.device))
                            starts=torch.where((step//4)%3==0,torch.zeros_like(starts),starts)
                            starts=torch.where((step//4)%3==2,torch.clamp(l-52,min=0),starts)
                        short=torch.stack([x[i,s:s+52] for i,s in enumerate(starts)],0).clone();short[:,:,[0,2]]-=short[:,:1,[0,2]]
                        sl=torch.full((len(x),),52,dtype=torch.long,device=x.device);sh,_=model.motion(short,sl)
                        loss=loss+.25*reconstruction_loss(model.reconstruct(sh,sl,52),short,sl,mean,std)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
                if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                optimizer.step();scheduler.step();step+=1;losses.append([float(loss),float(ctl),float(rec)])
                if step%250==0:
                    resource_guard();save_checkpoint(dest/'last.pt',model,optimizer,scheduler,epoch,bi+1,step,best,bad,contract,losses)
                if pilot and step>=cfg['pilot_updates']:break
            torch.cuda.synchronize()
            if pilot:
                report=dict(updates=step,seconds=time.time()-started,seconds_per_update=(time.time()-started)/max(1,step-start_step),
                    losses_start=np.mean(losses[:10],0).tolist(),losses_end=np.mean(losses[-10:],0).tolist(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),families= families,epoch_steps=epoch_steps)
                save_checkpoint(dest/'last.pt',model,optimizer,scheduler,epoch,bi+1,step,best,bad,contract,losses)
                write(dest/'DONE.json',report);print('PILOT_DONE',report,flush=True);break
            _,m,tt=encode(model,dev['records'],text,stats)
            scores,_=retrieval(tt,m,dev['candidate_pools'][str(cfg['panel_seeds'][0])]);criterion=scores['R@1']
            improved=criterion>best+1e-6
            if improved:best=criterion;bad=0
            else:bad+=1
            curve=dict(epoch=epoch+1,step=step,loss=np.mean(losses,0).tolist(),development=scores,lr=optimizer.param_groups[0]['lr'],elapsed_seconds=time.time()-started,best=best,bad_epochs=bad)
            with open(dest/'curve.jsonl','a') as f:f.write(json.dumps(curve)+'\n')
            save_checkpoint(dest/'last.pt',model,optimizer,scheduler,epoch+1,0,step,best,bad,contract,[])
            if improved:shutil.copyfile(dest/'last.pt',dest/'best.pt')
            write(dest/'PROGRESS.json',curve);losses=[];resource_guard()
            if epoch+1>=cfg['min_epochs'] and bad>=cfg['patience']:break
        if not pilot:
            report=dict(epochs=epoch+1,updates=step,best_development_R1=best,seconds=time.time()-started,checkpoint_sha256=sha(dest/'best.pt'),selection='primary seed predeclared; maximum development R@1; ties keep earliest; min10 epochs patience8',stop='patience' if bad>=cfg['patience'] else 'max_epochs')
            write(dest/'DONE.json',report);print('TRAIN_DONE',domain,seed,report,flush=True)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    ledger('pilot_done' if pilot else 'training_done',domain=domain,seed=seed,report=report)
    return report

def freeze_plan():
    cfg=read(CODE/'config.json');pilot=read(ROOT/'runs/bones'/f'pilot_seed{cfg["seed_release"]}'/'DONE.json')
    quotas={};total=0
    for d in ['bones','humanml3d']:
        rr=list(rows(ROOT/'data'/f'{d}_train.jsonl'));n=len({r['source_family_id'] for r in rr});steps=math.ceil(n/cfg['batch_size'])
        hours=steps*cfg['max_epochs']*pilot['seconds_per_update']/3600*1.5
        quotas[d]=dict(families_per_epoch=n,batch=cfg['batch_size'],updates_max=steps*cfg['max_epochs'],epochs_max=cfg['max_epochs'],eta_hours_per_seed_upper=hours,seeds=[cfg['seed_release'],cfg['seed_sensitivity']]);total+=hours*2
    if total>32:raise RuntimeError('Pilot forecast >32h train quota; freeze reduced plan explicitly before training')
    plan=dict(pilot=pilot,quotas=quotas,optimizer='AdamW',config=cfg,total_train_eta_hours_upper=total,reserve_validation_and_one_correction_hours=12,primary_release_seed=cfg['seed_release'],selection='development R@1, stable earliest tie; confirmation opened only after policy freeze',order=['bones primary','humanml3d primary','bones second','humanml3d second'])
    write(ROOT/'TRAIN_PLAN.json',plan);ledger('plan_frozen',plan=plan);return plan
