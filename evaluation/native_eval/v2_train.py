"""Controlled ablations. Full semantic supervision, raw motion-only crops."""
import collections
import gc
import math
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .common import *
from .model import DualEncoder,contrastive,reconstruction_loss
from .schema import canonicalize
from .text import TextCache
from .encoding import encode
from .metrics import retrieval,embedding_health

def raw_crop(a,variant,epoch,sample_id):
    position=0 if variant=='R0' else (epoch+int(identity(sample_id)[:8],16))%3
    starts=(0,(len(a)-52)//2,len(a)-52)
    return canonicalize(a[starts[position]:starts[position]+52]),int(starts[position])

class Motions(Dataset):
    def __init__(self,records,text,stats,variant,epoch,seed):
        self.rr=records;self.text=text;self.stats=stats;self.variant=variant;self.epoch=epoch;self.seed=seed
    def __len__(self):return len(self.rr)
    def __getitem__(self,i):
        r=dict(self.rr[i]);a=np.load(r['feature_path'],allow_pickle=False)
        rng=np.random.default_rng(int(identity([self.seed,self.epoch,r['sample_id']])[:16],16))
        c=r.get('caption') or r['captions'][int(rng.integers(len(r['captions'])))];r['caption']=c
        short,start=raw_crop(a,self.variant,self.epoch,r['sample_id']);r['short_start']=start
        mean,std=self.stats['mean'],self.stats['std']
        return (canonicalize(a)-mean)/std,(short-mean)/std,self.text[c],r

def collate(items):
    lengths=torch.tensor([len(x[0]) for x in items]);x=torch.zeros(len(items),int(lengths.max()),414)
    for i,v in enumerate(items):x[i,:len(v[0])]=torch.from_numpy(v[0])
    return x,lengths,torch.from_numpy(np.stack([v[1] for v in items])),torch.from_numpy(np.stack([v[2] for v in items])),[v[3] for v in items]

def epoch_records(rr,seed,epoch):
    groups=collections.defaultdict(list);rng=np.random.default_rng(seed+epoch*100003)
    for r in rr:groups[r['source_family_id']].append(r)
    chosen=[rs[int(rng.integers(len(rs)))] for _,rs in sorted(groups.items())];rng.shuffle(chosen)
    chunks=[]
    for i in range(0,len(chosen),1024):
        bucket=sorted(chosen[i:i+1024],key=lambda r:r['frames'])
        parts=[bucket[j:j+32] for j in range(0,len(bucket),32)];rng.shuffle(parts)
        chunks.extend(parts)
    return [r for part in chunks for r in part]

def train_contract(cfg,variant,domain,seed,text,pilot):
    return dict(config=dict(cfg,variant=variant),domain=domain,seed=seed,variant=variant,pilot=pilot,
        data_sha256=sha(ROOT/'data'/f'{domain}_train.jsonl'),development_panel_sha256=sha(ROOT/'panels'/f'{domain}_development.json'),statistics_sha256=sha(ROOT/'data'/f'{domain}_stats.npz'),text_cache=text.identities,
        code={name:sha(CODE/'native_eval'/name) for name in ('v2_train.py','model.py','encoding.py','schema.py','metrics.py','common.py','text.py')})

def checkpoint(path,model,opt,sched,contract,epoch,batch,step,best_r,best_d,bad,losses):
    output_path(path)
    st=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),contract=contract,epoch=epoch,batch=batch,step=step,best_r=best_r,best_d=best_d,bad=bad,losses=losses,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
    tmp=path.with_suffix('.tmp');torch.save(st,tmp);os.replace(tmp,path)

@torch.inference_mode()
def development(model,panel,text,stats,domain):
    full,m,t=encode(model,panel['records'],text,stats);scores,_=retrieval(t,m,panel['candidate_pools']['4101'])
    # Held-out development reconstruction of raw center52 windows is a
    # distribution/dynamic selection proxy, NOT an FID quality certificate.
    rr=read(ROOT/'panels'/f'{domain}_development_distribution.json')['records']
    mean=torch.tensor(stats['mean'],device='cuda');std=torch.tensor(stats['std'],device='cuda');vals=[];features=[]
    for i in range(0,len(rr),32):
        batch=rr[i:i+32];a=[]
        for r in batch:
            raw=np.load(r['feature_path']);s=(len(raw)-52)//2;a.append((canonicalize(raw[s:s+52])-stats['mean'])/stats['std'])
        x=torch.tensor(np.stack(a),device='cuda');l=torch.full((len(x),),52,device='cuda',dtype=torch.long)
        h,_=model.motion(x,l);pred=model.reconstruct(h,l,52)
        rec=reconstruction_loss(pred,x,l,mean,std)
        dynamic=((pred[:,1:,:410]-pred[:,:-1,:410])-(x[:,1:,:410]-x[:,:-1,:410])).square().mean()
        vals.append([len(x),float(rec),float(dynamic)]);features.append(h.cpu().numpy())
    vv=np.array(vals);return dict(retrieval=scores,reconstruction=float(np.average(vv[:,1],weights=vv[:,0])),dynamic_reconstruction=float(np.average(vv[:,2],weights=vv[:,0])),health=embedding_health(np.concatenate(features)))

def train(domain,variant,seed,pilot=False):
    cfg=read(CODE/'config.json');cfg.pop('variant',None)
    dest=ROOT/'runs'/variant/domain/(f'pilot_seed{seed}' if pilot else f'seed{seed}');dest.mkdir(parents=True,exist_ok=True)
    text=TextCache();contract=train_contract(cfg,variant,domain,seed,text,pilot)
    if (dest/'DONE.json').exists():
        if read(dest/'CONTRACT.json')!=contract:raise ValueError('Completed training contract changed')
        return read(dest/'DONE.json')
    rr=list(rows(ROOT/'data'/f'{domain}_train.jsonl'));stats=dict(np.load(ROOT/'data'/f'{domain}_stats.npz'));panel=read(ROOT/'panels'/f'{domain}_development.json')
    for r in rr:
        if sha(r['feature_path'])!=r['actual_feature_sha256']:raise ValueError('Training motion changed after admission: '+r['sample_id'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=DualEncoder(dict(cfg,variant=variant)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    steps=math.ceil(len({r['source_family_id'] for r in rr})/32);maximum=cfg['pilot_updates'] if pilot else cfg['max_epochs']*steps
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1,(s+1)/steps)*max(.1,.5*(1+math.cos(math.pi*s/maximum))))
    st=dict(epoch=0,batch=0,step=0,best_r=-1.,best_d=1e30,bad=0,losses=[])
    if (dest/'last.pt').exists():
        st=torch.load(dest/'last.pt',map_location='cpu',weights_only=False)
        if st['contract']!=contract:raise ValueError('Resume refused: code/data/config/text/statistics contract changed')
        model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);sched.load_state_dict(st['scheduler'])
        random.setstate(st['rng']['python']);np.random.set_state(st['rng']['numpy']);torch.set_rng_state(st['rng']['torch']);torch.cuda.set_rng_state_all(st['rng']['cuda'])
    write(dest/'CONTRACT.json',contract);mean=torch.tensor(stats['mean'],device='cuda');std=torch.tensor(stats['std'],device='cuda')
    step=st['step'];start_step=step;best_r=st['best_r'];best_d=st['best_d'];bad=st['bad'];losses=st['losses'];start=time.time()
    status('TRAINING_V2',domain=domain,variant=variant,seed=seed,run=str(dest))
    with gpu_budget(f'train:{variant}:{domain}:{seed}:pilot={pilot}'):
        for epoch in range(st['epoch'],cfg['max_epochs']):
            chosen=epoch_records(rr,seed,epoch)
            if pilot:chosen=chosen[:32*cfg['pilot_updates']]
            loader=DataLoader(Motions(chosen,text,stats,variant,epoch,seed),batch_size=32,num_workers=2,collate_fn=collate,pin_memory=True,prefetch_factor=2,generator=torch.Generator().manual_seed(seed+epoch*100003))
            for bi,(x,l,short,t,records) in enumerate(loader):
                if epoch==st['epoch'] and bi<st['batch']:continue
                x=x.cuda();l=l.cuda();short=short.cuda();t=t.cuda();model.train();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    h,m=model.motion(x,l);tt=model.text(t);ctl=contrastive(m,tt,t,records,cfg['temperature']);rec=reconstruction_loss(model.reconstruct(h,l,x.shape[1]),x,l,mean,std);loss=ctl+rec
                    if step%4==0:
                        sl=torch.full_like(l,52);sh,_=model.motion(short,sl);loss=loss+.25*reconstruction_loss(model.reconstruct(sh,sl,52),short,sl,mean,std)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(gn):raise ValueError('Nonfinite gradients')
                opt.step();sched.step();step+=1;losses.append([float(loss),float(ctl),float(rec)])
                if step%250==0:
                    checkpoint(dest/'last.pt',model,opt,sched,contract,epoch,bi+1,step,best_r,best_d,bad,losses);resource_guard()
                if pilot and step>=maximum:break
            if pilot:
                torch.cuda.synchronize();result=dict(updates=step,seconds=time.time()-start,seconds_per_update=(time.time()-start)/(step-start_step),peak_gpu_bytes=torch.cuda.max_memory_allocated(),parameters=sum(p.numel() for p in model.parameters()),families=len(chosen),pilot_only=True)
                write(dest/'DONE.json',result);break
            dev=development(model,panel,text,stats,domain);r=dev['retrieval']['R@1'];improved=r>best_r+1e-6
            best_r=max(best_r,r);bad=0 if improved else bad+1
            # Retrieval admissibility, then low bottleneck recon; dynamic error
            # is a disclosed tie-break. No corruption-specific tuning.
            score=dev['reconstruction'];distribution_improved=r>=max(.1,best_r-.02) and dev['health']['effective_rank']>=8 and score<best_d
            if distribution_improved:best_d=score
            curve=dict(epoch=epoch+1,step=step,loss=np.mean(losses,axis=0).tolist(),development=dev,best_r=best_r,best_d=best_d,bad_epochs=bad,elapsed_seconds=time.time()-start)
            with open(dest/'curve.jsonl','a') as f:f.write(json.dumps(curve)+'\n')
            checkpoint(dest/'last.pt',model,opt,sched,contract,epoch+1,0,step,best_r,best_d,bad,[])
            if improved:shutil.copyfile(dest/'last.pt',dest/'retrieval.pt')
            if distribution_improved:shutil.copyfile(dest/'last.pt',dest/'distribution.pt')
            write(dest/'PROGRESS.json',curve);losses=[];resource_guard()
            if epoch+1>=cfg['min_epochs'] and bad>=cfg['patience']:break
        if not pilot:
            result=dict(epochs=epoch+1,updates=step,seconds=time.time()-start,parameters=sum(p.numel() for p in model.parameters()),compute=dict(optimizer_updates=step,approx_family_presentations=step*32,motion_only_updates=(step+3)//4,peak_gpu_bytes=torch.cuda.max_memory_allocated(),input_projection_MACs_per_52_frames=52*414*cfg['hidden'],note='R0/R1 project each frame then mean; R2 projects ordered4 flatten; same leading input multiply count per complete4 block, R2 has more parameters; transformer token lengths equal'),checkpoints={k:sha(dest/f'{k}.pt') for k in ('retrieval','distribution','last') if (dest/f'{k}.pt').exists()},best_development_R1=best_r,best_distribution_reconstruction=best_d,stop='patience' if bad>=cfg['patience'] else 'max_epochs',selection='separate explicit retrieval and bottleneck-reconstruction checkpoints; same epochs/optimizer/early-stop budget for all variants')
            write(dest/'DONE.json',result)
    ledger('training_done',variant=variant,domain=domain,seed=seed,result=result)
    del model,opt;gc.collect();torch.cuda.empty_cache();return result
