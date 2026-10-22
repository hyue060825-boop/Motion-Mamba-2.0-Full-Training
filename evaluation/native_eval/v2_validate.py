"""Development-only selection followed by one identity-bound confirmation."""
import gc
import numpy as np
import torch
from .common import *
from .model import DualEncoder
from .text import TextCache
from .encoding import encode
from .metrics import retrieval,ranks,distances,normalize,family_mean_ci
from .v2_train import development
from .v2_statistics import distribution,health
from .v2_perturbations import edit,EDITS
from .v2_prepare import DOMAINS

def load_model(domain,variant,seed,head):
    run=ROOT/'runs'/variant/domain/f'seed{seed}';ck=run/f'{head}.pt';done=read(run/'DONE.json')
    if sha(ck)!=done['checkpoints'][head]:raise ValueError('Checkpoint content mismatch')
    st=torch.load(ck,map_location='cpu',weights_only=False);c=st['contract']
    for name,h in c['code'].items():
        if sha(CODE/'native_eval'/name)!=h:raise ValueError('Training code changed '+name)
    if c['statistics_sha256']!=sha(ROOT/'data'/f'{domain}_stats.npz'):raise ValueError('Statistics changed')
    m=DualEncoder(c['config']).cuda().eval();m.load_state_dict(st['model']);return m,ck

def select():
    out=ROOT/'SELECTION_V2.json'
    if out.exists():return read(out)
    if (ROOT/'CONFIRMATION_OPENED.json').exists():raise ValueError('Holdout already exposed: cannot reselect')
    cfg=read(CODE/'config.json');text=TextCache();all_results={};selected={}
    for d in DOMAINS:
        stats=dict(np.load(ROOT/'data'/f'{d}_stats.npz'));p=read(ROOT/'panels'/f'{d}_development.json');candidates=[];all_results[d]={}
        for variant in ('R0','R1','R2'):
            all_results[d][variant]={}
            for head in ('retrieval','distribution'):
                ck=ROOT/'runs'/variant/d/'seed1701'/f'{head}.pt'
                if not ck.exists():continue
                m,ck=load_model(d,variant,1701,head);dev=development(m,p,text,stats,d)
                item=dict(variant=variant,head=head,checkpoint_sha256=sha(ck),development=dev);candidates.append(item);all_results[d][variant][head]=item
                del m;gc.collect();torch.cuda.empty_cache()
        retrieval_best=sorted(candidates,key=lambda c:(-c['development']['retrieval']['R@1'],c['variant'],c['head']))[0]
        eligible=[c for c in candidates if c['development']['retrieval']['R@1']>=max(.1,retrieval_best['development']['retrieval']['R@1']-.02) and c['development']['health']['effective_rank']>=8]
        distribution_best=sorted(eligible,key=lambda c:(c['development']['reconstruction'],c['development']['dynamic_reconstruction'],c['variant'],c['head']))[0] if eligible else retrieval_best
        selected[d]=dict(retrieval=retrieval_best,distribution=distribution_best)
    result=dict(frozen_at=__import__('time').time(),policy_sha256=sha(ROOT/'VALIDATION_POLICY_V2.json'),selected=selected,development=all_results,holdout_not_used=True)
    write(out,result)
    runs={str(p.relative_to(ROOT/'runs')):read(p) for p in (ROOT/'runs').rglob('DONE.json') if 'pilot' not in str(p)}
    write(ROOT/'ABLATION_RESULTS.json',dict(runs=runs,development=all_results,selection=selected,rule='retrieval maximum; distribution minimum raw-center52 bottleneck reconstruction among retrieval within 0.02 of best; dynamics tie break; no corruption or generator rank selection'))
    ledger('selection_frozen',selection_sha256=sha(out));return result

def mirror_results(model,d,split,text,stats):
    pairs=read(ROOT/'panels'/f'{d}_{split}_direction.json');out=[]
    for pair in pairs:
        rr=[pair['original'],pair['mirror']];_,m,t=encode(model,rr,text,stats);s=normalize(t)@normalize(m).T
        out.append(dict(sample_id=rr[0]['sample_id'],family=rr[0]['source_family_id'],actor=rr[0]['actor_id_namespaced'],margin=float((s[0,0]-s[0,1]+s[1,1]-s[1,0])/2),accuracy=(int(s[0,0]>s[0,1])+int(s[1,1]>s[1,0]))/2))
    return dict(N=len(out),samples=out,margin=family_mean_ci([r['margin'] for r in out],[r['family'] for r in out],repeats=1000) if out else None,accuracy=float(np.mean([r['accuracy'] for r in out])) if out else None)

def encode_confirmation(d,seed,selection):
    dest=ROOT/'validation'/d/f'seed{seed}'/'holdout';dest.mkdir(parents=True,exist_ok=True);result_path=dest/'ENCODING_RESULT.json'
    spec={metric:dict(variant=c['variant'],head=c['head'],checkpoint_sha256=sha(ROOT/'runs'/c['variant']/d/f'seed{seed}'/(c['head']+'.pt'))) for metric,c in selection.items()}
    contract=dict(spec=spec,policy_sha256=sha(ROOT/'VALIDATION_POLICY_V2.json'),selection_sha256=sha(ROOT/'SELECTION_V2.json'),stats_sha256=sha(ROOT/'data'/f'{d}_stats.npz'),retrieval_panel_sha256=sha(ROOT/'panels'/f'{d}_holdout.json'),distribution_panel_sha256=sha(ROOT/'panels'/f'{d}_holdout_distribution.json'),validation_code_sha256=sha(CODE/'native_eval/v2_validate.py'),perturbations_sha256=sha(CODE/'native_eval/v2_perturbations.py'))
    if result_path.exists():
        saved=read(result_path)
        if saved['contract']!=contract or sha(dest/'embeddings.npz')!=saved['embeddings_sha256']:raise ValueError('Confirmation contract/cache changed')
        return saved
    write(dest/'CONTRACT.json',contract);text=TextCache();stats=dict(np.load(ROOT/'data'/f'{d}_stats.npz'));p=read(ROOT/'panels'/f'{d}_holdout.json');rr=p['records'];families=[r['source_family_id'] for r in rr]
    m,ck=load_model(d,spec['retrieval']['variant'],seed,spec['retrieval']['head'])
    with gpu_budget(f'confirmation_retrieval:{d}:{seed}'):
        full,motion,t=encode(m,rr,text,stats);scores={};rk={}
        for s,pool in p['candidate_pools'].items():scores[s],rk[s]=retrieval(t,motion,pool)
        perm=np.random.default_rng(6701).permutation(len(t));shuffled,srank=retrieval(t[perm],motion,p['candidate_pools']['4101'])
        # Report per-query and fixed candidate IDs; target duration is only an oracle diagnostic.
        length=np.array([r['frames'] for r in rr]);lr=ranks((length[:,None]-length[None,:])**2,p['candidate_pools']['4101'],list(range(len(rr))))
        matched=p['length_matched'];eligible=matched['eligible'];mr=ranks(distances(t[eligible],motion),matched['pools'],eligible) if eligible else np.array([])
        mirrors=mirror_results(m,d,'holdout',text,stats)
    result=dict(contract=contract,N=len(rr),retrieval=scores,shuffled=shuffled,R1_ci=family_mean_ci((rk['4101']<=1).astype(float),families,repeats=1000),paired_minus_shuffled=family_mean_ci((rk['4101']<=1).astype(float)-(srank<=1),families,repeats=1000),length_matched=dict(N=len(mr),coverage=len(mr)/len(rr),R1=float(np.mean(mr<=1)) if len(mr) else None),length_oracle_R1=float(np.mean(lr<=1)),mirrors=mirrors,query_ids=[r['sample_id'] for r in rr],ranks=rk['4101'].tolist(),full_health=health(full))
    del m;gc.collect();torch.cuda.empty_cache()
    dp=read(ROOT/'panels'/f'{d}_holdout_distribution.json');dr=dp['records'];m,ck=load_model(d,spec['distribution']['variant'],seed,spec['distribution']['head'])
    with gpu_budget(f'confirmation_distribution:{d}:{seed}'):
        h,_,_=encode(m,dr,text,stats,mode='52');variants={}
        for kind in EDITS:
            z,_,_=encode(m,dr,text,stats,mode='52',perturb=lambda a,r,kind=kind:edit(a,kind,r['sample_id']));variants[kind]=z
        # Explicit zero-effect and small untrained perturbation control.
        variants['identity']=h.copy()
    np.savez(dest/'embeddings.npz',motion=motion,text=t,full=full,retrieval_ids=np.array([r['sample_id'] for r in rr]),h52=h,ids=np.array([r['sample_id'] for r in dr]),**variants)
    result['embeddings_sha256']=sha(dest/'embeddings.npz');write(result_path,result)
    del m;gc.collect();torch.cuda.empty_cache();return result

def analyze_confirmation(d,seed):
    dest=ROOT/'validation'/d/f'seed{seed}'/'holdout';target=dest/'RESULT.json'
    if target.exists():return read(target)
    result=read(dest/'ENCODING_RESULT.json');a=np.load(dest/'embeddings.npz');rr=read(ROOT/'panels'/f'{d}_holdout_distribution.json')['records'];h=a['h52'];variants={k:a[k] for k in list(EDITS)+['identity']}
    result['distribution']=distribution(h,variants,rr,1000)
    if d=='bones':result['actor_cluster_distribution']=distribution(h,{k:variants[k] for k in ('static','repeat4','shuffle4','shuffle52','identity')},rr,1000,True)
    result['order']=dict(status='INSUFFICIENT_EVIDENCE',candidates=read(ROOT/'panels/SEMANTIC_PANELS.json')[d+':holdout'],reason='No independently verified within-motion A/B counterfactual event timeline; reversal is auxiliary only')
    result['full_sequence_distribution']=dict(status='NOT_TESTED',reason='short-window reference does not validate continuous 9/20/40 second FID')
    write(target,result);return result

def confirm():
    sel=read(ROOT/'SELECTION_V2.json')
    if sel['policy_sha256']!=sha(ROOT/'VALIDATION_POLICY_V2.json'):raise ValueError('Frozen policy changed')
    opened=ROOT/'CONFIRMATION_OPENED.json';binding=dict(selection_sha256=sha(ROOT/'SELECTION_V2.json'),policy_sha256=sha(ROOT/'VALIDATION_POLICY_V2.json'))
    if opened.exists():
        if read(opened)!=binding:raise ValueError('Holdout already exposed with a different selection')
    else:write(opened,binding);ledger('holdout_opened_once',**binding)
    # All selected identities fixed before either seed sees confirmation.
    for d in DOMAINS:
        for s in (1701,2903):encode_confirmation(d,s,sel['selected'][d])
    for d in DOMAINS:
        for s in (1701,2903):analyze_confirmation(d,s)
    result={d:{str(s):read(ROOT/'validation'/d/f'seed{s}'/'holdout/RESULT.json') for s in (1701,2903)} for d in DOMAINS}
    write(ROOT/'VALIDATION_RESULTS_V2.json',result);return result
