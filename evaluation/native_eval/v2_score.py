"""Frozen native-v2 per-metric scoring; v1 metadata shape remains compatible."""
from pathlib import Path
import numpy as np
import torch
from .common import read,rows,sha,identity,write,output_path
from .model import DualEncoder
from .schema import load_sample,diagnostics
from .encoding import encode_arrays
from .metrics import fid,retrieval,family_mean_ci

METRICS={'retrieval':'basic_text_motion_retrieval','fid':'short_window_distribution','event_order':'event_order_discrimination','full_sequence':'full_sequence_distribution'}
def gate(caps,metric,diagnostic):
    requested=['fid','retrieval'] if metric=='both' else [metric]
    for m in requested:
        if caps[METRICS[m]]['status']!='PASS' and not diagnostic:raise ValueError(f'{m}: {caps[METRICS[m]]["status"]}; --diagnostic required')
    if metric in ('event_order','full_sequence'):raise ValueError('No verified reference/protocol for this metric, including diagnostic scoring; see capability report')
    return requested

def verify_release(folder):
    folder=Path(folder);r=read(folder/'RELEASE.json')
    if identity({k:v for k,v in r.items() if k!='release_sha256'})!=r['release_sha256']:raise ValueError('Release identity changed')
    for name,h in r['files'].items():
        if sha(folder/name)!=h:raise ValueError('Modified frozen release asset '+name)
    for name in ('model.py','schema.py','encoding.py','metrics.py','v2_score.py'):
        if sha(Path(__file__).parent/name)!=sha(folder/'code/native_eval'/name):raise ValueError('Use matching release code')
    return r

def model_for(folder,head,device):
    st=torch.load(folder/f'{head}.pt',map_location='cpu',weights_only=False);m=DualEncoder(st['contract']['config']).to(device).eval();m.load_state_dict(st['model']);return m

def check_physical(a):
    diag=diagnostics(a)
    if diag['rotation_unit_norm_error']>.1 or diag['rotation_orthogonality_error']>.1:raise ValueError('Invalid rotation6D channels')
    if diag['position_velocity_forward_difference_rmse']>.5:raise ValueError('Velocity channels inconsistent with position/fps')
    if diag['contact_out_of_range_fraction']>0:raise ValueError('Contact channels outside [0,1]')
    return diag

@torch.inference_mode()
def score(folder,manifest,output,metric='both',diagnostic=False,device='cpu'):
    output=output_path(output);folder=Path(folder);release=verify_release(folder);requested=gate(release['capabilities'],metric,diagnostic)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    panel_name={'fid':'fid52_panel.json','retrieval':'retrieval_panel.json','both':'combined_panel.json'}[metric];p=read(folder/panel_name);rr=p['records'];ids=[r['sample_id'] for r in rr]
    meta=list(rows(manifest));by={m['sample_id']:m for m in meta}
    if len(by)!=len(meta) or set(by)!=set(ids):raise ValueError('Missing/extra/duplicate frozen-panel sample; no successful-subset filtering')
    for key in ('checkpoint_sha256','generator_code_snapshot','dataset','track','seed','postprocess'):
        if len({str(m[key]) for m in meta})!=1:raise ValueError('Mixed group '+key)
    if any(m['track']!=release['track'] or m['dataset']!=release['domain'] for m in meta):raise ValueError('Wrong domain/release')
    if len({m['conditioning']['mode'] for m in meta})!=1:raise ValueError('Mixed conditioning')
    stats=dict(np.load(folder/'stats.npz'));sk=read(folder/'SKELETON.json');ref=np.load(folder/'reference_embeddings.npz');qp=read(folder/'retrieval_panel.json');qby={r['sample_id']:r for r in qp['records']};arrays=[];short=[];physical=[]
    for r in rr:
        m=by[r['sample_id']]
        if m['panel_id']!=p['panel_sha256'] or m['source_family_id']!=r['source_family_id']:raise ValueError('Panel/family identity mismatch')
        a,(start,end)=load_sample(m,Path(manifest).parent,sk['sha256']);a=a[start:end]
        if len(a)!=r['frames']:raise ValueError('Frozen duration mismatch')
        if 'retrieval' in requested and r['sample_id'] in qby:
            cs=m['caption_intervals'];caption=qby[r['sample_id']]['caption']
            if len(cs)!=1 or cs[0]['start']!=start or cs[0]['end']!=end or cs[0]['text']!=caption or cs[0]['alignment']!='full_semantic_segment':raise ValueError('Retrieval requires exact full-semantic caption and interval')
        physical.append(dict(sample_id=r['sample_id'],diagnostics=check_physical(a)));arrays.append(a);s=(len(a)-52)//2;short.append(a[s:s+52])
    cache_key=identity(dict(release=release['release_sha256'],manifest=sha(manifest),arrays=[by[i]['array_sha256'] for i in ids],metric=metric,panel=p['panel_sha256'],precision='FP32'))
    cache=output_path(output.parent/'encoding_cache'/cache_key);cache.mkdir(parents=True,exist_ok=True);cp=cache/'embeddings.npz';complete=cache/'COMPLETE.json'
    if complete.exists():
        c=read(complete)
        if c['key']!=cache_key or c['sha256']!=sha(cp):raise ValueError('Embedding cache identity changed')
        encoded=dict(np.load(cp))
    else:
        encoded={}
        if 'fid' in requested:
            m=model_for(folder,'distribution',device);encoded['h52']=encode_arrays(m,rr,short,stats)[0];del m
        if 'retrieval' in requested:
            m=model_for(folder,'retrieval',device);encoded['motion']=encode_arrays(m,rr,arrays,stats)[1];del m
        np.savez(cp,**encoded);write(complete,dict(key=cache_key,sha256=sha(cp)))
    result=dict(version='native-v2',release_sha256=release['release_sha256'],track=release['track'],diagnostic_only=diagnostic,status='DIAGNOSTIC_UNVALIDATED' if diagnostic else 'SCORED',N=len(rr),completion_rate=1.,cache_key=cache_key,capabilities=release['capabilities'],generator_test_exposure='not established',physical_diagnostics=physical,postprocess=meta[0]['postprocess'],checkpoint_sha256=meta[0]['checkpoint_sha256'],full_sequence='NOT_TESTED')
    if 'fid' in requested:
        if list(ref['distribution_ids'])!=ids:raise ValueError('FID reference ordering mismatch')
        result['FID_native52_center_v2']=fid(ref['h52'],encoded['h52'])
    if 'retrieval' in requested:
        qids=[r['sample_id'] for r in qp['records']]
        if list(ref['retrieval_ids'])!=qids:raise ValueError('Retrieval reference ordering mismatch')
        mm=encoded['motion'][[ids.index(k) for k in qids]];result['R_T2M_pool32']={}
        for seed,pools in qp['candidate_pools'].items():
            scores,rank=retrieval(ref['text'],mm,pools);result['R_T2M_pool32'][seed]=scores
            if seed=='4101':result['R1_family_ci']=family_mean_ci((rank<=1).astype(float),[r['source_family_id'] for r in qp['records']],repeats=1000)
    write(output,result);return result
