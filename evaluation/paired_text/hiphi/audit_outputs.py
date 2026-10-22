from pathlib import Path
import os,json,csv,time,math
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from execute import R,rd,wr,sha,schedule,verify
from prepare_scoring import representation
from native_eval.metrics import fid
import numpy as np,torch

def norm(x):x=np.asarray(x,np.float64);return x/np.linalg.norm(x,axis=-1,keepdims=True)
def main():
 verify();torch.set_num_threads(2);rep,_=representation();checks=[];cases=[json.loads(s) for s in (R/'CASES.jsonl').open() if json.loads(s)['split']=='locked_test'];bank=dict(np.load(R/'REAL_PAIRED_FEATURES.npz'));ix={int(v):i for i,v in enumerate(bank['catalog_indices'])};pools=rd(R/'QUERY_POOLS.json');models=list(rd(R/'CHECKPOINTS.json'))
 assert len(models)==14
 for method in models:
  for task in ['short52','nominal9','nominal20','nominal40']:
   d=R/'models'/method/task;s=R/'scoring/models'/method/task;score=rd(s/'SCORES.json');receipt=rd(s/'COMPLETE.json')
   for n,h in receipt['files'].items():assert sha(s/n)==h,(method,task,n)
   gate=rd(d/'ACCELERATION_GATE.json');assert gate['passed'] and len(gate['rows'])==4 and all(all(x['checks'].values()) for x in gate['rows'])
   W=len(schedule(task)[0]);roots=[];ends=[]
   for ci,c in enumerate(cases):
    states,texts=schedule(task,c)
    for si,seed in enumerate(c['noise_seeds']):
     f=d/'trials'/f'{ci:03d}_s{si}';a=rd(f/'INPUT_AND_EVENTS.json');co=rd(f/'COMPLETE.json');assert sha(f/'geometry.npz')==co['geometry_sha256'] and sha(f/'INPUT_AND_EVENTS.json')==co['audit_sha256'] and co['seed']==seed
     assert a['no_hidden_GT'] and a['no_paired_GT_motion_supplied_to_generator'] and a['detailed_text_pairing']
     for wi,v in enumerate(a['events']):assert v['prompt']==texts[wi] and v['stage']==states[wi] and v['target_motion_is_None']
     with np.load(f/'geometry.npz') as z:
      x=z['physical'];j=z['joints'];root=z['root'];assert x.shape==(W*52,414) and np.array_equal(root,x[:,:3])
      with torch.inference_mode():fk=rep.inverse(torch.tensor(x[-1:])[None],is_normalized=False)['posed_joints'][0,0].numpy().astype(np.float64)
      assert np.max(np.abs(fk-j[-1]))<2e-5
      roots.append(float(np.sqrt(np.square(root[-1,[0,2]].astype(np.float64)-np.asarray(c['root_goal_xz_m'])).sum())*100));ends.append(float(np.sqrt(np.square(fk[c['selected_joint_index']]-np.asarray(c['end_goal_xyz_m'])).sum())*100))
   assert abs(np.mean(roots)-score['root_CE_cm']['mean'])<1e-9 and abs(np.mean(ends)-score['end_CE_cm']['mean'])<.002
   feat=dict(np.load(s/'FEATURES.npz'));fids=[];R3=[]
   for wi,phase in enumerate(rd(R/'PLAN.json')['paired_phases'][task]):
    ids=[ix[c['paired_reference_indices'][phase]] for c in cases];fids.append(fid(norm(feat['distribution'][wi].reshape(384,-1)),norm(bank['distribution'][ids])))
    meta=pools[str(phase)];q=meta['query_case_indices'];realids=[ids[j] for j in q];values=[]
    for head in [1701,2903]:
     t=norm(bank['text'+str(head)][realids])
     for ps,pp in meta['pools'].items():
      for noise in range(3):
       m=norm(feat['retrieval'+str(head)][wi,q,noise]);dist=np.maximum(0,2-2*t@m.T);ranks=[]
       for qi,choices in enumerate(pp):
        assert len(choices)==len(set(choices))==32
        order=np.argsort(dist[qi,choices],kind='stable');ranks.append(1+int(np.flatnonzero(np.asarray(choices)[order]==qi)[0]))
       values.append(float((np.asarray(ranks)<=3).mean()*100))
    R3.append(np.mean(values))
   assert abs(np.mean(fids)-score['mean_window_L2_FID'])<1e-6 and abs(np.mean(R3)-score['R3_percent'])<1e-9
   latency=rd(d/'LATENCY.json');assert latency['batch']==1 and latency['warmup_sequences']==2 and len(latency['rows'])==32
   times=np.array([v['elapsed_s'] for v in latency['rows']])*1000/W;assert abs(np.median(times)-latency['median_ms_window'])<1e-9
   checks.append(dict(model=method,task=task,root_CE=float(np.mean(roots)),end_CE=float(np.mean(ends)),R3=float(np.mean(R3)),L2_FID=float(np.mean(fids)),latency=float(np.median(times)),geometry_FK_input_caption_and_noise_verified=True))
   wr(R/'AUDIT_PROGRESS.json',dict(completed=len(checks),total=56,time=time.time()))
 wr(R/'INDEPENDENT_FINAL_AUDIT.json',dict(passed=True,conditions=56,rollouts=21504,rows=checks,strict32_detailed_paired_caption=True,no_old_model_scores_reused=True,source_sha256=sha(__file__),time=time.time()))
if __name__=='__main__':main()
