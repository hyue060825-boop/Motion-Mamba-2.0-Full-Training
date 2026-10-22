"""Paired timed-goal evaluation from fixed held-out-family student boundaries."""
from motion_mamba2.paths import artifact_path
from common import *
import argparse,math,gc,statistics
import numpy as np
import torch
from motion_mamba2.models import load_model
from motion_training import runtime,sampling,history
from motion_training.evaluation import initialize,set_text,L
from motion_training.io import rows
from motion_training._compat.window_score import score_window
import future_goal as future

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--step',type=int,required=True)
a=p.parse_args();cfg=read(ROOT/'config.json')
from budget_process import start_budget
start_budget('timed_goal_evaluation:'+str(a.step),devices=1,closing=True)
split='FINAL' if a.step==5000 else 'DEV'
panelroot=ROOT/'evaluation/future_panels'/split;panelroot.mkdir(parents=True,exist_ok=True)
dest=ROOT/'evaluation'/str(a.step)/'FUTURE'/split;dest.mkdir(parents=True,exist_ok=True)
source_indices=[10,11] if split=='FINAL' else [3,4]
if not (panelroot/'PANEL.json').exists():
    m,c,ck=runtime.load(cfg['parent'],False);probe=sampling.NoiseProbe(m)
    official=load_model('g1',device=str(m.device),text_encoder=False).eval();op=sampling.NoiseProbe(official)
    specs=[s for s in rows(L/'SPLITS.jsonl') if s['split']==split and s['group']=='W' and s['source_index'] in source_indices]
    train_families={s['source_family_id'] for s in read(ROOT/'COLLECTION_SPECS.json')['specs']}
    assert not train_families.intersection(s['source_family_id'] for s in specs)
    cases=[]
    try:
      with torch.inference_mode(),sampling.capture_context() as captured:
        for spec in specs:
          for ni,seed in enumerate(spec['seeds']):
            runner,_=initialize(m,c,probe,spec,cfg,captured);set_text(runner,spec,'walk',cfg)
            noises=sampling.fresh_noise(seed,100)
            for z in noises[:4]:runner.step(z)
            item=sampling.payload(runner,captured,runner.text,{})
            for deadline in [127,219]:
                shadow=future.shadow_of(runner);ph=[]
                for wi in range(math.ceil((deadline+1)/52)):
                    q,dec,world,_=future.official_proposal(official,op,shadow,noises[50+wi])
                    ph.append(m.motion_rep.unnormalize(dec)[0].cpu().numpy());future.shadow_accept(shadow,dec,world)
                ph=np.concatenate(ph);goal=ph[deadline,[0,2]].tolist()
                key=f'{spec["source_index"]}_n{ni}_d{deadline}'
                path=panelroot/(key+'.pt');torch.save(copy_state(item),path)
                cases.append(dict(id=key,path=str(path),goal=goal,deadline=deadline,seed=seed,
                    family=spec['source_family_id'],source=spec,split=split,
                    goal_policy='natural official endpoint; only XZ/time shown',parent_sha256=sha(cfg['parent'])))
        write(panelroot/'PANEL.json',dict(cases=cases,train_families_disjoint=True,
            initial_history='112k actual accepted history; same physical boundary for every candidate',
            conditions='same text/XZ/time/noise; no future body or route',fps=25,G=52,
            thresholds=dict(arrival_m=.2,early_progress_m=.05,leg_activity_deg=5),
            finite_horizon=True,actual_ui=False,physics=False))
    finally:probe.close();op.close()
    del official,m,c,ck,probe,op;gc.collect();torch.cuda.empty_cache()

cases=read(panelroot/'PANEL.json')['cases'];results=[]
models=dict(baseline112k=cfg['parent'],interaction105k=artifact_path('checkpoint5000_pt_0be2f0dae9'),candidate=a.checkpoint,official=cfg['parent'])
for name,path in models.items():
    # Reuse fixed baseline results on later checkpoints, never resample them.
    cached=panelroot/(name+'_RESULT.json')
    if name!='candidate' and cached.exists():results+=read(cached)['rows'];continue
    m,c,ck=runtime.load(path,False);probe=sampling.NoiseProbe(m)
    official=load_model('g1',device=str(m.device),text_encoder=False).eval() if name=='official' else None
    op=sampling.NoiseProbe(official) if official is not None else None
    modelrows=[];checkpoint_sha=sha(path)
    try:
      with torch.inference_mode():
        for case in cases:
            item=copy_state(torch.load(case['path'],map_location='cpu',weights_only=False),m.device)
            runner=sampling.Native(m,c,probe,item['target']['text'])
            history.HistoryContext(item['chunks'],item['physical'],case['parent_sha256']).prime(m,c,runner.text)
            runner.physical=item['physical'];runner.previous=item['previous'];runner.translation=item['translation']
            runner.heading=item['heading'];runner.i=item['meta']['depth']
            noise=sampling.fresh_noise(case['seed'],100);qs=[];phs=[];checks=[];receipts=[]
            origin=runner.translation[0,[0,2]].cpu().numpy();goal=np.asarray(case['goal']);distance=float(np.linalg.norm(goal-origin))
            direction=(goal-origin)/max(distance,1e-6)
            for wi,remaining in enumerate(range(case['deadline'],-1,-52)):
                previous=runner.physical.clone()
                if official is None:
                    obs=future.observation(m,case['goal'],remaining)
                    _,q,dec,world,_=runner.step(noise[4+wi],future_observation=obs)
                else:
                    q,dec,world,_=future.official_proposal(official,op,runner,noise[4+wi],case['goal'],remaining)
                    future.shadow_accept(runner,dec,world)
                ctl={} if remaining>=52 else dict(root2d=[dict(frame=remaining,xz=case['goal'])])
                check,ph=score_window(m,dec,q,previous,ck['config'],'walk',ctl)
                qs.append(q);phs.append(ph);checks.append(check)
                receipts.append(dict(window=wi,goal_relative_frame=remaining,
                    extent=max(52,4*math.ceil((remaining+1)/4)),goal_visible=True,
                    displacement_m=float(np.linalg.norm(ph[-1,[0,2]]-previous[0,-1,[0,2]].cpu().numpy()))))
            ph=np.concatenate(phs);q=np.concatenate(qs)
            arrival=float(np.linalg.norm(ph[case['deadline'],[0,2]]-goal))
            early=float((ph[51,[0,2]]-origin)@direction)
            legs=float(np.rad2deg(np.ptp(q[:,7:19],axis=0)).mean())
            geometry=all(x['geometry_pass'] and x['valid'] for x in checks)
            preparation=bool(distance<.15 or early>=min(.05,distance*.1))
            function=bool(arrival<=.2 and preparation and legs>=5)
            artifact=dest/(name+'_'+case['id']+'.npz')
            np.savez_compressed(artifact,qpos=q,physical=ph,goal=goal,deadline=case['deadline'],fps=25)
            row=dict(id=case['id'],candidate=name,checkpoint=path,checkpoint_sha256=checkpoint_sha,
                family=case['family'],seed=case['seed'],deadline=case['deadline'],arrival_error_m=arrival,
                early_progress_m=early,leg_activity_deg=legs,early_preparation=preparation,
                function_pass=function,geometry_pass=geometry,joint_pass=function and geometry,
                contact_slip_percent=statistics.mean(x['metrics']['whole']['contact_slip_percent'] for x in checks),
                windows=checks,receipts=receipts,artifact=str(artifact),actual_ui=False,physics=False)
            modelrows.append(row);print('FUTURE_EVAL',name,case['id'],round(arrival,3),round(early,3),row['joint_pass'],flush=True)
    finally:probe.close();op.close() if op is not None else None
    if name!='candidate':write(cached,dict(rows=modelrows))
    results+=modelrows
    del m,c,ck,probe,official,op;gc.collect();torch.cuda.empty_cache()
summary={}
for name in models:
    rr=[r for r in results if r['candidate']==name]
    summary[name]=dict(n=len(rr),function=sum(r['function_pass'] for r in rr),joint=sum(r['joint_pass'] for r in rr),
        mean_arrival_m=statistics.mean(r['arrival_error_m'] for r in rr),
        mean_early_progress_m=statistics.mean(r['early_progress_m'] for r in rr),
        mean_contact_slip_percent=statistics.mean(r['contact_slip_percent'] for r in rr))
write(dest/'RESULT.json',dict(rows=results,summary=summary,split=split,complete=True,
    source_scope='separate training families; existing project benchmark families; not claimed entirely unseen research test',
    physical_simulation_verified=False,quality_success_not_assumed=True))
print('SUMMARY',summary,flush=True)
