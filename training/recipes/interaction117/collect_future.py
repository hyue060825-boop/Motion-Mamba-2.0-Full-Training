from common import *
import argparse,math,copy,collections
import numpy as np
import torch
from motion_mamba2.models import load_model
from motion_training import runtime,sampling,distributed as parallel
from motion_training.evaluation import initialize,set_text
from motion_training._compat.window_score import score_window
from reference_helpers import base_record
import future_goal as future

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--generation',type=int,required=True)
a=p.parse_args();parallel.setup();cfg=read(ROOT/'config.json')
dest=ROOT/'pools'/f'g{a.generation}'/'future';dest.mkdir(parents=True,exist_ok=True)
from budget_process import start_budget
if parallel.leader():start_budget('collect_future.py_g'+str(a.generation),devices=parallel.world())
parent_hash=sha(a.checkpoint)
specs=[s for s in read(ROOT/'COLLECTION_SPECS.json')['specs'] if s['group']=='W']
specs=[s for s in specs if (s['source_index']-1000)%parallel.world()==parallel.rank()]
m,c,ck=runtime.load(a.checkpoint,False);probe=sampling.NoiseProbe(m)
official=load_model('g1',device=str(m.device),text_encoder=False).eval();op=sampling.NoiseProbe(official)
teacher_hash=identity(official.denoiser.model.state_dict());labels=[];attempts=[];queries=0

def score(shadow,dec,q,goal,rel):
    ctl={} if rel>=52 or rel is None else dict(root2d=[dict(frame=rel,xz=goal)])
    return score_window(m,dec,q,shadow.physical,ck['config'],'walk',ctl)

def query_chain(runner,noise_list,goal,remaining):
    global queries
    shadow=future.shadow_of(runner);checks=[];physical=[];qpositions=[];first=None;contracts=[];receipts=[]
    for wi in range(math.ceil((remaining+1)/52)):
        rel=remaining-wi*52
        q,dec,world,receipt=future.official_proposal(official,op,shadow,noise_list[wi],goal,rel)
        queries+=1
        target,prefix,contract=sampling.target_check(m,world,shadow.previous,shadow.translation,dec)
        check,ph=score(shadow,dec,q,goal,rel)
        if first is None:first=(target,prefix,dec)
        checks.append(check);physical.append(ph);qpositions.append(q);contracts.append(contract);receipts.append(receipt)
        future.shadow_accept(shadow,dec,world)
    ph=np.concatenate(physical);qp=np.concatenate(qpositions)
    origin=runner.translation[0,[0,2]].detach().cpu().numpy();distance=float(np.linalg.norm(np.asarray(goal)-origin))
    end_error=float(np.linalg.norm(ph[remaining,[0,2]]-goal))
    direction=(np.asarray(goal)-origin)/max(distance,1e-6)
    early=float((ph[min(51,remaining),[0,2]]-origin)@direction)
    speed=np.linalg.norm(np.diff(np.vstack([origin,ph[:,[0,2]]]),axis=0),axis=1)*25
    quality=all(x['geometry_pass'] and x['valid'] for x in checks)
    contract_ok=all(x['passed'] for x in contracts)
    prep_ok=remaining<52 or distance<.15 or early>=min(.05,distance*.1)
    passed=quality and contract_ok and end_error<=.2 and prep_ok and float(speed.max())<=2.0
    result=dict(passed=bool(passed),geometry=bool(quality),target_contract=bool(contract_ok),
        arrival_error_m=end_error,early_progress_m=early,early_preparation=bool(prep_ok),
        max_speed_m_s=float(speed.max()),deadline=remaining,checks=checks,contracts=contracts,
        windows=receipts,goal=goal)
    return first,result,ph,qp

try:
 with torch.inference_mode(),sampling.capture_context() as captured:
  for spec in specs:
   for deadline in cfg['teacher_future_horizons']:
    seed=cfg['seed']+100003*a.generation+spec['source_index']*100+deadline
    noises=sampling.fresh_noise(seed,100)
    runner,offset=initialize(m,c,probe,spec,cfg,captured);set_text(runner,spec,'walk',cfg)
    for z in noises[:4]:runner.step(z)
    # Obtain a feasible destination from an unconditioned official continuation.
    # Its body/route is NOT supplied to either timed-goal query or student.
    rng=runtime.rng();initial_count=len(captured);shadow=future.shadow_of(runner);natural=[]
    for wi in range(math.ceil((deadline+1)/52)):
        q,dec,world,_=future.official_proposal(official,op,shadow,noises[50+wi]);queries+=1
        natural.append(m.motion_rep.unnormalize(dec)[0].cpu().numpy())
        future.shadow_accept(shadow,dec,world)
    natural=np.concatenate(natural);goal=natural[deadline,[0,2]].tolist()
    origin=runner.translation[0,[0,2]].cpu().numpy();distance=float(np.linalg.norm(np.asarray(goal)-origin))
    runtime.restore_rng(rng);assert len(captured)==initial_count
    if not .15<=distance<=.8*((deadline+1)/25):
        attempts.append(dict(family=spec['source_family_id'],deadline=deadline,qualified=False,
            reason='natural endpoint lacks feasible active movement',distance_m=distance));continue
    student_ph=[];student_q=[];wi=0
    for remaining in range(deadline,-1,-52):
        item=sampling.payload(runner,captured,runner.text,{})
        item['target']['future_goal']=dict(version=future.VERSION,xz=goal,relative_frame=remaining)
        before=identity([runner.physical,runner.translation,runner.previous]);ncap=len(captured);rng=runtime.rng()
        first,result,teacher_ph,teacher_q=query_chain(runner,noises[4+wi:4+wi+8],goal,remaining)
        runtime.restore_rng(rng)
        assert before==identity([runner.physical,runner.translation,runner.previous]) and len(captured)==ncap
        key=f'g{a.generation}_F_{spec["source_index"]}_d{deadline}_w{wi}';path=dest/(key+'.pt')
        record=base_record(spec,path,parent_hash,teacher_hash,key,'walk','timed_goal',wi,seed+wi,a.generation)
        extent=max(52,4*math.ceil((remaining+1)/4))
        record.update(pool_kind='future',F=extent-52,qualified=result['passed'],future_contract=future.VERSION,
            batch_bucket=f'future:F{extent-52}:depth{runner.i}',prime_chunks=runner.i,
            teacher_chain=result,original_deadline=deadline,relative_goal_frame=remaining,
            phase='prepare' if remaining>=52 else 'arrival',
            stage_key='F|walk|timed_goal|'+('prepare' if remaining>=52 else 'arrival'),
            query_preserves_main_state=True,query_preserves_rng=True)
        if result['passed']:
            item['target'].update(target_latent=first[0].detach(),decoder_prefix=first[1].detach(),normalized=first[2].detach())
            torch.save(copy_state(item),path);record['label_sha256']=sha(path);labels.append(record)
        else:record['label']=None
        artifact=dest/(key+'_teacher.npz');np.savez_compressed(artifact,qpos=teacher_q,physical=teacher_ph,fps=25)
        record['teacher_trajectory']=str(artifact);attempts.append(record)
        obs=future.observation(m,goal,remaining)
        prior=runner.physical.clone();_,q,dec,world,_=runner.step(noises[4+wi],future_observation=obs)
        check,ph=score(runner,dec,q,goal,remaining) if False else score_window(m,dec,q,prior,ck['config'],'walk',{} if remaining>=52 else dict(root2d=[dict(frame=remaining,xz=goal)]))
        student_ph.append(ph);student_q.append(q);wi+=1
        print('FUTURE_LABEL',parallel.rank(),spec['source_index'],deadline,remaining,result['passed'],
              round(result['arrival_error_m'],3),round(result['early_progress_m'],3),flush=True)
    artifact=dest/f'{spec["source_index"]}_d{deadline}_student.npz'
    np.savez_compressed(artifact,qpos=np.concatenate(student_q),physical=np.concatenate(student_ph),goal=goal,deadline=deadline,fps=25)
    ph=np.concatenate(student_ph);write(artifact.with_suffix('.json'),dict(family=spec['source_family_id'],goal=goal,
        deadline=deadline,arrival_error_m=float(np.linalg.norm(ph[deadline,[0,2]]-goal)),
        initial_student_history_frames=312,student_only=True,teacher_commits=False,paired_goal=True))
finally:op.close();probe.close()
write(dest/f'rank{parallel.rank()}.json',dict(labels=labels,attempts=attempts,teacher_queries=queries,
    checkpoint=a.checkpoint,checkpoint_sha256=parent_hash))
parallel.barrier()
if parallel.leader():
 parts=[read(dest/f'rank{i}.json') for i in range(parallel.world())]
 ls=[x for p in parts for x in p['labels']]
 write(dest/'POOL.json',dict(labels=ls,attempts=[x for p in parts for x in p['attempts']],
    teacher_queries=sum(p['teacher_queries'] for p in parts),checkpoint=a.checkpoint,
    checkpoint_sha256=parent_hash,generation=a.generation,current_student_history=True,
    teacher_does_not_commit=True,qualified=len(ls),families=len({x['source_family_id'] for x in ls})))
 print('FUTURE_POOL_COMPLETE',len(ls),flush=True)
