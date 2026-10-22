from common import *
import argparse,math,collections,types,copy
import numpy as np
import torch
from motion_mamba2.models import load_model
from motion_training import runtime,sampling,distributed as parallel
from motion_training.acceleration import admitted_backend,snapshot,state_tree
from motion_training.evaluation import initialize,set_text,control,text_for,R
from motion_training.collectors import plans,required_stages
from motion_training._compat.window_score import score_window
from motion_training._compat.event_scores import event_score,phase
from reference_helpers import _official_proposal,phase_quality,base_record

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--generation',type=int,required=True);a=p.parse_args()
parallel.setup();cfg=read(ROOT/'config.json');dest=ROOT/'pools'/f'g{a.generation}'/'legacy';dest.mkdir(parents=True,exist_ok=True)
from budget_process import start_budget
if parallel.leader():start_budget('collect_legacy.py_g'+str(a.generation),devices=parallel.world())
parent_sha=sha(a.checkpoint);specs=read(ROOT/'COLLECTION_SPECS.json')['specs']
specs=[s for s in specs if (s['source_index']-1000)%parallel.world()==parallel.rank()]
m,c,ck=runtime.load(a.checkpoint,False);probe=sampling.NoiseProbe(m)
# Reuse a gate only for exactly the checked parent; newer generations use reference.
gate=read(cfg['kernel_gate_path']);admit=gate.get('passed') and gate.get('checkpoint_sha256')==parent_sha
backend=admitted_backend(c,cfg if admit else dict(cfg,kernel_gate_path=str(dest/'NO_ADMISSION.json')))
official=load_model('g1',device=str(m.device),text_encoder=False).eval();teacher_probe=sampling.NoiseProbe(official)
teacher_hash=identity(official.denoiser.model.state_dict());labels=[];attempts=[];queries=0
try:
 with torch.inference_mode(),sampling.capture_context() as captured:
  for spec in specs:
   seed=spec['seeds'][0]+100003*a.generation;noises=sampling.fresh_noise(seed,100)
   runner,offset=initialize(m,c,probe,spec,cfg,captured)
   set_text(runner,spec,'walk',cfg)
   for z in noises[:4]:runner.step(z)
   qs=[];phs=[];ws=[];events=[];receipts=[];wi=0;failed=None
   for ei,(action,kind,count) in enumerate(plans(spec['group'])):
    set_text(runner,spec,action,cfg)
    if spec['source_index']%2 and action in ('walk','backwalk','wave'):
     runner.text=text_for(cfg,read(R/'TEXTS.json')[action+'_1']['prompt'],m.device)
    state={};heading=float(runner.heading[0]);start=wi
    for j in range(count):
     try:ctl,state,intent=control(runner,spec,offset,action,kind,j,state)
     except ValueError as err:
      failed=repr(err);break
     item=sampling.payload(runner,captured,runner.text,ctl)
     assert len(item['chunks'])==runner.i and sum(t.shape[1]*4 for t,_ in item['chunks'])==runner.physical.shape[1]
     before=identity(state_tree(snapshot(runner)));chunks_before=identity(captured);rng=runtime.rng()
     shadow=types.SimpleNamespace(m=m,physical=runner.physical.clone(),previous=runner.previous.clone(),translation=runner.translation.clone(),heading=runner.heading.clone(),text=runner.text.clone(),mask=runner.mask.clone(),i=runner.i)
     horizon=2 if j==0 and kind!='continue' else 1
     shadow_state=copy.deepcopy(state);tq=[];tp=[];tw=[];tc=[];contracts=[];first=None
     for look in range(horizon):
      sc=ctl
      if look:sc,shadow_state,_=control(shadow,spec,offset,action,kind,j+look,shadow_state)
      prior=shadow.physical.clone();prior_tokens=shadow.previous.clone();origin=shadow.translation.clone()
      q,dec,world=_official_proposal(official,teacher_probe,shadow,noises[4+wi+look],sc);queries+=1
      target,prefix,contract=sampling.target_check(m,world,prior_tokens,origin,dec)
      check,ph=score_window(m,dec,q,prior,ck['config'],action,sc)
      if look==0:first=(target,prefix,dec)
      tq.append(q);tp.append(ph);tw.append(check);tc.append(sc);contracts.append(contract)
      physical=m.motion_rep.unnormalize(dec)
      shadow.physical=torch.cat((shadow.physical,physical),1);shadow.previous=world.clone()
      shadow.translation=torch.zeros_like(origin);shadow.translation[:,0]=physical[:,-1,0];shadow.translation[:,2]=physical[:,-1,2]
      shadow.heading=torch.atan2(physical[:,-1,4],physical[:,-1,3]);shadow.i+=1
     runtime.restore_rng(rng)
     assert identity(state_tree(snapshot(runner)))==before,'Teacher mutated student history'
     assert identity(captured)==chunks_before,'Teacher commits leaked into capture'
     assert identity(runtime.rng())==identity(rng)
     ok,note=phase_quality(tq,tp,tw,action,kind,j,heading,tc)
     good=bool(ok and all(x['geometry_pass'] for x in tw) and all(x['passed'] for x in contracts))
     key=f'g{a.generation}_{spec["group"]}_{spec["source_index"]}_w{wi}';path=dest/(key+'.pt')
     record=base_record(spec,path,parent_sha,teacher_hash,key,action,kind,j,seed+wi,a.generation)
     record.update(qualified=good,checks=tw[0],target_contract=contracts[0],phase_response=note,shadow_windows=horizon,normalizer_version=m.motion_contract,query_preserves_main_state=True,query_preserves_rng=True,prime_chunks=len(item['chunks']),pre_query_history_frames=int(item['physical'].shape[1]),history_producer='current_student_only_after104_GT',batch_bucket=f'{record["stage_key"]}:depth{runner.i}',rejection_reasons=[] if good else dict(function=not ok,geometry=[x['geometry_pass'] for x in tw],contracts=contracts))
     if good:
      item['target'].update(target_latent=first[0].detach(),decoder_prefix=first[1].detach(),normalized=first[2].detach(),noise=noises[4+wi].to(m.device),intent=intent)
      torch.save(copy_state(item),path);record['label_sha256']=sha(path);labels.append(record)
     else:record['label']=None
     attempts.append(record)
     prior=runner.physical.clone();_,q,dec,world,_=runner.step(noises[4+wi],ctl)
     check,ph=score_window(m,dec,q,prior,ck['config'],action,ctl)
     qs.append(q);phs.append(ph);ws.append(check);receipts.append(dict(window=wi,action=action,kind=kind,phase=phase(action,kind,j),accepted_producer='student',control=ctl));wi+=1
    if failed:break
    ev=event_score(np.concatenate(qs[start:wi]),np.concatenate(phs[start:wi]),ws[start:wi],action,kind,heading,[r['control'] for r in receipts[start:wi]])
    events.append(dict(ev,start_window=start,end_window=wi))
   if qs:
    artifact=dest/f'{spec["group"]}_{spec["source_index"]}_student.npz'
    np.savez_compressed(artifact,qpos=np.concatenate(qs),physical=np.concatenate(phs),fps=25)
    write(artifact.with_suffix('.json'),dict(source=spec,events=events,receipts=receipts,failure=failed,initial_GT_frames=104,burnin_frames=208,student_only=True))
   print('COLLECT',parallel.rank(),spec['group'],spec['source_index'],'labels',len(labels),'queries',queries,flush=True)
finally:teacher_probe.close();probe.close()
write(dest/f'rank{parallel.rank()}.json',dict(labels=labels,attempts=attempts,queries=queries,backend=backend))
parallel.barrier()
if parallel.leader():
 parts=[read(dest/f'rank{i}.json') for i in range(parallel.world())];alllabels=[r for x in parts for r in x['labels']];allattempts=[r for x in parts for r in x['attempts']]
 coverage=collections.Counter(r['stage_key'] for r in alllabels)
 pool=dict(checkpoint=a.checkpoint,checkpoint_sha256=parent_sha,generation=a.generation,labels=alllabels,attempts=len(allattempts),teacher_queries=sum(x['queries'] for x in parts),coverage=dict(coverage),missing_stages=sorted(set(required_stages())-set(coverage)),qualified=len(alllabels),families=len({r['source_family_id'] for r in alllabels}),current_student_history=True,teacher_does_not_commit=True,all_query_preservation_checks=True,full_student_events_recorded=True,full_teacher_events_not_claimed=True)
 write(dest/'POOL.json',pool);print('POOL_COMPLETE',pool['qualified'],pool['coverage'],'missing',pool['missing_stages'],flush=True)
