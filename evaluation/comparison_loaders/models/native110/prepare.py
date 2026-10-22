"""Freeze the existing GT-only cohort and controlled goal requests; no fitting."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import collections,hashlib,json,shutil,os,sys,time
import numpy as np
from util import ROOT,MODELS,TASKS,read,write,sha,canonical,rows

OLD=Path(artifact_path('window_three_models_20261016_739644bcea'))
HEAD=OLD/'hiphi_window_head';GEN=HEAD/'generator'
AUDIT=ROOT/'source_protocol'

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'PREPARED.json').exists(),'Already frozen; verify existing plan rather than replace'
    cohort=rows(AUDIT/'LOCOMOTION_CANDIDATES_GT_ONLY.jsonl')
    original=read(AUDIT/'SUPPLEMENTAL_PROTOCOL_PLAN.json')
    assert sha(AUDIT/'LOCOMOTION_CANDIDATES_GT_ONLY.jsonl')==original['candidate_manifest_sha256']
    assert sha(AUDIT/'ALIGNED_FIXED_GOAL_REQUESTS.jsonl')==original['requests_sha256']
    assert len(cohort)==506 and len({r['actor_uid'] for r in cohort})==22
    assert collections.Counter(r['difficulty_bucket'] for r in cohort)==original['buckets']
    assert read(HEAD/'WINDOW_EVALUATOR_QUALIFICATION.json')['status']=='PASS'
    models=read(GEN/'CHECKPOINTS.json')
    models['native117']=read(Path(artifact_path('checkpoints_json_3208734ffd')))['native117']
    assert list(models)==MODELS
    ck=[]
    for m,info in models.items():
        p=Path(info['checkpoint']);assert p.is_file() and sha(p)==info['sha256'],m
        st=p.stat();info['frozen_file_stat']=dict(size=st.st_size,mtime_ns=st.st_mtime_ns)
        ck.append(dict(model=m,step=info['step'],sha256=info['sha256'],**info['frozen_file_stat']))
    write(ROOT/'CHECKPOINTS.json',models)
    sampler=ROOT/'sampler';sampler.mkdir(exist_ok=True)
    for name in ['evaluate_humanml3d_table45.py','evaluate_motion_mamba2_official_table5.py','checkpoint_loader.py','long_memory_protocol.py']:
        shutil.copy2(GEN/name,sampler/name)
    shutil.copy2(ROOT/'CHECKPOINTS.json',sampler/'CHECKPOINTS.json')
    # Reuse all independently qualified decoded-motion embedding/physical code.
    # Only the old GT-endpoint scorer is replaced with the imposed fixed goal.
    score=(OLD/'score_hiphi_job.py').read_text()
    begin=score.index('    constraints=[]\n');end=score.index('    np.savez_compressed',begin)
    fixed='''    constraints=[]
    for i,r in enumerate(rows):
        assert r['goal_joints']==[0] and len(r['goal_frames'])==1
        frame=int(r['goal_frames'][0]);goal=np.array(r['fixed_goal_root_xz_m'])
        err=float(np.linalg.norm(geo['generated']['root_positions'][i,frame,[0,2]]-goal)*100)
        constraints.append(dict(panel_id=r['panel_id'],actor_uid=r['actor_uid'],source_id=r['source_id'],
            generation_window=frame//52+1,goal_frame=frame,goal_joint=0,constraint_error_cm=err,
            target_role='imposed fixed root goal; natural reference trajectory is not a required route'))
        assert abs(err-old[i]['constraint_error_cm'])<.001
'''
    score=score[:begin]+fixed+score[end:]
    (ROOT/'score_job.py').write_text(score)
    oldcache=GEN/'text_cache';cache=ROOT/'text_cache';cache.mkdir(exist_ok=True)
    ix={r['id']:r for r in read(oldcache/'index.json')['samples']}
    emb=np.load(oldcache/'embeddings.npy',mmap_mode='r')
    np.save(cache/'embeddings.npy',np.stack([emb[ix[r['id']]['row']].copy() for r in cohort]))
    write(cache/'index.json',dict(samples=[dict(id=r['id'],row=i,caption=r['caption']) for i,r in enumerate(cohort)]))
    shutil.copy2(oldcache/'PROVENANCE.json',cache/'PARENT_PROVENANCE.json')
    manifest=[dict(id=r['id'],feature_path=r['feature_path'],frames=r['frames'],fps=25,captions=[r['caption']],actor_uid=r['actor_uid'],source_id=r['source_family_id'],source_split='generator_train_exposed',is_mirror=False) for r in cohort]
    (ROOT/'manifest.jsonl').write_text(canonical(manifest));manifest_sha=sha(ROOT/'manifest.jsonl')
    coords={};facts=[]
    for r in cohort:
        fp=Path(r['feature_path']);assert sha(fp)==r['features_hash'],r['id']
        x=np.load(fp,mmap_mode='r');s=r['crop_start']
        assert x.shape==(r['frames'],414) and s+1012<=len(x),r['id']
        seed=np.array(x[s+23,[0,2]],dtype=np.float32)
        goal=np.array(x[s+499,[0,2]],dtype=np.float32)-seed
        dist=float(np.linalg.norm(goal))
        assert abs(dist-r['gt_root_distance_20s_m'])<2e-5,(r['id'],dist)
        coords[r['id']]=goal.tolist()
        facts.append(dict(id=r['id'],actor_uid=r['actor_uid'],bucket=r['difficulty_bucket'],goal_xz_m=goal.tolist(),distance_m=dist,natural_tail_frames=len(x)-s,feature_sha256=r['features_hash']))
    write(ROOT/'COORDINATE_AUDIT.json',dict(status='PASS_ALL_506_REAL_TARGETS_AND_CROPS',N=506,actors=22,target_rule='physical root XZ at crop+499 minus crop+23; no root Y, heading, EE or path input',minimum_tail=min(x['natural_tail_frames'] for x in facts),sources=facts,time=time.time()))
    # Six fixed pilot cases: two from each GT-only distance stratum, not output-selected.
    pilot=[]
    for b in original['buckets']:pilot += [r['id'] for r in cohort if r['difficulty_bucket']==b][:2]
    request_lookup={(r['source_id'],r['generated_frames']):r for r in rows(AUDIT/'ALIGNED_FIXED_GOAL_REQUESTS.jsonl')}
    catalogs=[];specs={};allpanels={}
    paneldir=ROOT/'panels';paneldir.mkdir(exist_ok=True)
    for task in TASKS:
        aligned=task.startswith('aligned')
        generated=52*int(task[-2:]) if aligned else 572
        goal=generated-1 if aligned else int(task.split('f')[1])
        specs[task]=dict(generated_frames=generated,total_frames=generated+24,goal_generated_frame=goal,goal_window=goal//52+1,within_window_goal_frame=goal%52,effective_current_window_frames=goal%52+1,total_duration_sec=(generated+24)/25,goal_time_including_prefix_sec=(goal+25)/25,generated_duration_sec=generated/25)
        panel=[]
        for r in cohort:
            if aligned:
                req=request_lookup[r['id'],generated]
                assert req['goal_generated_frame']==goal and req['within_window_goal_frame']==51
            noise=int(hashlib.sha256(f'controlled_20261017:generator_seed0:{r["id"]}'.encode()).hexdigest()[:15],16)
            panel.append(dict(id=r['id'],panel_id=f'controlled:{r["id"]}:{r["crop_start"]}:{task}',actor_uid=r['actor_uid'],source_id=r['source_family_id'],feature_path=r['feature_path'],caption=r['caption'],crop_start=r['crop_start'],crop_end=r['crop_start']+generated+24,total_frames=generated+24,history_frames=24,horizon_frames=52,selection_span_frames=1012,goal_frames=[goal],goal_joints=[0],fixed_goal_root_xz_m=coords[r['id']],goal_coordinate_source_frame=r['crop_start']+499,noise_seed=noise,augmentation_seed=20260831,source_manifest_sha256=manifest_sha,protocol='out_of_horizon',split_role='posthoc_controlled_diagnostic',difficulty_bucket=r['difficulty_bucket'],goal_distance_m=r['gt_root_distance_20s_m'],task=task))
        assert len({tuple(p['fixed_goal_root_xz_m']) for p in panel})>450
        for shard,start in enumerate(range(0,506,32)):
            pp=paneldir/f'{task}_shard{shard:02d}.jsonl';pp.write_text(canonical(panel[start:start+32]))
            catalogs.append(dict(task=task,shard=shard,panel=str(pp),N=len(panel[start:start+32]),sha256=sha(pp)))
        pp=paneldir/f'{task}_pilot.jsonl';pp.write_text(canonical([next(p for p in panel if p['id']==sid) for sid in pilot]))
        allpanels[task]=panel
    for i in range(506):
        for task in TASKS:
            a=allpanels[task][i];b=allpanels[TASKS[0]][i]
            assert all(a[k]==b[k] for k in ['id','caption','crop_start','noise_seed','fixed_goal_root_xz_m','difficulty_bucket'])
    plan=dict(status='FROZEN_AWAITING_GPU_GATES',created=time.time(),models=models,N_sources=506,actors=22,buckets=original['buckets'],tasks=specs,catalog=catalogs,shard_size=32,batch_size=2,pilot_ids=pilot,paired_generator_realizations=1,source_protocol=original,phase_design='Four prescribed goal deadlines 455/475/507/527; common complete572-frame rollout, score at requested goal frame, retain later frames only for continuity diagnostics',metrics=dict(primary='terminal-only root XZ constraint error in cm; success at10/20/50cm; near/medium/far plus actor/source paired intervals',no_root_ADE=True,quality='Reuse unchanged complete1024 four-model local52 FID/R@3 study with explicit provenance; new root-only506 task gets physical/seam diagnostics and conditional local52 scores only if qualified; far27 no FID ranking'),cards=[4,5,6,7],disk_floor_GiB=20,max_processes_per_gpu=1,training_unchanged=True,online_unchanged=True,source_exposure='Known generator training materials, post-hoc protocol diagnosis; not a new unseen benchmark',prepared_hashes={str(p.relative_to(ROOT)):sha(p) for p in list(sampler.glob('*.py'))+[ROOT/'manifest.jsonl',ROOT/'CHECKPOINTS.json',cache/'embeddings.npy',cache/'index.json']})
    plan['execution_source_hashes']={n:sha(ROOT/n) for n in ['util.py','run_job.py','score_job.py','controller.py','aggregate.py']}
    write(ROOT/'PLAN.json',plan)
    write(ROOT/'PREPARED.json',dict(status='PASS_CPU_COHORT_GOALS_SEEDS_AND_CHECKPOINT_IDENTITIES',plan_sha256=sha(ROOT/'PLAN.json'),checkpoints=ck,N_requests=506*7*4,formal_jobs=len(catalogs)*4,coordinate_audit_sha256=sha(ROOT/'COORDINATE_AUDIT.json'),time=time.time()))
    print('PREPARED',506*7*4,'trajectories',len(catalogs)*4,'formal jobs',flush=True)

if __name__=='__main__':main()
