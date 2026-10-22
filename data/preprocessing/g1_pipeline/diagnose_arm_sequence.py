"""Full-clip development test of human-only arm seeds; never a release runner."""
import io
import json
import numpy as np
from .common import ROOT, BONES, read_jsonl, write_json, sha, atomic_bytes
from .gmr_soma import parse
from .locked_target import create_retargeter, initialize_root_from_human
from .gmr_arm_seed import arm_seed
from .gmr_weighted_convergence import retarget, converge, cost
from .bones_continuity_qa import metrics as continuity
from .robot import read_csv, FORMAT
from .resample import qpos as resample


def run():
    selection = ROOT/'audit/bones_arm_branch_diagnostic.json'
    cfgpath = ROOT/'configs/bones_soma_weighted_convergence_v8.json'
    cfg = json.loads(cfgpath.read_text())
    rows = {r['sample_id']:r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl')}
    selected = json.loads(selection.read_text())['results']
    output = ROOT/'work/bones_arm_sequence_v10'
    output.mkdir(exist_ok=True)
    design = {'status':'DEVELOPMENT_ONLY', 'selection_sha256':sha(selection),
              'config_sha256':sha(cfgpath), 'selected_ids':[x['sample_id'] for x in selected],
              'trigger_arm_weighted_squared_error':100.,
              'rule':'first frame or arm weighted squared residual >100 after both GMR stages; bounded arm seeds followed by GMR stage2',
              'no_reference_input_to_optimizer':True, 'production_adoption':False,
              'code_hashes':{p.name:sha(p) for p in [ROOT/'g1_pipeline/diagnose_arm_sequence.py',ROOT/'g1_pipeline/gmr_arm_seed.py',ROOT/'g1_pipeline/gmr_weighted_convergence.py']}}
    write_json(output/'design.json',design)
    results=[]
    for item in selected:
        row=rows[item['sample_id']];mid=row['source_motion_id']
        base=ROOT/'work/bones_soma_rootweight_dev_v6'
        time_path=base/(mid+'.time.json')
        times=np.asarray(json.loads(time_path.read_text())['source']['target_timestamps'])
        bvh=ROOT/'work/bones_pairs_dev'/(mid+'.bvh')
        frames,_=parse(bvh,{n:n for n in cfg['human_scale_table']},target_timestamps=times,rotation_basis='world_only')
        solver=create_retargeter(cfgpath,ROOT/'assets/project_g1_v1/g1.xml')
        initialize_root_from_human(solver,frames[0])
        poses=[];events=[];objective=[]
        for index,frame in enumerate(frames):
            pose,_=retarget(solver,frame,cfg['weighted_convergence'])
            seeded=False
            for side in ('left','right'):
                tasks=[solver.human_body_to_task2[solver.ik_match_table2[f'{side}_{p}_link'][0]] for p in ('shoulder_yaw','elbow','wrist_yaw')]
                if index==0 or cost(solver.configuration,tasks)>100.:
                    _,info=arm_seed(solver,side)
                    events.append({'frame':index,**info});seeded=True
            if seeded:converge(solver,solver.tasks2,**cfg['weighted_convergence'])
            poses.append(solver.configuration.q.copy())
            objective.append(cost(solver.configuration,solver.tasks2))
        q=np.asarray(poses)
        buf=io.BytesIO();np.save(buf,q);path=output/(mid+'.npy');atomic_bytes(path,buf.getvalue())
        # Official reference is loaded only after the entire candidate is finalized.
        ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
        ref,tt,_=resample(ref,rt,solver.model.jnt_range[1:],end=np.nextafter(times[-1],np.inf))
        if q.shape!=ref.shape or not np.allclose(tt,times,atol=1e-10,rtol=0):raise ValueError('TIME_MISMATCH')
        prior=np.load(base/(mid+'.npy'))
        entry={'sample_id':row['sample_id'],'frames':len(q),'status':'MEASURED',
               'hinge_mae_deg':float(np.rad2deg(abs(q[:,7:]-ref[:,7:])).mean()),
               'baseline_hinge_mae_deg':float(np.rad2deg(abs(prior[:,7:]-ref[:,7:])).mean()),
               'objective_mean':float(np.mean(objective)), 'objective_max':float(max(objective)),
               'candidate_continuity':continuity(solver.model,q,times),
               'baseline_continuity':continuity(solver.model,prior,times),
               'reference_continuity':continuity(solver.model,ref,times),
               'seed_events':events,'candidate_sha256':sha(path),'source_sha256':sha(bvh),
               'time_sha256':sha(time_path)}
        results.append(entry)
        write_json(ROOT/'audit/bones_arm_sequence_v10.json',{'design':design,'results':results,'expected':len(selected),'status':'DEVELOPMENT_ONLY'})
        print(row['sample_id'],entry['baseline_hinge_mae_deg'],entry['hinge_mae_deg'],
              'max_speed',max(entry['candidate_continuity']['hinge_speed_max_rad_s']),
              'seed_events',len(events),flush=True)
    return results


if __name__=='__main__':run()
