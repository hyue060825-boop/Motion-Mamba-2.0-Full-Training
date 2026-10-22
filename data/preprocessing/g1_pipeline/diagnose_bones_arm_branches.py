"""Development-only IK branch evidence on high hinge-error clips; no fitting."""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,sha
from .gmr_soma import parse
from .locked_target import create_retargeter
from .robot import read_csv,FORMAT
from .resample import qpos as resample
from .gmr_weighted_convergence import cost


def run():
    ns='bones_soma_rootweight_dev_v6';src=ROOT/'qa/aggregate'/f'{ns}.json';previous=json.loads(src.read_text())
    selected={r['sample_id'] for r in previous['results'] if r.get('hinge_mae_deg',0)>20}
    config=ROOT/'configs/bones_soma_rootweight_v6.json';cfg=json.loads(config.read_text());results=[]
    for row in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl'):
        if row['sample_id'] not in selected:continue
        base=ROOT/'work'/ns;mid=row['source_motion_id'];q=np.load(base/(mid+'.npy'))
        tt=np.array(json.loads((base/(mid+'.time.json')).read_text())['source']['target_timestamps'])
        g=create_retargeter(config,ROOT/'assets/project_g1_v1/g1.xml')
        ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
        ref,_,_=resample(ref,rt,g.model.jnt_range[1:],end=np.nextafter(tt[-1],np.inf))
        frames,_=parse(ROOT/'work/bones_pairs_dev'/(mid+'.bvh'),{k:k for k in cfg['human_scale_table']},target_timestamps=tt,rotation_basis='world_only')
        sampled=[]
        for ix in sorted(set([0,len(q)-1]+list(range(0,len(q),25)))):
            g.update_targets(frames[ix]);values={}
            for label,pose in [('candidate',q[ix]),('reference',ref[ix])]:
                g.configuration.update(pose);data=g.configuration.data;mujoco.mj_kinematics(g.model,data)
                errors={}
                for body in ('left_shoulder_yaw_link','right_shoulder_yaw_link','left_wrist_yaw_link','right_wrist_yaw_link'):
                    human=cfg['ik_match_table2'][body][0];pos,quat=g.scaled_human_data[human];bid=g.model.body(body).id
                    errors[body]={'orientation_deg':float(np.rad2deg((R.from_matrix(data.xmat[bid].reshape(3,3)).inv()*R.from_quat(quat,scalar_first=True)).magnitude())),
                                 'position_m':float(np.linalg.norm(data.xpos[bid]-pos))}
                values[label]={'weighted_task2_cost':cost(g.configuration,g.tasks2),'bodies':errors}
            sampled.append({'frame':ix,**values})
        results.append({'sample_id':row['sample_id'],'frames':len(q),'sampled_frames':sampled})
    report={'status':'DEVELOPMENT_DIAGNOSTIC_ONLY','selection':'v6 dev clips with hinge MAE >20deg; first,last and every25th frame; no accept data',
            'denominator':200,'selected':len(selected),'source_sha256':sha(src),'config_sha256':sha(config),'code_sha256':sha(__file__),'results':results}
    write_json(ROOT/'audit/bones_arm_branch_diagnostic.json',report)
    for row in results:
        means={key:float(np.mean([f[key]['weighted_task2_cost'] for f in row['sampled_frames']])) for key in ('candidate','reference')}
        print(row['sample_id'],means,flush=True)
    return report

if __name__=='__main__':run()
