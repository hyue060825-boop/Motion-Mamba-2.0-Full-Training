"""Full-denominator development comparison of two first-frame alignments."""
import argparse,json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,sha
from .robot import read_csv,compile_target,FORMAT
from .resample import qpos as resample
from .initial_planar_alignment import initial_alignment

LIMITS={'root_rmse_aligned_m':.15,'root_orientation_mean_deg':20.,'torso_orientation_mean_deg':20.,
        'left_foot_mean_m':.08,'right_foot_mean_m':.08,'left_wrist_mean_m':.15,'right_wrist_mean_m':.15}


def run(namespace):
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';source=ROOT/'qa/aggregate'/f'{namespace}_qa_summary.json'
    original=json.loads(source.read_text());previous={r['sample_id']:r for r in original['clips']}
    m,asset=compile_target();d=mujoco.MjData(m)
    bodies=[m.body(n).id for n in ['left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link','torso_link']]
    def fk(q):
        pos=[];rot=[]
        for frame in q:
            d.qpos[:]=frame;mujoco.mj_kinematics(m,d);pos.append(d.xpos[bodies].copy());rot.append(d.xmat[bodies[-1]].reshape(3,3).copy())
        return np.array(pos),np.array(rot)
    results=[]
    for row in read_jsonl(manifest):
        sid=row['sample_id'];mid=row['source_motion_id'];base=ROOT/'work'/namespace
        entry={'sample_id':sid,'status':'ERROR'}
        try:
            path=base/(mid+'.npy');q=np.load(path);t=np.array(json.loads((base/(mid+'.time.json')).read_text())['source']['target_timestamps'])
            ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps'])
            ref,tt,_=resample(ref,rt,m.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
            if q.shape!=ref.shape or not np.allclose(t,tt,atol=1e-10,rtol=0):raise ValueError('PAIR_TIME_MISMATCH')
            align=initial_alignment(q[0],ref[0]);turn=R.from_rotvec([0,0,align['yaw_rad']]);shift=np.array(align['translation'])
            cp,cr=fk(q);rp,rr=fk(ref)
            dif=np.linalg.norm(turn.apply(cp.reshape(-1,3)).reshape(cp.shape)+shift-rp,axis=-1)
            metrics={'root_rmse_aligned_m':float(np.sqrt(np.mean(np.sum((turn.apply(q[:,:3])+shift-ref[:,:3])**2,axis=1)))),
                     'root_orientation_mean_deg':float(np.rad2deg((R.from_quat(ref[:,3:7],scalar_first=True).inv()*turn*R.from_quat(q[:,3:7],scalar_first=True)).magnitude()).mean()),
                     'torso_orientation_mean_deg':float(np.rad2deg((R.from_matrix(rr).inv()*turn*R.from_matrix(cr)).magnitude()).mean())}
            metrics.update({name+'_mean_m':float(dif[:,j].mean()) for j,name in enumerate(['left_foot','right_foot','left_wrist','right_wrist'])})
            failures=[k for k in previous[sid]['measured_pairwise_grace_failures'] if k not in LIMITS]
            failures.extend(k for k,limit in LIMITS.items() if metrics[k]>limit)
            entry.update(status='MEASURED',alternative_alignment=align,alternative_metrics=metrics,
                         original_failures=previous[sid]['measured_pairwise_grace_failures'],alternative_failures=failures,
                         original_metrics={k:previous[sid]['metrics'][k] for k in LIMITS},candidate_sha256=sha(path))
        except Exception as exc:entry['error']=repr(exc)
        results.append(entry)
    measured=[r for r in results if r['status']=='MEASURED']
    report={'status':'DEVELOPMENT_ALIGNMENT_COMPARISON_ONLY','namespace':namespace,'expected':len(read_jsonl(manifest)),
            'measured':len(measured),'errors':len(results)-len(measured),
            'original_grace_pass':original['pairwise_grace_pass_count'],
            'alternative_grace_pass':sum(not r['alternative_failures'] for r in measured),
            'metric_thresholds_unchanged':True,'contract_contacts_geometry_hinge_metrics':'unchanged; inherited from existing paired QA',
            'preregistered_design':'audit/initial_alignment_v2_development_design.json',
            'asset_hash':asset['asset_hash'],'dependencies':{str(p):sha(p) for p in [manifest,source,ROOT/'g1_pipeline/initial_planar_alignment.py',ROOT/'g1_pipeline/diagnose_planar_alignment.py']},'results':results}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_alignment_comparison.json',report)
    print({k:v for k,v in report.items() if k not in ('results','dependencies')},flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--namespace',required=True);a=p.parse_args();run(a.namespace)
