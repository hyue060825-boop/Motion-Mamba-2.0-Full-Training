"""Physical-time continuity evidence for candidate AND official development data."""
import json
import numpy as np
import mujoco
from .common import ROOT,BONES,read_jsonl,sha,write_json
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample


def metrics(m,q,t):
    if len(t)<3 or q.shape!=(len(t),m.nq) or not np.isfinite(q).all():raise ValueError('INVALID_CONTINUITY_INPUT')
    dt=np.diff(t)
    if np.max(np.abs(dt-.04))>1e-6:raise ValueError('NON_25HZ_TIME')
    velocity=np.empty((len(q)-1,m.nv))
    for i,d in enumerate(dt):mujoco.mj_differentiatePos(m,velocity[i],float(d),q[i],q[i+1])
    acceleration=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
    joint_abs=np.abs(velocity[:,6:]);joint_acc=np.abs(acceleration[:,6:])
    return {'frames':len(q),'duration_last_sample_s':float(t[-1]-t[0]),
            'timestep_max_error_s':float(np.max(np.abs(dt-.04))),
            'hinge_speed_p95_rad_s':np.percentile(joint_abs,95,axis=0).tolist(),
            'hinge_speed_max_rad_s':joint_abs.max(axis=0).tolist(),
            'hinge_acceleration_p95_rad_s2':np.percentile(joint_acc,95,axis=0).tolist(),
            'hinge_acceleration_max_rad_s2':joint_acc.max(axis=0).tolist(),
            'root_translation_speed_p95_m_s':float(np.percentile(np.linalg.norm(velocity[:,:3],axis=1),95)),
            'root_angular_speed_p95_rad_s':float(np.percentile(np.linalg.norm(velocity[:,3:6],axis=1),95)),
            'root_linear_acceleration_max_m_s2':float(np.max(np.linalg.norm(acceleration[:,:3],axis=1))),
            'unit_quaternion_max_error':float(np.max(np.abs(np.linalg.norm(q[:,3:7],axis=1)-1))),
            'hinge_limit_excess_max_rad':float(max(0,np.max(m.jnt_range[1:,0]-q[:,7:]),np.max(q[:,7:]-m.jnt_range[1:,1]))),
            'status':'MEASURED','hardware_and_relative_anomaly_gate':None,
            'reason':'hardware envelope and action-stratified baseline thresholds are not frozen'}


def run(namespace='bones_soma_rest_dev_v4'):
    if not namespace.replace('_','').isalnum():raise ValueError('INVALID_DIAGNOSTIC_NAMESPACE')
    m,asset=compile_target();manifest=ROOT/'manifests/bones_pairs_dev.jsonl';results=[]
    root=ROOT/'work'/namespace
    for row in read_jsonl(manifest):
        entry={'sample_id':row['sample_id'],'status':'ERROR'}
        try:
            mid=row['source_motion_id'];path=root/(mid+'.npy');tp=root/(mid+'.time.json')
            q=np.load(path);t=np.array(json.loads(tp.read_text())['source']['target_timestamps'])
            rp=BONES/row['source_paths']['g1'];rq,rt,_=read_csv(rp,format_id=FORMAT,fps=row['original_fps'])
            rq,_,_=resample(rq,rt,m.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
            entry.update(status='MEASURED',candidate=metrics(m,q,t),reference=metrics(m,rq,t),
              candidate_sha256=sha(path),reference_sha256=sha(rp),time_sha256=sha(tp),category=row.get('category'))
        except Exception as exc:entry['error']=repr(exc)
        results.append(entry)
    report={'status':'DEVELOPMENT_MEASUREMENTS','namespace':namespace,'expected':200,'measured':sum(r['status']=='MEASURED' for r in results),
      'input_manifest_sha256':sha(manifest),'asset_hash':asset['asset_hash'],'method':'mujoco.mj_differentiatePos for free root and hinges; successive midpoint velocities for acceleration',
      'joint_names':[m.joint(j).name for j in range(1,m.njnt)],'results':results}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_continuity.json',report)
    print(report['status'],report['measured'],flush=True);return report
def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace',required=True)
    args=parser.parse_args(argv)
    return run(args.namespace)

if __name__=='__main__':main()
