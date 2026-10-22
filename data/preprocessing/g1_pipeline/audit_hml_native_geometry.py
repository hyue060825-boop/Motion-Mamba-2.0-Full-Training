"""Compare native human FK with independent BodyModel; no robot retargeting."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .common import ROOT,HML,read_jsonl,write_json,sha
from .hml_native_runner import native_parameters,estimate_rest_height
from .hml_native import native_pose_to_frames,sample_local,SMPLH_IDS
from .resample import time_grid


def run():
    sys.path.insert(0,str(HML))
    import torch
    from human_body_prior.body_model.body_model import BodyModel
    row=read_jsonl(ROOT/'manifests/hml_smoke100.jsonl')[0];interval=row['native_interval'];path=HML/row['source_paths']['native']
    p=native_parameters(path,start_frame=interval['native_first_index'],end_frame=interval['native_end_index_exclusive'])
    st=(p['start_frame']+np.arange(len(p['full_pose'])))/p['source_fps'];tt,mapping=time_grid(st,end=interval['end_sec'])
    frames=native_pose_to_frames(p['full_pose'],p['trans'],p['parents'],p['offsets'],p['source_fps'],tt,start_frame=p['start_frame'])
    source_rot=Rotation.from_rotvec(p['full_pose'].reshape(-1,3)).as_matrix().reshape(-1,52,3,3)
    sampled,trans,_=sample_local(source_rot,p['trans'],p['offsets'],st,tt)
    poses=Rotation.from_matrix(sampled.reshape(-1,3,3)).as_rotvec().reshape(-1,156)
    dmpls=np.column_stack([np.interp(tt,st,p['dmpls'][:,i]) for i in range(8)])
    geom=p['geometry'];bm=BodyModel(bm_fname=geom['model_path'],dmpl_fname=geom['dmpl_path'],num_betas=16,num_dmpls=8)
    oracle=[]
    for start in range(0,len(tt),16):
        sel=slice(start,start+16);a=torch.tensor(poses[sel]).float();n=len(a)
        with torch.no_grad():
            out=bm(root_orient=a[:,:3],pose_body=a[:,3:66],pose_hand=a[:,66:],trans=torch.tensor(trans[sel]).float(),
                   betas=torch.tensor(np.tile(p['betas'],(n,1))).float(),dmpls=torch.tensor(dmpls[sel]).float())
        oracle.append(out.Jtr.numpy())
    oracle=np.concatenate(oracle);errors={}
    for name,index in SMPLH_IDS.items():
        actual=np.array([f[name][0] for f in frames]);err=np.linalg.norm(actual-oracle[:,index],axis=1)
        errors[name]={'mean_m':float(err.mean()),'p95_m':float(np.percentile(err,95)),'max_m':float(err.max()),'native_joint_index':index}
    height=estimate_rest_height(path)
    sources=[path,ROOT/'manifests/hml_smoke100.jsonl',ROOT/'g1_pipeline/hml_native.py',ROOT/'g1_pipeline/hml_native_runner.py',
             ROOT/'g1_pipeline/audit_hml_native_geometry.py',geom['model_path'],geom['dmpl_path'],
             HML/'human_body_prior/body_model/body_model.py',HML/'human_body_prior/body_model/lbs.py']
    report={'status':'NATIVE_FK_PASS' if max(v['max_m'] for v in errors.values())<1e-5 else 'FAIL',
            'scope':'one full native train clip; native human geometry only, not robot mapping, calibration, or HML100',
            'sample_id':row['sample_id'],'native_frames':len(st),'sampled_frames':len(tt),'output_fps':25,
            'mapping':mapping,'world_basis':'identity: preserve AMASS capture world and native local axes',
            'shape':'matched SMPL-H with 16 betas and time-varying 8 DMPL coefficients',
            'height_m':height,'height_strategy':'canonical shaped mesh head-to-sole Y extent, DMPL zero',
            'per_landmark_error':errors,'oracle':'independent human_body_prior.BodyModel at resampled native parameters',
            'semantic_source':'SMPLH body indices 0..21; wrists 20/21, elbows 18/19, shoulders 16/17; no finger proxy',
            'dependencies':{str(v):sha(v) for v in sources},'robot_retarget_executed':False,
            'robot_mapping_validation':None,'robot_mapping_reason':'source-specific rest offsets, shape scaling and compatibility not frozen'}
    write_json(ROOT/'audit/hml_native_geometry_v2.json',report)
    print(report['status'],len(tt),'max_landmark_m',max(v['max_m'] for v in errors.values()),'mesh_height_m',height,flush=True)
    return report

if __name__=='__main__':run()
