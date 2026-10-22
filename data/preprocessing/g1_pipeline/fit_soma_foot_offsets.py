"""Fit only constant source-foot position offsets on the designated development set.

Rotation offsets, scale, weights and timestamps remain fixed. Held-out acceptance
is never opened. Fits are clip-balanced robust estimates in corrected foot frames.
"""
import copy,json
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,sha,now
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample_qpos
from .gmr_soma import parse


def run():
    cfgpath=ROOT/'configs/bones_soma_rest_v4.json';cfg=json.loads(cfgpath.read_text())
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';rows=read_jsonl(manifest)
    m,asset=compile_target();data=mujoco.MjData(m)
    bodies={'left_ankle_roll_link':'LeftFoot','right_ankle_roll_link':'RightFoot'}
    values={b:[] for b in bodies};ids=[];references={}
    for row in rows:
        if row['project_split']!='train' or row['adapter_split']!='dev':raise ValueError('DEVELOPMENT_ONLY')
        mid=row['source_motion_id'];base=ROOT/'work/bones_soma_rest_dev_v4'
        timestamps=np.asarray(json.loads((base/(mid+'.time.json')).read_text())['source']['target_timestamps'])
        frame,_=parse(ROOT/'work/bones_pairs_dev'/(mid+'.bvh'),{'Hips':'Hips','LeftFoot':'LeftFoot','RightFoot':'RightFoot'},target_timestamps=timestamps,rotation_basis='world_only')
        path=BONES/row['source_paths']['g1'];ref,t,_=read_csv(path,format_id=FORMAT,fps=row['original_fps'])
        ref,_,_=resample_qpos(ref,t,m.jnt_range[1:],end=np.nextafter(timestamps[-1],np.inf))
        current={b:[] for b in bodies}
        for source,q in zip(frame,ref):
            data.qpos[:]=q;mujoco.mj_forward(m,data)
            hp=source['Hips'][0];root_scale=cfg['human_scale_table']['Hips']
            for body,name in bodies.items():
                rotation=R.from_quat(source[name][1],scalar_first=True)*R.from_quat(cfg['ik_match_table1'][body][4],scalar_first=True)
                scaled=root_scale*hp+cfg['human_scale_table'][name]*(source[name][0]-hp)
                desired=rotation.inv().apply(data.xpos[m.body(body).id]-scaled)
                current[body].append(desired)
        for body in bodies:values[body].append(np.asarray(current[body]))
        ids.append(row['sample_id']);references[row['sample_id']]=sha(path)
    report={'status':'DEVELOPMENT_FIT_NOT_ACCEPTED','time':now(),'input_manifest_sha256':sha(manifest),
            'development_samples':ids,'reference_hashes':references,'input_config_sha256':sha(cfgpath),
            'asset_hash':asset['asset_hash'],'method':'constant local position offset; clip-balanced Huber residual scale 2 cm',
            'unchanged':['orientation offsets','scale','IK weights','target XML','ground strategy','time map'],
            'fits':{}}
    fitted=copy.deepcopy(cfg)
    for body,clips in values.items():
        old=np.asarray(cfg['ik_match_table1'][body][3]);samples=np.concatenate(clips)
        # Equal clip mass before robustification; long recordings cannot dominate.
        weights=np.concatenate([np.full(len(v),1/np.sqrt(len(v))) for v in clips])
        # Use explicit Huber loss prior to clip weighting so the cutoff remains 2 cm.
        def residual(x):
            diff=samples-x;scaled=diff/.02;loss=np.where(np.abs(scaled)<=1,.5*scaled**2,np.abs(scaled)-.5)
            return (np.sign(diff)*np.sqrt(2*loss)*.02*weights[:,None]).reshape(-1)
        fit=least_squares(residual,old,ftol=1e-12,gtol=1e-12,xtol=1e-12)
        if not fit.success:raise RuntimeError('OFFSET_FIT_FAILED')
        new=fit.x
        for table in ('ik_match_table1','ik_match_table2'):fitted[table][body][3]=new.tolist()
        report['fits'][body]={'old_offset_m':old.tolist(),'new_offset_m':new.tolist(),
            'delta_m':(new-old).tolist(),'delta_norm_m':float(np.linalg.norm(new-old)),
            'clip_macro_target_position_error_before_m':float(np.mean([np.linalg.norm(v-old,axis=1).mean() for v in clips])),
            'clip_macro_target_position_error_after_m':float(np.mean([np.linalg.norm(v-new,axis=1).mean() for v in clips])),
            'per_clip_median_offsets_m':[np.median(v,axis=0).tolist() for v in clips]}
    destination=ROOT/'configs/bones_soma_footfit_v5.json';write_json(destination,fitted)
    report['output_config_sha256']=sha(destination)
    write_json(ROOT/'audit/soma_foot_offset_development_fit_v5.json',report)
    print({body:{k:v for k,v in fit.items() if k!='per_clip_median_offsets_m'} for body,fit in report['fits'].items()},flush=True)
    return report
if __name__=='__main__':run()
