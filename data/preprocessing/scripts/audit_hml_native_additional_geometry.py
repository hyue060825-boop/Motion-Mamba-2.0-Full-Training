"""Independent BodyModel checks for a male and a non-integer-FPS train clip."""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT,HML,read_jsonl,write_json,sha
from g1_pipeline.hml_native_runner import native_parameters
from g1_pipeline.hml_native import native_pose_to_frames,sample_local,SMPLH_IDS
from g1_pipeline.resample import time_grid


def run():
    sys.path.insert(0,str(HML))
    import torch
    from human_body_prior.body_model.body_model import BodyModel
    rows=read_jsonl(ROOT/'manifests/hml_smoke100.jsonl')
    selected=[next(r for r in rows if r['native_metadata']['gender']=='male'),
              next(r for r in rows if abs(r['original_fps']-round(r['original_fps']))>1e-6)]
    results=[]
    for row in selected:
        interval=row['native_interval'];path=HML/row['source_paths']['native'];p=native_parameters(path,start_frame=interval['native_first_index'],end_frame=interval['native_end_index_exclusive'])
        st=(p['start_frame']+np.arange(len(p['full_pose'])))/p['source_fps'];tt,mapping=time_grid(st,end=interval['end_sec'])
        frames=native_pose_to_frames(p['full_pose'],p['trans'],p['parents'],p['offsets'],p['source_fps'],tt,start_frame=p['start_frame'])
        local=R.from_rotvec(p['full_pose'].reshape(-1,3)).as_matrix().reshape(-1,52,3,3)
        sampled,trans,_=sample_local(local,p['trans'],p['offsets'],st,tt);poses=R.from_matrix(sampled.reshape(-1,3,3)).as_rotvec().reshape(-1,156)
        dmpls=np.column_stack([np.interp(tt,st,p['dmpls'][:,i]) for i in range(8)]);g=p['geometry']
        bm=BodyModel(bm_fname=g['model_path'],dmpl_fname=g['dmpl_path'],num_betas=16,num_dmpls=8);oracle=[]
        for start in range(0,len(tt),16):
            sl=slice(start,start+16);a=torch.tensor(poses[sl]).float();n=len(a)
            with torch.no_grad():
                obj=bm(root_orient=a[:,:3],pose_body=a[:,3:66],pose_hand=a[:,66:],trans=torch.tensor(trans[sl]).float(),
                    betas=torch.tensor(np.tile(p['betas'],(n,1))).float(),dmpls=torch.tensor(dmpls[sl]).float())
            oracle.append(obj.Jtr.numpy())
        oracle=np.concatenate(oracle);errors={}
        for name,index in SMPLH_IDS.items():
            error=np.linalg.norm(np.array([f[name][0] for f in frames])-oracle[:,index],axis=1)
            errors[name]={'max_m':float(error.max()),'mean_m':float(error.mean())}
        maximum=max(x['max_m'] for x in errors.values())
        results.append({'sample_id':row['sample_id'],'status':'PASS' if maximum<1e-5 else 'FAIL','frames':len(tt),'native_fps':p['source_fps'],
            'gender':p['gender'],'landmark_errors':errors,'source_sha256':sha(path),'model_sha256':sha(g['model_path']),'dmpl_model_sha256':sha(g['dmpl_path'])})
        print(row['sample_id'],len(tt),p['gender'],p['source_fps'],maximum,flush=True)
    write_json(ROOT/'audit/hml_native_additional_geometry_v3.json',{'status':'PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',
        'selection':'first male and first non-integer native FPS in unchanged fixed HML100 order','robot_retargeting_run':False,'results':results,
        'code_sha256':sha(__file__),'bodymodel_code_sha256':sha(HML/'human_body_prior/body_model/body_model.py')})


if __name__=='__main__':run()
