"""Source-only high-frequency inspection, without any robot motion solve."""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT,HML,read_jsonl,write_json,sha
from g1_pipeline.hiphi_native import read_bvh,LANDMARKS
from g1_pipeline.spectral_qa import local_rotation_spectrum,velocity_spectrum


def run(source):
    manifest=ROOT/'manifests'/('hml_smoke100.v2.jsonl' if source=='humanml3d' else 'hiphi_smoke100.jsonl');results=[]
    for row in read_jsonl(manifest):
        if source=='humanml3d':
            path=HML/row['source_paths']['native'];interval=row['native_interval']
            with np.load(path,allow_pickle=False) as native:
                first,end=interval['native_first_index'],interval['native_end_index_exclusive'];pose=native['poses'][first:end,:66].reshape(-1,22,3)
                trans=native['trans'][first:end];fps=row['original_fps'];rotation=R.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(len(pose),22,3,3)
        else:
            path=ROOT/'work/hiphi_packages'/(row['source_motion_id']+'.bvh');native=read_bvh(path);fps=native['fps']
            ids=[i for i,n in enumerate(native['nodes']) if n['name'] in LANDMARKS];rotation=native['local_rotations'][:,ids];trans=native['local_positions_m'][:,0]
        joint=local_rotation_spectrum(rotation,fps);root=velocity_spectrum(np.diff(trans,axis=0)*fps,fps,rms_floor=.01)
        results.append({'sample_id':row['sample_id'],'native_sha256':sha(path),'joint_angular_velocity':joint,'root_translation_velocity':root})
        print(row['sample_id'],joint['review_required'],flush=True)
    report={'status':'NATIVE_SPECTRAL_MEASUREMENTS_ONLY','source':source,'expected':100,'measured':len(results),
        'angular_review_clips':sum(r['joint_angular_velocity']['review_required'] for r in results),
        'root_review_clips':sum(r['root_translation_velocity']['review_required'] for r in results),
        'angular_high_band_rms_max_rad':max(r['joint_angular_velocity'].get('high_band_displacement_rms_max',0) for r in results),
        'clip_p95_angular_high_band_rms_rad':float(np.percentile([r['joint_angular_velocity'].get('high_band_displacement_rms_max',0) for r in results],95)),
        'source_manifest_sha256':sha(manifest),'module_sha256':sha(ROOT/'g1_pipeline/spectral_qa.py'),'results':results,
        'robot_retargeting_run':False,'filter_applied':False}
    write_json(ROOT/'audit'/f'{source}_native_spectrum_v1.json',report);print({k:v for k,v in report.items() if k!='results'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['humanml3d','hiphi'],required=True);run(p.parse_args().source)
