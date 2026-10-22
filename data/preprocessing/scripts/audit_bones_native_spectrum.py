"""BONES dev200 source-rate frequencies; no retarget or filtering changes."""
import numpy as np
import mujoco
from g1_pipeline.common import ROOT,BONES,read_jsonl,write_json,sha
from g1_pipeline.robot import compile_target,read_csv,FORMAT
from g1_pipeline.spectral_qa import velocity_spectrum


def run():
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';m,_=compile_target();results=[]
    for row in read_jsonl(manifest):
        path=BONES/row['source_paths']['g1'];q,t,fps=read_csv(path,format_id=FORMAT,fps=row['original_fps']);velocity=np.empty((len(q)-1,35))
        for i in range(len(velocity)):mujoco.mj_differentiatePos(m,velocity[i],1/fps,q[i],q[i+1])
        joint=velocity_spectrum(velocity[:,6:],fps);root=velocity_spectrum(velocity[:,:6],fps)
        results.append({'sample_id':row['sample_id'],'source_sha256':sha(path),'joint_velocity':joint,'root_velocity':root})
    report={'status':'NATIVE_DEVELOPMENT_SPECTRAL_AUDIT','expected':200,'measured':len(results),'results':results,
        'joint_spectral_review_clips':sum(r['joint_velocity']['review_required'] for r in results),
        'root_spectral_review_clips':sum(r['root_velocity']['review_required'] for r in results),'source_manifest_sha256':sha(manifest),
        'module_sha256':sha(ROOT/'g1_pipeline/spectral_qa.py'),'filter_applied':False,'robot_retargeting_run':False,
        'frozen_calibration_changed':False,'note':'Diagnostic source-frequency evidence only; no new held-out tuning or filter selection'}
    report['angular_high_band_rms_max_rad']=max(r['joint_velocity'].get('high_band_displacement_rms_max',0) for r in results)
    report['clip_p95_angular_high_band_rms_rad']=float(np.percentile([r['joint_velocity'].get('high_band_displacement_rms_max',0) for r in results],95))
    write_json(ROOT/'audit/bones_native_spectrum_v2.json',report);print({k:report[k] for k in ['measured','joint_spectral_review_clips','root_spectral_review_clips','angular_high_band_rms_max_rad','clip_p95_angular_high_band_rms_rad']})


if __name__=='__main__':run()
