"""Headless side-by-side full-clip replay of the fixed smoke panel."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,subprocess
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,sha
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample


def run(namespace='bones_soma_rest_smoke_v3',panel='smoke',selection=None):
    if panel not in ('smoke','dev'): raise ValueError('RENDER_PANEL_NOT_DEVELOPMENT')
    model,asset=compile_target();left=mujoco.MjData(model);right=mujoco.MjData(model)
    report=json.loads((ROOT/'qa/aggregate'/f'{namespace}.json').read_text());results={r['sample_id']:r for r in report['results']}
    out=ROOT/'videos'/namespace;out.mkdir(parents=True,exist_ok=True);rows=[]
    panel_rows=read_jsonl(ROOT/f'manifests/bones_pairs_{panel}.jsonl')
    if selection is not None:
        requested=set(selection)
        panel_rows=[r for r in panel_rows if r['sample_id'] in requested]
        if len(panel_rows)!=len(requested): raise ValueError('UNKNOWN_RENDER_SAMPLE')
    with mujoco.Renderer(model,height=360,width=480) as renderer:
        for row in panel_rows:
            mid=row['source_motion_id'];base=ROOT/'work'/namespace
            candidate=np.load(base/(mid+'.npy'))
            t=np.asarray(json.loads((base/(mid+'.time.json')).read_text())['source']['target_timestamps'])
            ref,rt,_=read_csv(BONES/row['source_paths']['g1'],format_id=FORMAT,fps=row['original_fps']);ref,_,_=resample(ref,rt,model.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
            aligned=candidate.copy();a=results[row['sample_id']]['fixed_planar_alignment'];rotation=R.from_euler('z',a['yaw_rad'])
            aligned[:,:3]=rotation.apply(candidate[:,:3])+a['translation'];aligned[:,3:7]=(rotation*R.from_quat(candidate[:,3:7],scalar_first=True)).as_quat(scalar_first=True)
            path=out/(mid+'.mp4');tmp=out/(mid+'.tmp.mp4')
            command=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','25','-i','-',
                     '-vf',"drawtext=text='Official BONES':x=15:y=15:fontsize=20:fontcolor=white,drawtext=text='GMR candidate (fixed alignment)':x=495:y=15:fontsize=20:fontcolor=white",
                     '-an','-c:v','libx264','-preset','fast','-threads','2','-pix_fmt','yuv420p',str(tmp)]
            process=subprocess.Popen(command,stdin=subprocess.PIPE)
            camera=mujoco.MjvCamera();camera.distance=3.;camera.azimuth=135;camera.elevation=-18
            try:
                for a,b in zip(ref,aligned):
                    # Same camera for both views; reference-root follow is only a
                    # viewer operation. Full qpos trajectory is unmodified.
                    camera.lookat[:]=[a[0],a[1],max(.6,a[2])]
                    images=[]
                    for d,x in ((left,a),(right,b)):
                        d.qpos[:]=x;mujoco.mj_forward(model,d);renderer.update_scene(d,camera=camera);images.append(renderer.render().copy())
                    process.stdin.write(np.concatenate(images,axis=1).tobytes())
                process.stdin.close()
                if process.wait()!=0:raise RuntimeError('FFMPEG_FAILED')
                tmp.replace(path)
            finally:
                if process.poll() is None:process.terminate();process.wait()
            rows.append({'sample_id':row['sample_id'],'path':str(path),'sha256':sha(path),'frames':len(candidate),'fps':25,
                         'alignment':results[row['sample_id']]['fixed_planar_alignment'],'asset_hash':asset['asset_hash']})
            write_json(out/'manifest.json',{'source_panel':f'bones_pairs_{panel}','expected':len(panel_rows),'rendered':len(rows),'videos':rows})
            print(mid,len(candidate),'rendered',flush=True)
    return rows
if __name__=='__main__':run()
