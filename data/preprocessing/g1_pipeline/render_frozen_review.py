"""Full-clip evidence and contact sheets, with hashes of the replayed trajectories."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
import subprocess
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation as R
from .common import ROOT,BONES,read_jsonl,write_json,sha
from .robot import compile_target,read_csv,FORMAT
from .resample import qpos as resample


def run(namespace,ids,*,source='bones',panel='accept'):
    if not namespace.replace('_','').isalnum():raise ValueError('INVALID_NAMESPACE')
    base=ROOT/'work'/namespace;ledger=base/'results.jsonl';records={r['sample_id']:r for r in read_jsonl(ledger)}
    if source=='bones':metadata={r['sample_id']:r for r in read_jsonl(ROOT/f'manifests/bones_pairs_{panel}.jsonl')}
    out=ROOT/'videos'/namespace;out.mkdir(exist_ok=True);mp=out/'manifest.json'
    previous=json.loads(mp.read_text()).get('videos',[]) if mp.exists() else []
    videos={r['sample_id']:r for r in previous};m,asset=compile_target();left=mujoco.MjData(m);right=mujoco.MjData(m)
    with mujoco.Renderer(m,height=360,width=480) as renderer:
        for sid in ids:
            row=records[sid];mid=sid.split(':',1)[1];bindings={};targets=None
            if source=='bones':
                meta=metadata[sid];qp=base/(mid+'.npy');tp=base/(mid+'.time.json');reference=BONES/meta['source_paths']['g1']
                candidate=np.load(qp);times=np.array(json.loads(tp.read_text())['source']['target_timestamps'])
                ref,rt,_=read_csv(reference,format_id=FORMAT,fps=meta['original_fps'])
                ref,_,_=resample(ref,rt,m.jnt_range[1:],end=np.nextafter(times[-1],np.inf))
                transform=row['fixed_planar_alignment'];rot=R.from_euler('z',transform['yaw_rad'])
                candidate=candidate.copy();candidate[:,:3]=rot.apply(candidate[:,:3])+transform['translation']
                candidate[:,3:7]=(rot*R.from_quat(candidate[:,3:7],scalar_first=True)).as_quat(scalar_first=True)
                files=[qp,tp,reference];labels=('Official BONES','GMR fixed alignment')
            else:
                qp=ROOT/row['output_paths']['qpos_path'];raw=qp.parent/'unprocessed_qpos.npy';targetpath=qp.parent/'human_targets.npz'
                candidate=np.load(qp);ref=np.load(raw);files=[qp,raw,targetpath,ROOT/row['metrics_path']]
                with np.load(targetpath) as data:targets=data['scaled_target_positions_m'].copy()
                labels=('Before postprocessing','Final and scaled human targets')
            if ref.shape!=candidate.shape or candidate.shape[1]!=36:raise ValueError('REVIEW_TRAJECTORY_SHAPE')
            for path in files:bindings[str(path)]=sha(path)
            old=videos.get(sid,{})
            if old.get('input_hashes')==bindings and all((ROOT/old[k]).exists() and sha(ROOT/old[k])==old[h] for k,h in [('path','sha256'),('sheet_path','sheet_sha256')]):continue
            path=out/(mid+'.mp4');tmp=out/(mid+'.tmp.mp4');sheet=out/(mid+'.review.jpg')
            command=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','25','-i','-',
                '-vf',f"drawtext=text='{labels[0]}':x=12:y=12:fontsize=18:fontcolor=white,drawtext=text='{labels[1]}':x=492:y=12:fontsize=18:fontcolor=white",
                '-an','-c:v','libx264','-preset','fast','-threads','2','-pix_fmt','yuv420p',str(tmp)]
            proc=subprocess.Popen(command,stdin=subprocess.PIPE);camera=mujoco.MjvCamera();camera.distance=3.;camera.azimuth=135;camera.elevation=-18
            indices=set(np.linspace(0,len(candidate)-1,min(12,len(candidate)),dtype=int));stills=[]
            try:
                for i,(a,b) in enumerate(zip(ref,candidate)):
                    camera.lookat[:]=[a[0],a[1],max(.6,a[2])];images=[]
                    for side,(data,pose) in enumerate(((left,a),(right,b))):
                        data.qpos[:]=pose;mujoco.mj_forward(m,data);renderer.update_scene(data,camera=camera)
                        if targets is not None and side==1:
                            for target in targets[i]:
                                scene=renderer.scene
                                if scene.ngeom>=scene.maxgeom:raise ValueError('VISUAL_TARGET_CAPACITY')
                                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.018]*3),target,np.eye(3).ravel(),np.array([.1,1,.1,.8]))
                                scene.ngeom+=1
                        images.append(renderer.render().copy())
                    pair=np.concatenate(images,axis=1);proc.stdin.write(pair.tobytes())
                    if i in indices:
                        still=Image.fromarray(pair).resize((640,240));ImageDraw.Draw(still).text((5,5),f'{mid}  t={i/25:.2f}s | {labels[0]} / {labels[1]}',fill='white');stills.append(still)
                proc.stdin.close()
                if proc.wait()!=0:raise RuntimeError('FFMPEG_FAILED')
                tmp.replace(path)
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
            canvas=Image.new('RGB',(1280,240*((len(stills)+1)//2)))
            for j,still in enumerate(stills):canvas.paste(still,((j%2)*640,(j//2)*240))
            canvas.save(sheet,quality=90)
            videos[sid]={'sample_id':sid,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'sheet_path':str(sheet.relative_to(ROOT)),
                'sheet_sha256':sha(sheet),'frames':len(candidate),'fps':25,'input_hashes':bindings,'asset_hash':asset['asset_hash'],'full_clip':True}
            write_json(mp,{'source':source,'panel':panel,'rendered':len(videos),'videos':list(videos.values()),'renderer_sha256':sha(__file__)})
            print(sid,'rendered',len(candidate),flush=True)
    return list(videos.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--namespace',required=True);p.add_argument('--ids',nargs='+',required=True)
    p.add_argument('--source',choices=['bones','humanml3d','hiphi'],default='bones');p.add_argument('--panel',default='accept')
    a=p.parse_args();run(a.namespace,a.ids,source=a.source,panel=a.panel)
