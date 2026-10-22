"""Show existing wrist-target residuals at explicit frames; no solver or QA changes."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from g1_pipeline.common import ROOT,sha,write_json,now
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline.robot import compile_target


def run():
    output=BASE/'wrist_visual_evidence';output.mkdir(exist_ok=True)
    ids=['humanml3d:M011855','hiphi:Self_motion-vault_0070__mirror','hiphi:Cause_motion-hit_0039__mirror']
    selected={x['record']['sample_id']:x['record'] for x in load(BASE/'final_review/selection.json')['records']}
    model,_=compile_target();model.vis.global_.offwidth=1400;model.vis.global_.offheight=1000;data=mujoco.MjData(model)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,20);small=ImageFont.truetype(fontpath,17);title=ImageFont.truetype(fontpath,24)
    records=[]
    with mujoco.Renderer(model,height=540,width=640) as renderer:
        for sid in ids:
            row=selected[sid];path=ROOT/row['qpos_path'];poses=np.load(path);metrics=load(ROOT/row['metrics_path'])
            wrist=max(['left_wrist_yaw_link','right_wrist_yaw_link'],key=lambda b:metrics['human_target_metrics'][b]['position_p95_m'])
            j=metrics['human_target_body_order'].index(wrist);bid=model.body(wrist).id
            with np.load(path.parent/'human_targets.npz') as archive:
                targets=archive['scaled_target_positions_m'][:,j].copy();stored=archive['position_error_m'][:,j].copy()
            points=[]
            for pose in poses:
                data.qpos[:]=pose;mujoco.mj_forward(model,data);points.append(data.xpos[bid].copy())
            points=np.asarray(points);errors=np.linalg.norm(points-targets,axis=1);p95=np.percentile(errors,95)
            frame=int(np.argmin(abs(errors-p95)));difference=float(abs(errors-stored).max());assert difference<2e-6
            data.qpos[:]=poses[frame];mujoco.mj_forward(model,data)
            actual=points[frame];target=targets[frame];center=(actual+target)/2
            # Two cameras give depth context; exact numeric distance is in3D.
            delta=target-actual;azimuth=float(np.rad2deg(np.arctan2(delta[1],delta[0]))+90)
            pictures=[]
            for angle in [azimuth,azimuth+90]:
                camera=mujoco.MjvCamera();camera.lookat[:]=center;camera.distance=1.45;camera.elevation=-8;camera.azimuth=angle
                renderer.update_scene(data,camera=camera);scene=renderer.scene
                for point,color,radius in [(actual,[1,.10,.15,1],.024),(target,[.1,1,.15,1],.024)]:
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([radius]*3),point,np.eye(3).ravel(),np.array(color));scene.ngeom+=1
                geom=scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1,.85,.1,1]))
                mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.004,actual,target);scene.ngeom+=1
                picture=Image.fromarray(renderer.render().copy());overlay=ImageDraw.Draw(picture)
                gl=mujoco.mjv_averageCamera(scene.camera[0],scene.camera[1])
                forward=np.asarray(gl.forward);up=np.asarray(gl.up);right=np.cross(forward,up)
                def project(point):
                    v=point-np.asarray(gl.pos);depth=float(v@forward)
                    halfheight=gl.frustum_top/gl.frustum_near*depth
                    return (320+float(v@right)/halfheight*270,270-float(v@up)/halfheight*270)
                rp,tp=project(actual),project(target)
                # Joint origins lie inside the wrist mesh. Project markers on
                # top of the image so the numerical endpoints remain visible.
                overlay.line([rp,tp],fill=(255,220,35),width=3)
                for pt,color,label in [(rp,(255,70,85),'R'),(tp,(55,255,80),'T')]:
                    x,y=pt;overlay.ellipse((x-8,y-8,x+8,y+8),fill=color,outline='white',width=2)
                    overlay.text((x+12,y-12),label,font=font,fill=color,stroke_width=2,stroke_fill=(0,0,0))
                pictures.append(picture)
            canvas=Image.new('RGB',(1280,830),(18,25,35));draw=ImageDraw.Draw(canvas)
            draw.text((20,12),sid,font=title,fill='white')
            draw.text((20,47),f"{wrist.replace('_yaw_link','')} | frame {frame}, t={frame/25:.2f}s | 3D distance {errors[frame]*100:.1f} cm",font=font,fill='white')
            draw.text((20,79),'RED: robot wrist joint center',font=font,fill=(255,100,110))
            draw.text((465,79),'GREEN: scaled + offset human target',font=font,fill=(90,255,100))
            for k,picture in enumerate(pictures):canvas.paste(picture,(k*640,115))
            draw.text((20,122),'View A',font=font,fill='white');draw.text((660,122),'View B (90 deg around)',font=font,fill='white')
            # Full-clip error trace; explicitly show the selected high-residual frame.
            x0,y0,x1,y1=65,697,1215,777;maximum=max(errors)*100*1.08
            draw.line((x0,y0,x0,y1,x1,y1),fill=(140,150,170),width=2)
            xy=[(x0+i/max(1,len(errors)-1)*(x1-x0),y1-e*100/maximum*(y1-y0)) for i,e in enumerate(errors)]
            draw.line(xy,fill=(255,210,55),width=2);xx,yy=xy[frame];draw.line((xx,y0,xx,y1),fill=(255,90,100),width=2);draw.ellipse((xx-4,yy-4,xx+4,yy+4),fill=(255,90,100))
            draw.text((20,664),f"Entire clip wrist error (cm): median {np.median(errors)*100:.1f}; p95 {p95*100:.1f}; max {max(errors)*100:.1f}",font=small,fill='white')
            draw.text((8,y0),f'{maximum:.0f}',font=small,fill='white');draw.text((35,y1-12),'0',font=small,fill='white')
            draw.text((x0,y1+4),'0 s',font=small,fill='white');draw.text((x1-100,y1+4),f'{(len(errors)-1)/25:.2f} s',font=small,fill='white')
            draw.text((20,808),'Near-p95 frame, not typical. R/T markers project through mesh. Target is a mapped landmark.',font=small,fill=(180,195,215))
            imagepath=output/(sid.split(':')[1]+'.png');canvas.save(imagepath)
            records.append({'sample_id':sid,'wrist':wrist,'frame':frame,'time_sec':frame/25,'distance_cm':float(errors[frame]*100),
                            'median_cm':float(np.median(errors)*100),'p95_cm':float(p95*100),'max_cm':float(max(errors)*100),
                            'selection':'nearest to full-clip p95 on side with larger p95; high residual, not typical frame',
                            'robot_wrist_xyz_m':actual.tolist(),'mapped_target_xyz_m':target.tolist(),
                            'recomputed_vs_saved_error_max_m':difference,'image_path':str(imagepath.relative_to(ROOT)),
                            'image_sha256':sha(imagepath),'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [path,path.parent/'human_targets.npz',ROOT/row['metrics_path']]}})
    write_json(output/'evidence.json',{'time':now(),'records':records,'solver_rerun':False,'trajectories_modified':False,
              'interpretation':'Measured mapped-target body-origin residual. Does not alone establish semantic failure, precise hand-surface error, or raw-human anatomical mismatch.',
              'previous_visual_layout_correction':'Previous left panel was robot before postprocessing, not the original human; right panel final robot plus target markers.'})
    print(records)


if __name__=='__main__':run()
