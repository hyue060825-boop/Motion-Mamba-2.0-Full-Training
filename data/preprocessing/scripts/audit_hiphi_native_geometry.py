"""Independent input-geometry verification on fixed HiPHI smoke metadata."""
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha,atomic_bytes
from g1_pipeline.hiphi_native import read_bvh,world_fk,frames_at_25hz


def run():
    sys.path.insert(0,str(ROOT/'deps/GMR'))
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh as independent_read
    from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk
    rows=read_jsonl(ROOT/'manifests/hiphi_smoke100.jsonl');chosen=[];actors=set()
    for row in rows:
        if row['actor_id_namespaced'] not in actors:
            chosen.append(row);actors.add(row['actor_id_namespaced'])
        if len(chosen)==3:break
    results=[]
    for row in chosen:
        path=ROOT/'work/hiphi_packages'/(row['source_motion_id']+'.bvh')
        source=read_bvh(path);p,r=world_fk(source['local_positions_m'],source['local_rotations'],[n['parent'] for n in source['nodes']])
        # Upstream read_bvh splits motion rows on literal single spaces. Make a
        # whitespace-only input copy; numeric tokens and hierarchy are preserved.
        normalized=ROOT/'work/hiphi_geometry_v2'/(row['source_motion_id']+'.normalized.bvh')
        atomic_bytes(normalized,('\n'.join(' '.join(line.split()) for line in path.read_text().splitlines() if line.strip())+'\n').encode())
        other=independent_read(str(normalized));quat,positions=quat_fk(other.quats,other.pos,other.parents)
        ids={node['name']:i for i,node in enumerate(source['nodes'])};indices=[ids[name] for name in other.bones]
        re=R.from_quat(quat.reshape(-1,4),scalar_first=True).as_matrix().reshape(len(quat),len(other.bones),3,3)
        position_error=np.linalg.norm(p[:,indices]-positions*.01,axis=-1)
        rotation_error=np.rad2deg((R.from_matrix(r[:,indices].reshape(-1,3,3)).inv()*R.from_matrix(re.reshape(-1,3,3))).magnitude())
        frames,info=frames_at_25hz(path)
        offsets=np.array([node['offset'] for node in source['nodes']])*.01
        rest,_=world_fk(offsets[None],np.tile(np.eye(3),(1,len(offsets),1,1)),[n['parent'] for n in source['nodes']])
        local=source['local_positions_m'][:,indices]
        variations=np.ptp(np.linalg.norm(local[:,1:],axis=-1),axis=0)
        result={'sample_id':row['sample_id'],'source_sha256':sha(path),'normalized_whitespace_sha256':sha(normalized),'actor_id':row['actor_id_namespaced'],
                'frames':source['frames'],'articulated_joints':len(other.bones),'end_sites':len(source['nodes'])-len(other.bones),
                'native_fk_max_position_m':float(position_error.max()),'native_fk_max_orientation_deg':float(rotation_error.max()),
                'metadata_height_m':row.get('height_m'),'rest_offset_full_span_y_m':float(np.ptp(rest[0,:,1])),
                'nonroot_local_segment_length_change_max_m':float(variations.max()),
                'target_frames':len(frames),'source_fps':info['source_fps'],'stage':'NATIVE_GEOMETRY_NOT_ROBOT_QA',
                'status':'NATIVE_FK_PASS' if position_error.max()<1e-8 and rotation_error.max()<1e-6 else 'FAIL'}
        results.append(result);print(result,flush=True)
    vendor=ROOT/'deps/GMR/general_motion_retargeting/utils/lafan_vendor'
    write_json(ROOT/'audit/hiphi_native_geometry_v2.json',{'status':'NATIVE_INPUT_GEOMETRY_ONLY','results':results,
         'source_specific_chest':'Spine4, the shoulder-chain parent; old Spine2 robot mapping remains unvalidated',
         'human_only':True,'robot_retarget_run':False,
         'hashes':{str(p):sha(p) for p in [ROOT/'g1_pipeline/hiphi_native.py',vendor/'extract.py',vendor/'utils.py',ROOT/'scripts/audit_hiphi_native_geometry.py']}})


if __name__=='__main__':run()
