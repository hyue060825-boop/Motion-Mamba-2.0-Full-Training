"""Explicit HiPHI native BVH geometry; no robot retargeting occurs here.

Six position channels encode local translations in the released dialect,
replacing corresponding hierarchy OFFSET components. The vendored independent
BVH reader and real per-file proportions are used to verify this interpretation.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from .resample import time_grid,linear,rotations


WORLD_BASIS=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
LANDMARKS=('Hips','Spine4','LeftUpLeg','LeftLeg','LeftFoot','RightUpLeg','RightLeg','RightFoot',
           'LeftArm','LeftForeArm','LeftHand','RightArm','RightForeArm','RightHand')


def read_bvh(path):
    lines=Path(path).read_text().splitlines();nodes=[];stack=[];channels=[];motion=None
    for i,line in enumerate(lines):
        parts=line.strip().split()
        if not parts:continue
        if parts[0]=='MOTION':motion=i;break
        if parts[0] in ('ROOT','JOINT'):
            name=parts[1]
            if any(n['name']==name for n in nodes):raise ValueError('DUPLICATE_BVH_NODE')
            parent=stack[-1] if stack else -1
            nodes.append({'name':name,'parent':parent,'offset':None,'channels':[],'end_site':False});stack.append(len(nodes)-1)
        elif parts[:2]==['End','Site']:
            if not stack:raise ValueError('END_SITE_WITHOUT_PARENT')
            nodes.append({'name':nodes[stack[-1]]['name']+'#EndSite','parent':stack[-1],'offset':None,'channels':[],'end_site':True});stack.append(len(nodes)-1)
        elif parts[0]=='OFFSET':
            if not stack or len(parts)!=4:raise ValueError('INVALID_BVH_OFFSET')
            nodes[stack[-1]]['offset']=[float(x) for x in parts[1:]]
        elif parts[0]=='CHANNELS':
            if not stack or int(parts[1])!=len(parts)-2:raise ValueError('INVALID_BVH_CHANNEL_DECLARATION')
            if nodes[stack[-1]]['end_site']:raise ValueError('CHANNELS_ON_END_SITE')
            declared=parts[2:]
            if len(set(declared))!=len(declared) or any(x not in ('Xposition','Yposition','Zposition','Xrotation','Yrotation','Zrotation') for x in declared):raise ValueError('INVALID_BVH_CHANNEL')
            nodes[stack[-1]]['channels']=declared;channels.extend((stack[-1],c) for c in declared)
        elif parts[0]=='}':
            if not stack:raise ValueError('UNBALANCED_BVH_BRACES')
            stack.pop()
    if motion is None or stack or not nodes or any(n['offset'] is None for n in nodes):raise ValueError('INCOMPLETE_BVH_HIERARCHY')
    if sum(n['parent']==-1 for n in nodes)!=1:raise ValueError('MULTIPLE_BVH_ROOTS')
    n=int(lines[motion+1].split(':')[1]);dt=float(lines[motion+2].split(':')[1])
    values=np.array([[float(x) for x in line.split()] for line in lines[motion+3:] if line.strip()])
    if n<1 or not np.isfinite(dt) or dt<=0 or values.shape!=(n,len(channels)) or not np.isfinite(values).all():raise ValueError('INVALID_BVH_MOTION_DATA')
    positions=np.tile(np.array([node['offset'] for node in nodes],dtype=float),(n,1,1))*.01
    local_rotation=np.tile(np.eye(3),(n,len(nodes),1,1))
    for index,node in enumerate(nodes):
        rotation_indices=[];order=''
        for ci,(owner,name) in enumerate(channels):
            if owner!=index:continue
            if name.endswith('position'):positions[:,index,'XYZ'.index(name[0])]=values[:,ci]*.01
            else:rotation_indices.append(ci);order+=name[0]
        if rotation_indices:
            if len(rotation_indices)!=3 or set(order)!=set('XYZ'):raise ValueError('EXPECTED_THREE_DISTINCT_ROTATION_AXES')
            local_rotation[:,index]=R.from_euler(order,values[:,rotation_indices],degrees=True).as_matrix()
    return {'nodes':nodes,'local_positions_m':positions,'local_rotations':local_rotation,'timestamps':np.arange(n)*dt,
            'fps':1/dt,'frames':n,'translation_semantics':'position_channels_replace_offset_components'}


def world_fk(local_position,local_rotation,parents):
    n,j,_=local_position.shape;p=np.empty_like(local_position);r=np.empty_like(local_rotation)
    for k,parent in enumerate(parents):
        if parent==-1:p[:,k]=local_position[:,k];r[:,k]=local_rotation[:,k]
        else:
            if not 0<=parent<k:raise ValueError('INVALID_PARENT_TOPOLOGY')
            p[:,k]=p[:,parent]+np.einsum('nij,nj->ni',r[:,parent],local_position[:,k]);r[:,k]=r[:,parent]@local_rotation[:,k]
    return p,r


def frames_at_25hz(path):
    source=read_bvh(path);nodes=source['nodes'];names={node['name']:i for i,node in enumerate(nodes)}
    articulated=[node for node in nodes if not node['end_site']]
    if len(articulated)!=55 or sum(n['end_site'] for n in nodes)!=13:raise ValueError('UNSUPPORTED_HIPHI_NODE_COUNTS')
    expected=['Xposition','Yposition','Zposition','Zrotation','Xrotation','Yrotation']
    if any(n['channels']!=expected for n in articulated) or not set(LANDMARKS)<=set(names):raise ValueError('UNSUPPORTED_HIPHI_SCHEMA')
    times,mapping=time_grid(source['timestamps']);local_p=linear(source['local_positions_m'],mapping)
    local_r=rotations(source['local_rotations'],mapping)
    p,r=world_fk(local_p,local_r,[n['parent'] for n in nodes]);frames=[]
    for i in range(len(times)):
        frames.append({name:[WORLD_BASIS@p[i,names[name]],R.from_matrix(WORLD_BASIS@r[i,names[name]]).as_quat(scalar_first=True)] for name in LANDMARKS})
    return frames,{'source_fps':source['fps'],'source_frames':source['frames'],'target_fps':25,'target_frames':len(times),
                   'time_mapping':mapping,'landmarks':list(LANDMARKS),'translation_semantics':source['translation_semantics'],
                   'world_basis':WORLD_BASIS.tolist(),'rotation_basis':'world-only; preserve source joint local axes',
                   'chest_mapping_reason':'Spine4 is the last spine node and parent of the shoulder chains; no reuse of SOMA Chest offsets',
                   'stage':'NATIVE_GEOMETRY_ONLY_NOT_ROBOT_ADAPTER_ACCEPTANCE'}
