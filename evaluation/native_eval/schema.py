"""Strict offline sample contract; no inference of frame roles from model H."""
from pathlib import Path
import numpy as np
from .common import read,sha

REQUIRED=['schema_version','sample_id','dataset','track','source_family_id','panel_id','checkpoint_sha256',
 'generator_code_snapshot','seed','representation','skeleton','skeleton_sha256','fps','units','array_path',
 'array_sha256','array_key','array_layout','coordinate_frame','up_axis','normalized','normalizer',
 'stored_frames','generated_frame_intervals','history_context_frames','history_in_array',
 'padding_frame_intervals','preroll_frame_intervals','frame_roles','caption_intervals','conditioning',
 'postprocess','completion_status','source_time_mapping','generation_time_mapping','generation_receipt']

def check_intervals(intervals,frames):
    out=np.zeros(frames,dtype=bool)
    for start,end in intervals:
        if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=frames or out[start:end].any():raise ValueError('Invalid/overlapping frame intervals')
        out[start:end]=True
    return out

def load_sample(meta,base,skeleton_hash):
    missing=[k for k in REQUIRED if k not in meta]
    if missing:raise ValueError('Missing fields: '+','.join(missing))
    expected=dict(schema_version='motion_eval_sample_v1',representation='physical414',skeleton='G1Skeleton34',skeleton_sha256=skeleton_hash,fps=25,units='meters',coordinate_frame='world',up_axis='y',completion_status='complete')
    for k,v in expected.items():
        if meta[k]!=v:raise ValueError(f'Invalid {k}: {meta[k]!r}')
    if meta['normalized'] is not False or meta['normalizer'] is not None:raise ValueError('Convert normalized legacy input with its exact audited original normalizer before native import')
    for key in ['sample_id','source_family_id','panel_id','dataset','track','postprocess']:
        if not isinstance(meta[key],str) or not meta[key]:raise ValueError('Unknown identity '+key)
    for key in ['seed','stored_frames','history_context_frames']:
        if type(meta[key]) is not int or meta[key]<0:raise ValueError('Invalid integer '+key)
    if type(meta['history_in_array']) is not bool:raise ValueError('History presence must be explicit boolean')
    if not isinstance(meta['source_time_mapping'],dict) or not isinstance(meta['generation_time_mapping'],dict):raise ValueError('Explicit time mapping dictionaries required')
    for mapping in ['source_time_mapping','generation_time_mapping']:
        if not meta[mapping]:raise ValueError('Unknown time mapping: '+mapping)
    if meta['array_layout'] not in ['generated_only','explicit_frame_roles']:raise ValueError('Unknown array_layout')
    for k in ['checkpoint_sha256','generator_code_snapshot','array_sha256']:
        if not isinstance(meta[k],str) or len(meta[k])!=64 or any(c not in '0123456789abcdef' for c in meta[k]):raise ValueError('Missing actual content identity '+k)
    path=Path(meta['array_path']);path=path if path.is_absolute() else Path(base)/path
    if sha(path)!=meta['array_sha256']:raise ValueError('Array hash mismatch')
    loaded=np.load(path,allow_pickle=False)
    if isinstance(loaded,np.lib.npyio.NpzFile):
        with loaded as pack:
            if meta['array_key']=='reference_physical':raise ValueError('reference_physical is GT, not generated output')
            a=np.array(pack[meta['array_key']],copy=True)
            if 'fps' in pack and float(pack['fps'])!=25:raise ValueError('Embedded fps mismatch')
    else:
        if meta['array_key'] is not None:raise ValueError('NPY key must be null')
        a=loaded
    if a.ndim!=2 or a.shape!=(meta['stored_frames'],414) or not np.isfinite(a).all():raise ValueError('Invalid motion shape/nonfinite input')
    n=len(a);gen=check_intervals(meta['generated_frame_intervals'],n)
    excluded=check_intervals(meta['padding_frame_intervals']+meta['preroll_frame_intervals'],n)
    if (gen&excluded).any():raise ValueError('Generated region includes padding/preroll')
    roles=np.full(n,'',dtype=object)
    for item in meta['frame_roles']:
        start,end=item['interval'];role=item['role']
        if role not in ['generated','gt_history','teacher_preroll','student_preroll','padding'] or not 0<=start<end<=n or any(roles[start:end]):raise ValueError('Invalid role partition')
        roles[start:end]=role
    if any(roles=='') or not np.array_equal(roles=='generated',gen):raise ValueError('Uncovered or contradictory frame roles')
    if not np.array_equal(excluded,np.isin(roles,['padding','teacher_preroll','student_preroll'])):raise ValueError('Preroll/padding interval lists contradict frame roles')
    if bool((roles=='gt_history').any())!=meta['history_in_array']:raise ValueError('History flag contradicts roles')
    if meta['array_layout']=='generated_only' and (not gen.all() or meta['history_in_array']):raise ValueError('generated_only contains excluded frames')
    if not gen.any():raise ValueError('No generated frames')
    # Do not concatenate disconnected generated intervals into a claimed continuous trajectory.
    if len(meta['generated_frame_intervals'])!=1:raise ValueError('Split disconnected generation into separate sample entries')
    for c in meta['caption_intervals']:
        s,e=c['start'],c['end']
        if not 0<=s<e<=n or not gen[s:e].all() or not c['text'].strip():raise ValueError('Invalid caption interval')
        if c.get('alignment') not in ['full_semantic_segment','weak_window','source_temporal_event']:raise ValueError('Unknown caption alignment')
    receipt=meta['generation_receipt']
    for k in ['H','F','sampling_steps','cfg','history_type','continuous_generation','resets']:
        if k not in receipt:raise ValueError('Missing generation receipt '+k)
    if receipt['H']!=meta['history_context_frames']:raise ValueError('Conflicting history length')
    if type(receipt['continuous_generation']) is not bool or not isinstance(receipt['resets'],list):raise ValueError('Unknown continuity receipt')
    if type(receipt['F']) is not int or receipt['F']<0 or type(receipt['sampling_steps']) is not int or receipt['sampling_steps']<=0 or not isinstance(receipt['cfg'],(int,float)) or not np.isfinite(receipt['cfg']):raise ValueError('Invalid sampler/control receipt')
    for k in ['mode','future_constraints','priming_receipt']:
        if k not in meta['conditioning']:raise ValueError('Missing conditioning '+k)
    if len(a[gen])>=225 and (not receipt['continuous_generation'] or receipt['resets']):raise ValueError('Long sequence was reset or not continuously generated')
    if sha(path)!=meta['array_sha256']:raise ValueError('Sample changed while reading; not a completed stable artifact')
    return a,meta['generated_frame_intervals'][0]

def canonicalize(a):
    a=np.asarray(a,dtype='float32').copy()
    a[:,[0,2]]-=a[0,[0,2]]
    return a

def diagnostics(a,skeleton=None):
    rot=a[:,104:308].reshape(-1,34,6);u,v=rot[...,:3],rot[...,3:]
    root_speed=np.linalg.norm(np.diff(a[:,:3],axis=0)*25,axis=-1)
    # Local position channels retain world vertical height and heading, only pelvis X/Z is removed.
    pos=np.concatenate((a[:,:3,None].transpose(0,2,1),a[:,5:104].reshape(-1,33,3).copy()),axis=1)
    pos[:,1:,0]+=a[:,None,0];pos[:,1:,2]+=a[:,None,2]
    dv=np.diff(pos,axis=0)*25;vel=a[:-1,308:410].reshape(-1,34,3)
    result=dict(rotation_unit_norm_error=float(np.mean(abs(np.linalg.norm(u,axis=-1)-1)+abs(np.linalg.norm(v,axis=-1)-1))),
                rotation_orthogonality_error=float(np.mean(abs((u*v).sum(-1)))),
                position_velocity_forward_difference_rmse=float(np.sqrt(np.mean((dv-vel)**2))),
                root_speed_mean_mps=float(root_speed.mean()) if len(root_speed) else None,
                contact_out_of_range_fraction=float(np.mean((a[:,410:]<0)|(a[:,410:]>1))),
                rotation_position_fk_consistency='PENDING_INDEPENDENT_FK_VALIDATION',
                qpos_projection_residual='NOT_PROVIDED',contact_truth='predicted contacts are not ground truth')
    if skeleton is not None:
        rest=np.asarray(skeleton['data']);parents=skeleton['parents']
        xx=u/np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-12)
        zz=np.cross(xx,v);zz=zz/np.maximum(np.linalg.norm(zz,axis=-1,keepdims=True),1e-12)
        yy=np.cross(zz,xx);rotmat=np.stack([xx,yy,zz],axis=-1)
        fk=np.zeros_like(pos);fk[:,0]=a[:,:3]
        for j in range(1,34):
            par=parents[j];fk[:,j]=fk[:,par]+np.einsum('tij,j->ti',rotmat[:,par],rest[j]-rest[par])
        result['rotation_position_fk_consistency']=dict(mean_joint_error_m=float(np.linalg.norm(fk-pos,axis=-1).mean()),
            max_joint_error_m=float(np.linalg.norm(fk-pos,axis=-1).max()),
            degenerate_rotation_fraction=float(np.mean((np.linalg.norm(u,axis=-1)<1e-8)|(np.linalg.norm(np.cross(u,v),axis=-1)<1e-8))))
    return result
