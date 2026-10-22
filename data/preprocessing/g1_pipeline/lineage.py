"""Metadata-only inventory, original-time lineage, deterministic grouped panels."""
from __future__ import annotations
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .common import *

def csv_rows(path):
    with open(path,newline='') as f:
        return list(csv.DictReader(f))

def actor_split(actor, seed):
    bucket=int.from_bytes(hashlib.sha256(f'{seed}:{actor}'.encode()).digest()[:8],'little')%1000
    return 'train' if bucket<900 else ('val' if bucket<950 else 'test')

def npz_header(path):
    with np.load(path,allow_pickle=False) as z:
        shapes={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
        fps_values=[float(z[k]) for k in ('mocap_framerate','mocap_frame_rate') if k in z]
        if not fps_values or len(set(fps_values))!=1 or not np.isfinite(fps_values[0]) or fps_values[0]<=0:
            raise ValueError('FPS_MISSING_OR_CONFLICT')
        gender=str(z['gender'].item()) if 'gender' in z else None
        if isinstance(z['gender'].item(),bytes): gender=z['gender'].item().decode()
    layout=shapes.get('poses',{}).get('shape',[])
    if len(layout)!=2 or layout[1] not in (72,156,165):
        raise ValueError('UNSUPPORTED_AMASS_LAYOUT')
    return {'keys':shapes,'fps':fps_values[0],'frames':layout[0],'pose_width':layout[1],
            'model_layout_candidate':{72:'smpl',156:'smplh',165:'smplx'}[layout[1]],'gender':gender,
            'model_type_evidence':'AMASS release path and layout; matching model FK still required'}

def hml_native_interval(source, start_frame, end_frame, native_fps, native_frames):
    stride=int(native_fps/20)
    if stride<1 or start_frame<0:
        raise ValueError('UNSUPPORTED_HML_NATIVE_INDEX')
    offsets={'Eyes_Japan_Dataset':60,'MPI_HDM05':60,'TotalCapture':20,'MPI_Limits':20,'Transitions_mocap':10}
    offset=next((v for k,v in offsets.items() if k in source),0)
    available=(native_frames+stride-1)//stride-offset
    # Match Python [start:end] behavior, including negative end and clipping.
    lo,hi,step=slice(start_frame,end_frame).indices(max(0,available))
    if hi<=lo:
        raise ValueError('EMPTY_HML_NATIVE_INTERVAL')
    first=(offset+lo)*stride;requested_exclusive=(offset+hi)*stride
    # The processed index can end on the final partial native stride. Keep the
    # real native support and record the clipped boundary; never let consumers
    # extrapolate to requested_exclusive.
    exclusive=min(native_frames,requested_exclusive)
    if exclusive<=first: raise ValueError('EMPTY_HML_NATIVE_INTERVAL_AFTER_NATIVE_CLIP')
    return {'start_sec':first/native_fps,'end_sec':exclusive/native_fps,
            'last_sample_sec':min((native_frames-1)/native_fps,exclusive/native_fps),
            'processed_stride':stride,'pre_crop_processed_frames':offset,
            'native_first_index':first,'native_end_index_exclusive':exclusive,
            'requested_native_end_index_exclusive':requested_exclusive,
            'native_end_clipped':bool(exclusive!=requested_exclusive),
            'processed_interval_frames':hi-lo,'native_index_formula':'(pre_crop_offset + index_frame) * int(native_fps/20)',
            'nominal_processed_fps':20,'effective_processed_fps':native_fps/stride}

def caption_intervals(path, interval):
    out=[]
    if not Path(path).is_file(): return out
    for i,line in enumerate(Path(path).read_text().splitlines()):
        parts=line.split('#')
        if len(parts)!=4: raise ValueError('HML_CAPTION_FORMAT')
        a,b=float(parts[2]),float(parts[3])
        if not np.isfinite([a,b]).all():
            # Official preprocessing treats NaN time tags as whole-motion captions.
            a=b=0.
        full=a==0 and b==0
        if full:
            start,end=interval['start_sec'],interval['end_sec']
        else:
            start=interval['start_sec']+int(a*20)/interval['effective_processed_fps']
            end=interval['start_sec']+int(b*20)/interval['effective_processed_fps']
            start=max(start,interval['start_sec']);end=min(end,interval['end_sec'])
            if end<=start: continue
        out.append({'caption_id':Path(path).stem+f':{i}','text':parts[0],'native_start_sec':start,'native_end_sec':end,
                    'original_tags_sec':[a,b],'whole_motion':full})
    return out

def family_leakage(rows):
    groups=defaultdict(set)
    for r in rows:
        groups[r['source_family_id']].add(r.get('official_split') or r.get('project_split'))
    conflicts={k for k,v in groups.items() if 'train' in v and (v & {'val','test'})}
    for r in rows:
        if r['source_family_id'] in conflicts and (r.get('official_split') or r.get('project_split'))=='train':
            r['leakage_status']='EXCLUDED_TRAIN_FAMILY_IN_EVAL'
            r['eligibility']='rejected';r['reason']='TRAIN_FAMILY_IN_OFFICIAL_VAL_TEST'
        else:r['leakage_status']='FAMILY_SPLIT_CLEAR'
    return sorted(conflicts)

def grouped_panel(rows,count,seed,key='category'):
    candidates=[r for r in rows if r.get('project_split')=='train' and r['eligibility']=='candidate' and not r['mirror_flag']]
    candidates.sort(key=lambda r:digest([seed,r['source_family_id'],r['sample_id']]))
    bins=defaultdict(list);seen=set()
    for r in candidates:
        if r['source_family_id'] in seen: continue
        seen.add(r['source_family_id']);bins[r.get(key) or 'unknown'].append(r)
    out=[]
    while len(out)<count and any(bins.values()):
        for category in sorted(bins):
            if bins[category] and len(out)<count:out.append(bins[category].pop(0))
    if len(out)!=count:raise ValueError(f'INSUFFICIENT_INDEPENDENT_FAMILIES: required {count}, available {len(out)}')
    return out

def build_bones():
    rows=[];meta=BONES/'metadata/seed_metadata_v004.csv';mh=sha(meta)
    for r in csv_rows(meta):
        family='bones:'+':'.join([r['take_date'],r['take_actor'],r['take_name']])
        row={'sample_id':'bones:'+r['filename'],'dataset':'bones','dataset_release':'seed_metadata_v004',
             'source_motion_id':r['filename'],'source_family_id':family,'actor_id_namespaced':'bones:'+r['actor_uid'],
             'actor_known':True,'take_id':r['take_name'],'session_id':r['take_date'],'mirror_flag':r['is_mirror'].lower()=='true',
             'source_paths':{'g1':r['move_g1_path'],'soma_uniform':r['move_soma_uniform_path'],'soma_proportional':r['move_soma_proportional_path']},
             'provenance_evidence':{'metadata_row_key':r['filename'],'metadata_sha256':mh,'path_columns':['move_g1_path','move_soma_uniform_path']},
             'original_fps':120.,'original_frames':int(float(r['move_duration_frames'])),'height_m':float(r['actor_height_cm'])/100,
             'official_split':None,'official_split_reason':'project actor-disjoint protocol; not an official split',
             'project_split':actor_split(r['actor_uid'],20260824),'category':r['content_type_of_movement'],
             'body_position':r['content_body_position'],'props':r['content_props'],
             'captions':[r[k] for k in ['content_natural_desc_1','content_natural_desc_2','content_natural_desc_3','content_natural_desc_4'] if r[k]],
             'eligibility':'candidate','status':'DISCOVERED','source_hashes':None,'source_hashes_reason':'hash at materialization; raw CSV existence audited now',
             'scale_strategy':'official G1 main pool; SOMA uniform calibration must use uniform skeleton height',
             'time_alignment':'unverified until paired frame count and offset audit'}
        row['native_start_sec']=0.;row['native_end_sec']=row['original_frames']/120
        if not (BONES/r['move_g1_path']).is_file():row.update(eligibility='missing',reason='MISSING_OFFICIAL_G1_CSV')
        rows.append(row)
    # Actor separation inside project train; do not use project val/test for adapter fitting or acceptance.
    actors=sorted({r['actor_id_namespaced'] for r in rows if r['project_split']=='train'},key=lambda x:digest([20260915,x]))
    dev_actors=set(actors[:max(1,int(.55*len(actors)))])
    dev=[r for r in rows if r['actor_id_namespaced'] in dev_actors]
    accept=[r for r in rows if r['actor_id_namespaced'] not in dev_actors]
    first=grouped_panel(dev,216,20260915)
    panels={'smoke':first[:16],'dev':first[16:],'accept':grouped_panel(accept,300,20260916)}
    for name,panel in panels.items():
        for r in panel:r['adapter_split']=name
        write_jsonl(ROOT/f'manifests/bones_pairs_{name}.jsonl',panel)
    return rows

def build_hiphi():
    meta=HIPHI/'metadata/hiphi_metadata.csv';mh=sha(meta)
    actors={r['actor_id']:r for r in csv_rows(HIPHI/'metadata/hiphi_actor_metadata.csv')}
    archives={}
    for a in csv_rows(HIPHI/'data/motion_to_part.csv'):
        archives[a['motion_id']]=a['archive_name'];archives[a['mirrored_motion_id']]=a['archive_name']
    rows=[]
    for r in csv_rows(meta):
        mid=r['motion_id'];body=r['is_hoi']=='false' and not r['object_categories'].strip()
        actor=actors.get(r['actor_id'])
        row={'sample_id':'hiphi:'+mid,'dataset':'hiphi','dataset_release':'local_snapshot_20260907_metadata_'+mh[:12],
             'source_motion_id':mid,'source_family_id':'hiphi:'+mid.removesuffix('__mirror'),
             'actor_id_namespaced':'hiphi:'+r['actor_id'],'actor_known':bool(actor),
             'mirror_flag':r['mirrored']=='true','original_frames':int(r['frame_count']),
             'original_fps':None,'original_fps_reason':'must read BVH Frame Time; metadata duration is not an exact FPS authority',
             'metadata_duration_sec':float(r['duration_sec']),'height_m':float(actor['height_cm'])/100 if actor else None,
             'height_source':'hiphi_actor_metadata.csv' if actor else None,'category':r['frame'],'lu':r['lu'],
             'captions':[r['text_annotation']] if r['text_annotation'] else [],'is_hoi':r['is_hoi'],
             'archive':archives.get(mid),'package_prefix':f"HiPHI/data/{r['frame']}/{r['lu']}/{mid}",
             'official_split':None,'official_split_reason':'no applicable local official actor split found',
             'project_split':actor_split(r['actor_id'],20260915),'provenance_evidence':{'metadata_sha256':mh,'metadata_row_key':mid},
             'eligibility':'candidate' if body else 'unsupported','status':'DISCOVERED',
             'reason':None if body else 'HOI_EXCLUDED','package_body_only_verified':False,
             'package_body_only_reason':'metadata candidate only; archive package must agree before retargeting'}
        if body and (not row['archive'] or not (HIPHI/'data'/row['archive']).is_file()):
            row.update(eligibility='missing',reason='MISSING_ARCHIVE')
        rows.append(row)
    write_jsonl(ROOT/'manifests/hiphi_smoke100.jsonl',grouped_panel(rows,100,20260915))
    return rows

def build_hml():
    idx=HML/'index.csv';index_hash=sha(idx)
    splits={k:set((HML/'HumanML3D'/f'{k}.txt').read_text().splitlines()) for k in ['train','val','test']}
    cache={};rows=[]
    for r in csv_rows(idx):
        relative=Path(r['source_path'].removeprefix('./pose_data/')).with_suffix('.npz')
        path=HML/'amass_data'/relative
        family='humanml3d:'+str(relative)
        info=None;err=None
        if 'humanact12' in str(relative).lower():err='MISSING_NATIVE_SOURCE_HUMANACT12'
        elif not path.is_file():err='MISSING_NATIVE_SOURCE'
        else:
            if str(path) not in cache:
                try:cache[str(path)]=npz_header(path)
                except Exception as exc:cache[str(path)]={'error':str(exc)}
            info=cache[str(path)]
            if 'error' in info:err=info['error']
        interval=None
        if not err:
            try:interval=hml_native_interval(str(relative),int(r['start_frame']),int(r['end_frame']),info['fps'],info['frames'])
            except ValueError as exc:err=str(exc)
        for mirror in [False,True]:
            ident=('M' if mirror else '')+Path(r['new_name']).stem
            memberships=[k for k,v in splits.items() if ident in v]
            split=memberships[0] if len(memberships)==1 else None
            reason=err or ('UNKNOWN_OR_CONFLICTING_OFFICIAL_SPLIT' if split is None else None)
            actor='humanml3d:'+':'.join(relative.parts[:2])
            row={'sample_id':'humanml3d:'+ident,'dataset':'humanml3d','dataset_release':'index_'+index_hash[:12],
                'source_motion_id':ident,'source_family_id':family,'actor_id_namespaced':actor,'actor_known':info is not None,
                'mirror_flag':mirror,'source_paths':{'native':str(Path('amass_data')/relative)},'category':relative.parts[0],
                'official_split':split,'project_split':split,'eligibility':'candidate' if reason is None else ('missing' if 'MISSING' in reason else 'unsupported'),
                'reason':reason,'status':'DISCOVERED','provenance_evidence':{'index_hash':index_hash,'index_row':r,
                  'processing_code_hash':sha(HML/'raw_pose_processing.ipynb'),'lineage_method':'official integer-stride AMASS FK followed by dataset-specific offset and index crop'},
                'native_metadata':info,'native_interval':interval,'original_fps':info.get('fps') if info else None,
                'original_frames':info.get('frames') if info else None,'source_body_model_type':info.get('model_layout_candidate') if info else None,
                'native_start_sec':interval['start_sec'] if interval else None,'native_end_sec':interval['end_sec'] if interval else None,
                'caption_source':f'HumanML3D/texts/{ident}.txt'}
            try:captions=caption_intervals(HML/row['caption_source'],interval) if interval else []
            except ValueError as exc:captions=[];row.update(eligibility='unsupported',reason=str(exc))
            row['caption_time_spans']=captions;row['captions']=[c['text'] for c in captions]
            rows.append(row)
    conflicts=family_leakage(rows)
    write_json(ROOT/'splits/hml_train_family_exclusions.json',{'families':conflicts,'policy':'keep official labels; exclude train members sharing any native take with val/test, including mirror'})
    write_json(ROOT/'audit/amass_layout_inventory.json',cache)
    write_jsonl(ROOT/'manifests/hml_smoke100.jsonl',grouped_panel(rows,100,20260915))
    return rows

def summary(rows):
    def duration(r):
        if r.get('native_interval'):return max(0,r['native_interval']['end_sec']-r['native_interval']['start_sec'])
        if r.get('original_fps'):return r['original_frames']/r['original_fps']
        return r.get('metadata_duration_sec')
    known=[duration(r) for r in rows if duration(r) is not None]
    return {'candidates':len(rows),'source_families':len({r['source_family_id'] for r in rows}),
        'actors':len({r['actor_id_namespaced'] for r in rows if r['actor_known']}),
        'eligibility':dict(Counter(r['eligibility'] for r in rows)),
        'exclusion_reasons':dict(Counter(r.get('reason') for r in rows if r.get('reason'))),
        'split_counts':dict(Counter(r.get('project_split') or 'unknown' for r in rows)),
        'known_candidate_hours':sum(known)/3600,'duration_unknown_count':len(rows)-len(known),
        'accepted':0,'retarget_attempted':0,'effective_train_hours':0,
        'note':'metadata inventory includes mirrors; hours sum records, not unique capture hours. Accepted remains zero until processing and QA.'}

def build_all():
    groups={'bones':build_bones(),'humanml3d':build_hml(),'hiphi':build_hiphi()}
    allrows=[]
    for name,rows in groups.items():
        if name!='humanml3d':family_leakage(rows)
        write_jsonl(ROOT/f'manifests/{name}_lineage.jsonl',rows);allrows.extend(rows)
    write_jsonl(ROOT/'manifests/full_candidates.jsonl',allrows)
    duplicate_groups=defaultdict(list)
    for r in allrows:duplicate_groups[r['source_family_id']].append(r['sample_id'])
    write_jsonl(ROOT/'manifests/duplicate_groups.jsonl',[{'family':k,'samples':v,'type':'lineage_family_not_proof_of_identical_motion'} for k,v in duplicate_groups.items() if len(v)>1])
    write_jsonl(ROOT/'manifests/unsupported.jsonl',[r for r in allrows if r['eligibility']=='unsupported'])
    write_jsonl(ROOT/'manifests/missing.jsonl',[r for r in allrows if r['eligibility']=='missing'])
    write_jsonl(ROOT/'manifests/rejected.jsonl',[r for r in allrows if r['eligibility']=='rejected'])
    report={k:summary(v) for k,v in groups.items()}
    report['archives']={str(p):p.stat().st_size for p in list(BONES.glob('*.tar.gz'))+list((HIPHI/'data').glob('*.zst'))}
    report['metadata_hashes']={str(p):sha(p) for p in [BONES/'metadata/seed_metadata_v004.csv',HIPHI/'metadata/hiphi_metadata.csv',HIPHI/'metadata/hiphi_actor_metadata.csv',HML/'index.csv']}
    write_json(ROOT/'audit/data_inventory.json',report)
    (ROOT/'reports/leakage_audit.md').write_text('# Lineage 与泄漏审计\n\n'
        'BONES 使用既有 actor_split(seed=20260824) 项目 90/5/5 split，不称官方 split。校准 smoke/dev 与 accept 按 actor 隔离，仅取项目 train。\n\n'
        'HumanML3D 保留官方 train/val/test 标识；同 native take 出现在 val/test 时，所有 train 成员隔离，镜像同 family。'
        'native timestamp 使用 int(native_fps/20) 的整数步长及数据集前置 crop，不能统一 index/20。\n\n'
        'HiPHI 暂用项目 actor 90/5/5(seed=20260915)；Body-only metadata 选择须再经 package 验证。\n\n'
        '未知：跨库真实 actor 身份、跨库近重复动作指纹核实、预训练 tokenizer/controller/evaluator 全部数据暴露。'
        '当前不能签发无泄漏结论或融合 QA gate；duplicate_groups 为同源 family 分组，不冒充精确 hash 重复检测。\n')
    return report
