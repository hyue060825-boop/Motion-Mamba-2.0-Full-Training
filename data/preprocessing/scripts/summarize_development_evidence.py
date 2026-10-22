"""Summarize measured residuals, source scene metadata, and HiPHI input scale."""
import importlib.util
import json
from collections import Counter
import numpy as np
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha,digest
from g1_pipeline.scene_policy import classify
from scripts.audit_hiphi_rest_proportions import nodes


def run():
    namespace='bones_temporal_arms_dev_v15';root=ROOT/'work'/namespace
    runpath=ROOT/'qa/aggregate'/f'{namespace}.json';run=json.loads(runpath.read_text())
    records=[];groups={}
    for row in run['results']:
        mid=row['sample_id'].split(':',1)[1];p=root/(mid+'.solver.json')
        if sha(p)!=row['solver_sha256']:raise ValueError('SEALED_SOLVER_REPORT_CHANGED')
        solver=json.loads(p.read_text());residuals=solver['body_residuals']
        records.append({'sample_id':row['sample_id'],'frames':row['frames'],'body_residuals':residuals,'solver_sha256':sha(p)})
        for body,metrics in residuals.items():
            for key,value in metrics.items():groups.setdefault((body,key),[]).append((value,row['frames']))
    aggregate={}
    for (body,key),values in groups.items():
        a,w=np.array(values).T
        aggregate.setdefault(body,{})[key]={'clip_macro_mean':float(a.mean()),'clip_median':float(np.median(a)),
            'across_clip_p95':float(np.percentile(a,95)),'clip_max':float(a.max()),
            'frame_weighted_mean_of_clip_metric':float(np.average(a,weights=w))}
    write_json(ROOT/'audit/bones_temporal_v15_human_target_residuals.json',{
        'status':'MEASURED_NOT_RELEASE','expected':200,'measured':len(records),'run_sha256':sha(runpath),
        'definition':'Euclidean target body position error and SO3 geodesic orientation against final scaled/offset human GMR task2 targets; no reference robot input',
        'aggregate':aggregate,'records':records,'unavailable':'per-frame max residuals were not stored by v15; no claim of maximum-error bounds'})
    # Recover prior rule from its archived source, never from candidate quality.
    archives=list((ROOT/'audit/code_snapshots').glob('*/scene_policy.py'))
    old=None
    for path in archives:
        if "category=str(row.get('category'" not in path.read_text():old=path;break
    if old is None:raise ValueError('PRIOR_SCENE_RULE_SNAPSHOT_MISSING')
    spec=importlib.util.spec_from_file_location('scene_v1',old);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl');changes=[]
    for row in rows:
        before=module.classify(row);after=classify(row)
        if before['scene_class']!=after['scene_class']:
            changes.append({'sample_id':row['sample_id'],'category':row.get('category'),'body_position':row.get('body_position'),
                            'before':before,'after':after})
    write_json(ROOT/'audit/bones_scene_policy_v2_review.json',{'status':'DEVELOPMENT_METADATA_RULE_REVIEW',
        'old_code_sha256':sha(old),'new_code_sha256':sha(ROOT/'g1_pipeline/scene_policy.py'),
        'before_counts':dict(Counter(module.classify(r)['scene_class'] for r in rows)),
        'after_counts':dict(Counter(classify(r)['scene_class'] for r in rows)),
        'changes':changes,'reason':'Read source action category as well as posture/props/take; distinguish intentional falls and cartwheels from ordinary standing. No motion quality is an input.'})
    prior=json.loads((ROOT/'audit/hiphi_rest_proportions_v1.json').read_text())
    original={r['sample_id']:r for r in prior['rows']};results=[]
    for row in read_jsonl(ROOT/'manifests/hiphi_smoke100.jsonl'):
        path=ROOT/'work/hiphi_packages'/(row['source_motion_id']+'.bvh');ns=nodes(path)
        with path.open() as f:
            declarations=[]
            for line in f:
                if line.strip()=='MOTION':break
                parts=line.split()
                if parts and parts[0]=='CHANNELS':declarations.append(parts[1:])
            expected=['6','Xposition','Yposition','Zposition','Zrotation','Xrotation','Yrotation']
            if len(declarations)!=55 or any(ch!=expected for ch in declarations):
                raise ValueError('UNSUPPORTED_HIPHI_FIRST_FRAME_CHANNELS')
            next(f);next(f);values=np.fromstring(next(f),sep=' ')
        if values.size!=55*6 or not np.isfinite(values).all():raise ValueError('UNSUPPORTED_HIPHI_FIRST_FRAME_SCHEMA')
        v=values.reshape(55,6)[:,:3]*.01;points=[];j=0;differences=[]
        for node in ns:
            offset=np.asarray(node['offset'])
            if not node['end_site']:
                offset=v[j];j+=1
                if node['parent']>=0:differences.append(float(np.linalg.norm(offset-node['offset'])))
            if node['parent']==-1:offset=np.zeros(3)
            points.append(offset+(points[node['parent']] if node['parent']>=0 else 0))
        span=float(np.ptp(np.array(points)[:,1]));source_hash=sha(path)
        if source_hash!=original[row['sample_id']]['source_sha256']:raise ValueError('HIPHI_SOURCE_CHANGED')
        results.append({'sample_id':row['sample_id'],'actor_id':row['actor_id_namespaced'],
                        'metadata_height_m':row['height_m'],'zero_rotation_firstframe_translation_span_m':span,
                        'rest_offset_span_m':original[row['sample_id']]['rest_offset_height_proxy_m'],
                        'largest_nonroot_translation_vs_offset_m':max(differences),
                        'source_sha256':source_hash,'proxy_over_metadata_ratio':span/row['height_m']})
    write_json(ROOT/'audit/hiphi_firstframe_translation_proportions_v2.json',{
        'status':'INPUT_SCALE_REVIEW_NOT_RETARGET_ACCEPTANCE','expected':100,'measured':len(results),'results':results,
        'conclusion':'Shared template rig discrepancy persists in the actual local translation channels; it is not only a stale hierarchy OFFSET header.',
        'limitation':'Only first-frame translations measured here; zero-rotation span is a skeleton proxy, not mesh/actor true height. No robot scale strategy is accepted yet.',
        'script_sha256':sha(__file__)})
    print('residual clips',len(records),'scene changes',len(changes),'HiPHI first frames',len(results),flush=True)


if __name__=='__main__':run()
