"""Versioned metadata repair; preserve official samples, splits and old evidence."""
import copy
import json
import os
import tempfile
from collections import Counter
from .common import ROOT,HML,read_jsonl,write_json,write_jsonl,sha
from .lineage import hml_native_interval,caption_intervals,summary


def repair(row):
    result=copy.deepcopy(row)
    if row.get('native_interval') is None:
        return result
    index=row['provenance_evidence']['index_row'];meta=row['native_metadata']
    interval=hml_native_interval(index['source_path'],int(index['start_frame']),int(index['end_frame']),meta['fps'],meta['frames'])
    interval['last_sample_sec']=(interval['native_end_index_exclusive']-1)/meta['fps']
    result.update(native_interval=interval,native_start_sec=interval['start_sec'],native_end_sec=interval['end_sec'])
    result['caption_time_spans']=caption_intervals(HML/row['caption_source'],interval)
    result['captions']=[r['text'] for r in result['caption_time_spans']]
    return result


def run():
    old=ROOT/'manifests/humanml3d_lineage.jsonl';panel=ROOT/'manifests/hml_smoke100.jsonl'
    source=read_jsonl(old);fixed=[];changes=[]
    invariant=('sample_id','source_family_id','actor_id_namespaced','source_paths','mirror_flag','official_split','project_split','eligibility','leakage_status')
    for row in source:
        revised=repair(row)
        assert all(revised.get(k)==row.get(k) for k in invariant)
        fields=[k for k in row if revised[k]!=row[k]]
        if fields:changes.append({'sample_id':row['sample_id'],'changed_fields':fields,'before_interval':row['native_interval'],'after_interval':revised['native_interval']})
        fixed.append(revised)
    byid={r['sample_id']:r for r in fixed};oldpanel=read_jsonl(panel)
    newpanel=[repair(r) for r in oldpanel]
    assert len(newpanel)==100 and len({r['source_family_id'] for r in newpanel})==100
    assert all(r==byid[r['sample_id']] for r in newpanel)
    outputs=[ROOT/'manifests/humanml3d_lineage.v2.jsonl',ROOT/'manifests/hml_smoke100.v2.jsonl',ROOT/'manifests/full_candidates.v2.jsonl']
    write_jsonl(outputs[0],fixed);write_jsonl(outputs[1],newpanel)
    # Stream the three-source inventory; other source records remain byte-identical.
    full=ROOT/'manifests/full_candidates.jsonl';fd,tmp=tempfile.mkstemp(dir=outputs[2].parent,prefix='.lineage-v2-')
    counts=Counter()
    try:
        with os.fdopen(fd,'w') as dest,full.open() as src:
            for line in src:
                row=json.loads(line);counts[row['dataset']]+=1
                dest.write(json.dumps(byid[row['sample_id']],sort_keys=True,allow_nan=False)+'\n' if row['dataset']=='humanml3d' else line)
            dest.flush();os.fsync(dest.fileno())
        os.replace(tmp,outputs[2])
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    staticpath=ROOT/'audit/hml_native_rest_mapping_v1.json';static=json.loads(staticpath.read_text())
    assert static['manifest_sha256']==sha(panel) and static['module_sha256']==sha(ROOT/'g1_pipeline/native_rest_mapping.py')
    assert {r['sample_id'] for r in static['results']}=={r['sample_id'] for r in newpanel}
    for row in static['results']:
        meta=byid[row['sample_id']]
        for path,expected in [(HML/meta['source_paths']['native'],row['native_sha256']),
                              (ROOT/row['config_path'],row['config_sha256']),(ROOT/row['audit_path'],row['audit_sha256'])]:
            if sha(path)!=expected:raise ValueError('STATIC_MAPPING_INPUT_CHANGED:'+str(path))
    static.update(manifest_sha256=sha(outputs[1]),metadata_revision={'prior_audit_sha256':sha(staticpath),
        'reason':'native interval metadata only; identical native source hashes, mirror flags, shapes, mapping configs and audits',
        'robot_motion_retargeting_run':False})
    staticout=ROOT/'audit/hml_native_rest_mapping_v2.json';write_json(staticout,static)
    report={'status':'PASS','module_sha256':sha(__file__),'old_bindings':{str(p.relative_to(ROOT)):sha(p) for p in (old,panel,full,staticpath)},
        'new_bindings':{str(p.relative_to(ROOT)):sha(p) for p in [*outputs,staticout]},'source_counts':dict(counts),
        'unchanged_identity_fields':invariant,'smoke100_changed_count':sum(a!=b for a,b in zip(oldpanel,newpanel)),
        'changed_count':len(changes),'changed_records':changes,'summary':summary(fixed),
        'formula':'last_sample_sec=(native_end_index_exclusive-1)/actual_native_fps; integer stride is int(actual_native_fps/20)',
        'old_files_preserved':True,'robot_retargeting_run':False}
    write_json(ROOT/'audit/hml_lineage_revision_v2.json',report)
    return {k:v for k,v in report.items() if k!='changed_records'}


if __name__=='__main__':print(json.dumps(run(),indent=2))
