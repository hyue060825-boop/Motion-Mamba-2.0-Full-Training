"""Header-only exported-rig proportion audit; spans are not actor body heights."""
import numpy as np
from collections import defaultdict
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha,digest


def nodes(path):
    result=[];stack=[]
    with path.open() as stream:
        for line in stream:
            p=line.split()
            if not p:continue
            if p[0]=='MOTION':break
            if p[0] in ('ROOT','JOINT'):
                result.append({'name':p[1],'parent':stack[-1] if stack else -1,'offset':None,'end_site':False});stack.append(len(result)-1)
            elif p[:2]==['End','Site']:
                result.append({'name':result[stack[-1]]['name']+'#EndSite','parent':stack[-1],'offset':None,'end_site':True});stack.append(len(result)-1)
            elif p[0]=='OFFSET':result[stack[-1]]['offset']=[float(x)*.01 for x in p[1:]]
            elif p[0]=='}':stack.pop()
    if stack or any(r['offset'] is None for r in result):raise ValueError('INVALID_HEADER')
    return result


def run():
    manifest=ROOT/'manifests/hiphi_smoke100.jsonl';rows=[];actors=defaultdict(list);rigs=defaultdict(list)
    for row in read_jsonl(manifest):
        path=ROOT/'work/hiphi_packages'/(row['source_motion_id']+'.bvh');skeleton=nodes(path);p=[]
        for node in skeleton:p.append(np.asarray(node['offset'])+(p[node['parent']] if node['parent']>=0 else 0))
        p=np.array(p);span=float(np.ptp(p[:,1]));metadata=float(row['height_m'])
        rig_hash=digest(skeleton)
        record={'sample_id':row['sample_id'],'actor_id':row['actor_id_namespaced'],'metadata_height_m':metadata,
                'rest_offset_height_proxy_m':span,'proxy_over_metadata_ratio':span/metadata,'rig_hash':rig_hash,
                'source_sha256':sha(path),'needs_height_strategy_review':abs(span/metadata-1)>.1,
                'height_interpretation':'coordinate span of rest offsets including End Sites; not a mesh-derived true body height'}
        rows.append(record);actors[row['actor_id_namespaced']].append(record);rigs[rig_hash].append(record)
    report={'status':'INPUT_PROPORTION_AUDIT_NOT_ADAPTER_ACCEPTANCE','manifest_sha256':sha(manifest),'code_sha256':sha(__file__),
            'expected':100,'measured':len(rows),'height_review_flags':sum(r['needs_height_strategy_review'] for r in rows),
            'actor_count':len(actors),'distinct_exported_rig_hashes':len(rigs),
            'same_rig_multiple_actors':[{'rig_hash':k,'actors':sorted({r['actor_id'] for r in v}),'metadata_heights_m':sorted({r['metadata_height_m'] for r in v})} for k,v in rigs.items() if len({r['actor_id'] for r in v})>1],
            'per_actor_proxy_range':{k:[min(r['rest_offset_height_proxy_m'] for r in v),max(r['rest_offset_height_proxy_m'] for r in v)] for k,v in actors.items()},
            'rows':rows,'rule':'flag >10% proxy/metadata discrepancy for input review; do not reinterpret proxy as measured actor height or tune retarget quality on this audit'}
    write_json(ROOT/'audit/hiphi_rest_proportions_v1.json',report)
    print({k:report[k] for k in ['measured','height_review_flags','actor_count','distinct_exported_rig_hashes']})
    print('shared rigs across actors',len(report['same_rig_multiple_actors']))


if __name__=='__main__':run()
