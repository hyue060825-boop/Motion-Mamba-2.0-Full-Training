"""Join the full development denominator, scene metadata and QA failure evidence."""
import json
from collections import Counter
import numpy as np
from .common import ROOT,read_jsonl,write_json,sha
from .scene_policy import classify


def analyze(namespace='bones_soma_rest_dev_v4'):
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';meta=read_jsonl(manifest)
    qa_path=ROOT/'work'/namespace/'pair_qa.json';summary_path=ROOT/'qa/aggregate'/f'{namespace}_qa_summary.json'
    qa=json.loads(qa_path.read_text());sm=json.loads(summary_path.read_text())
    perqa={r['sample_id']:r for r in qa['results']};persm={r['sample_id']:r for r in sm['clips']}
    rows=[];categories={}
    for row in meta:
        q=perqa[row['sample_id']];s=persm[row['sample_id']];scene=classify(row)
        failures={'measured_pairwise':s.get('measured_pairwise_grace_failures',[])}
        for role in ('candidate','reference'):
            g=q.get(role+'_geometry',{});reasons=[]
            for metric,threshold in [('ground_penetration_p95_m',.02),('ground_penetration_max_m',.05),('raw_self_collision_depth_over_1cm_frame_fraction',.001),('raw_self_collision_max_depth_m',.03)]:
                if g.get(metric,float('inf'))>threshold:reasons.append(metric)
            failures[role+'_raw_geometry']=reasons
        entry={'sample_id':row['sample_id'],'scene':scene,'failures':failures,
               'geometry_scope':'raw geometry only; self collision permitted-contact policy not frozen',
               'pairwise_metrics':s.get('metrics'),'frames':q.get('frames')}
        rows.append(entry);categories.setdefault(scene['scene_class'],[]).append(entry)
    counts={}
    for scene,group in categories.items():
        counts[scene]={'input_count':len(group),'measured_pairwise_grace_pass':sum(not r['failures']['measured_pairwise'] for r in group),
          'candidate_raw_geometry_pass':sum(not r['failures']['candidate_raw_geometry'] for r in group),
          'reference_raw_geometry_pass':sum(not r['failures']['reference_raw_geometry'] for r in group),
          'pairwise_and_both_raw_geometry_pass':sum(not any(r['failures'].values()) for r in group)}
    pair_counts=Counter()
    for q in qa['results']:
        for pair,n in q.get('reference_geometry',{}).get('raw_self_contact_pair_counts',{}).items():pair_counts[pair]+=n
    report={'status':'DEVELOPMENT_FAILURE_ANALYSIS','expected':200,'joined':len(rows),'namespace':namespace,
      'input_hashes':{str(p):sha(p) for p in [manifest,qa_path,summary_path]},'scene_groups':counts,
      'reference_contact_pair_counts':dict(pair_counts.most_common()),'rows':rows,
      'note':'No clips removed or reclassified using QA values; scene metadata rules are development-only and require freeze before release.'}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_failure_analysis.json',report)
    print(counts,flush=True);return report
if __name__=='__main__':analyze()
