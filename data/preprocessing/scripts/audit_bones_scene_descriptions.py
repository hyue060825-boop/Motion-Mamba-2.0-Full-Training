"""Classify all fixed dev200 using source descriptions; preserve full denominator."""
import json
from collections import Counter
from g1_pipeline.common import ROOT,write_json,sha
from g1_pipeline.bones_scene_evidence import load


def run():
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';rows=load(manifest)
    posture={r['sample_id']:r for r in json.loads((ROOT/'audit/bones_source_posture_v1.json').read_text())['results']}
    qa={r['sample_id']:r for r in json.loads((ROOT/'work/bones_collision_dev_v19/pair_qa.json').read_text())['results']}
    changes=[];unresolved=[]
    for row in rows:
        row['human_posture_review']=posture[row['sample_id']]['ground_multisupport_review']
        if row['scene']['scene_class']!=row['metadata_scene_without_description']['scene_class']:changes.append(row['sample_id'])
        if row['human_posture_review'] and row['scene']['core_eligible']:unresolved.append(row['sample_id'])
    # The following is a downstream join for diagnosis, not a classifier input.
    groups={}
    for row in rows:
        groups.setdefault(row['scene']['scene_class'],[]).append(row)
    counts={}
    for group,members in groups.items():
        counts[group]={'input_clips':len(members)}
        for role in ('candidate','reference'):
            g=[qa[row['sample_id']][role+'_geometry'] for row in members]
            counts[group][role+'_ground_pass']=sum(x['ground_penetration_p95_m']<=.02 and x['ground_penetration_max_m']<=.05 for x in g)
            counts[group][role+'_self_collision_pass']=sum(x['raw_self_collision_depth_over_1cm_frame_fraction']<=.001 and x['raw_self_collision_max_depth_m']<=.03 for x in g)
    result={'status':'SOURCE_DESCRIPTION_SCENE_REVIEW','expected':200,'measured':len(rows),
            'scene_counts':counts,'changed_scene_ids':changes,'remaining_core_human_posture_review_ids':unresolved,
            'rows':rows,'classifier_uses_robot_quality':False,'input_manifest_unchanged':True,
            'hashes':{str(p):sha(p) for p in [manifest,ROOT/'g1_pipeline/scene_policy.py',ROOT/'g1_pipeline/bones_scene_evidence.py',
                                           ROOT/'audit/bones_source_posture_v1.json',ROOT/'scripts/audit_bones_scene_descriptions.py']}}
    write_json(ROOT/'audit/bones_description_scene_v3.json',result)
    print(counts,'changes',changes,'remaining review',unresolved,flush=True)


if __name__=='__main__':run()
