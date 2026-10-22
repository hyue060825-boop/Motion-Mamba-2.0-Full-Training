"""Compare coarse self-contact proxies with native visual convex mesh distances.

Development evidence only. Visual convex hull distances are not exact triangle
mesh penetration and cannot silently replace the frozen collision model.
"""
from collections import defaultdict
import numpy as np
import mujoco
from g1_pipeline.common import ROOT,BONES,read_jsonl,write_json,sha
from g1_pipeline.robot import compile_target,read_csv,FORMAT
from g1_pipeline.resample import qpos as resample
from g1_pipeline.scene_policy import classify


def run():
    m,asset=compile_target();d=mujoco.MjData(m);floor=m.geom('floor').id
    manifest=ROOT/'manifests/bones_pairs_dev.jsonl';rows=[r for r in read_jsonl(manifest) if classify(r)['core_eligible']]
    visual={b:[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g]==2] for b in range(m.nbody)}
    results=[];by_pair=defaultdict(list)
    for row in rows:
        p=BONES/row['source_paths']['g1'];q,t,_=read_csv(p,format_id=FORMAT,fps=row['original_fps']);q,t,_=resample(q,t,m.jnt_range[1:])
        for index in range(0,len(q),10):
            d.qpos[:]=q[index];mujoco.mj_forward(m,d);seen=set()
            for contact in d.contact:
                if floor in (contact.geom1,contact.geom2) or contact.dist>=-.01:continue
                b1,b2=sorted([int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2])])
                if (b1,b2) in seen:continue
                seen.add((b1,b2));distances=[]
                for a in visual[b1]:
                    for b in visual[b2]:distances.append(float(mujoco.mj_geomDistance(m,d,a,b,1.,None)))
                pair=m.body(b1).name+' | '+m.body(b2).name
                value=min(distances) if distances else None
                record={'sample_id':row['sample_id'],'frame_25hz':index,'bodies':pair,'coarse_proxy_penetration_m':-float(contact.dist),
                        'visual_convex_min_signed_distance_m':value,'visual_pair_count':len(distances)}
                results.append(record);by_pair[pair].append(record)
    summary={}
    for pair,records in by_pair.items():
        values=[r['visual_convex_min_signed_distance_m'] for r in records if r['visual_convex_min_signed_distance_m'] is not None]
        summary[pair]={'events':len(records),'visual_measured':len(values),'visual_separated_events':sum(v>0 for v in values),
                       'visual_penetration_over_1cm_events':sum(v<-.01 for v in values),
                       'visual_signed_distance_min_m':min(values) if values else None,
                       'coarse_depth_max_m':max(r['coarse_proxy_penetration_m'] for r in records)}
    report={'status':'DEVELOPMENT_COLLISION_PROXY_DIAGNOSTIC','scope':'official dev flat-core candidates, every tenth 25Hz frame, raw coarse contacts >1cm',
            'fixed_core_candidates':len(rows),'events':len(results),'manifest_sha256':sha(manifest),'asset_hash':asset['asset_hash'],
            'code_sha256':sha(__file__),'method':'MuJoCo explicit geomDistance between visual convex meshes of the contacted bodies, including contype=0 visual geometry',
            'limitations':'convex hull distances do not establish exact mesh contact intent; no allowlist or threshold changed',
            'pair_summary':summary,'results':results}
    write_json(ROOT/'audit/bones_collision_proxy_comparison.json',report)
    print({'core_clips':len(rows),'events':len(results),'visual_separated_events':sum(r['visual_convex_min_signed_distance_m'] is not None and r['visual_convex_min_signed_distance_m']>0 for r in results)})


if __name__=='__main__':run()
