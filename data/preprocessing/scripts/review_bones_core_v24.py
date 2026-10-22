"""Join complete measured evidence; never promotes a development run to release."""
import json
import numpy as np
from collections import Counter
from g1_pipeline.common import ROOT,write_json,sha


def run():
    ns='bones_core_dev_v24'
    paths={k:ROOT/'qa/aggregate'/f'{v}.json' for k,v in {
        'run':ns,'summary':ns+'_qa_summary','relative':ns+'_relative_motion_v2',
        'targets':ns+'_human_targets','base_targets':'bones_collision_dev_v19_human_targets'}.items()}
    evidence={k:json.loads(p.read_text()) for k,p in paths.items()}
    paths['geometry']=ROOT/'work'/ns/'pair_qa.json';evidence['geometry']=json.loads(paths['geometry'].read_text())
    maps={k:{r['sample_id']:r for r in evidence[k]['results']} for k in ('relative','targets','base_targets','geometry')}
    paired={r['sample_id']:r for r in evidence['summary']['clips']}
    rows=[]
    for r in evidence['run']['results']:
        sid=r['sample_id'];g=maps['geometry'][sid];rel=maps['relative'][sid]
        cg=g['candidate_geometry'];rg=g['reference_geometry']
        out={'sample_id':sid,'scene_core_candidate':r['scene_core_candidate'],'scene_class':r['scene_class'],
             'quality_rejection_reasons':r['quality_rejection_reasons'],
             'ground_lift_max_m':r['ground_lift_max_m'],'rate_correction_max_rad':r['rate_correction_max_rad'],
             'paired_grace_failures':paired[sid]['measured_pairwise_grace_failures']}
        for role,geom in [('candidate',cg),('reference',rg)]:
            out[role+'_ground_pass']=geom['ground_penetration_p95_m']<=.02 and geom['ground_penetration_max_m']<=.05
            out[role+'_self_collision_pass']=geom['raw_self_collision_depth_over_1cm_frame_fraction']<=.001 and geom['raw_self_collision_max_depth_m']<=.03
        out['relative_motion_status']=rel['status']
        out['relative_motion_flags']={lab:{k:v['any_coordinate_exceed_fraction'] for k,v in ms.items() if v['any_coordinate_exceed_fraction']>0}
            for lab,ms in rel.get('roles',{}).get('candidate',{}).items()}
        out['relative_motion_flags']={k:v for k,v in out['relative_motion_flags'].items() if v}
        a=maps['targets'][sid];b=maps['base_targets'][sid]
        if a['status']!='MEASURED' or b['status']!='MEASURED':raise ValueError('INCOMPLETE_TARGET_QA')
        with np.load(a['per_frame_path']) as na,np.load(b['per_frame_path']) as nb:
            if not np.array_equal(na['timestamps'],nb['timestamps']):raise ValueError('TARGET_TIME_CHANGED')
            out['human_target_residual_change']={}
            for key in ['position_error_m','orientation_error_deg']:
                delta=na[key]-nb[key]
                out['human_target_residual_change'][key]={'worst_frame_increase':float(delta.max()),
                    'mean_change':float(delta.mean()),'candidate_max':float(na[key].max()),'base_max':float(nb[key].max())}
        rows.append(out)
    core=[r for r in rows if r['scene_core_candidate']]
    report={'status':'DEVELOPMENT_REVIEW_NOT_RELEASE','expected':200,'measured':len(rows),'core_clips':len(core),
      'core_ground_pass':sum(r['candidate_ground_pass'] for r in core),
      'core_self_collision_pass':sum(r['candidate_self_collision_pass'] for r in core),
      'core_within_transform_budget':sum(not r['quality_rejection_reasons'] for r in core),
      'core_relative_measured':sum(r['relative_motion_status']=='MEASURED' for r in core),
      'core_relative_flagged':sum(bool(r['relative_motion_flags']) for r in core),
      'scene_counts':dict(Counter(r['scene_class'] for r in rows)),
      'paired_grace_pass':evidence['summary']['pairwise_grace_pass_count'],
      'all_clips_keep_paired_denominator':True,'held_out_read':False,
      'outstanding':['relative outlier geometric/visual review','freeze source scene/contact/metric policy','independent held-out calibration'],
      'hardware_feasibility':None,'hardware_reason':'SONIC training rates are a development regularizer, not verified target motor specification',
      'results':rows,'hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths.values()}}
    write_json(ROOT/'audit/bones_core_v24_review.json',report)
    print({k:v for k,v in report.items() if k not in ('results','hashes')})


if __name__=='__main__':run()
