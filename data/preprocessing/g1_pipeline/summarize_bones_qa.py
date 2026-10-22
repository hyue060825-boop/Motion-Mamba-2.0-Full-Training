"""Development QA summary with explicit missing gate requirements."""
import numpy as np
import json
from .common import ROOT,write_json,sha


def summarize(namespace='bones_soma_rest_smoke_v3'):
    runpath=ROOT/'qa/aggregate'/f'{namespace}.json';qapath=ROOT/'work'/namespace/'pair_qa.json'
    run=json.loads(runpath.read_text());qa=json.loads(qapath.read_text());rows={r['sample_id']:r for r in qa['results']}
    per_clip=[]
    def distribution(values,weights):
        a=np.array(values);w=np.array(weights)
        return {'clip_macro_mean':float(a.mean()),'clip_median':float(np.median(a)),
                'clip_p05':float(np.percentile(a,5)),
                'clip_p95':float(np.percentile(a,95)),'clip_max':float(a.max()),
                'frame_weighted_mean_of_clip_metrics':float(np.average(a,weights=w))}
    for r in run['results']:
        q=rows.get(r['sample_id'],{})
        if r['status']!='DIAGNOSTIC_COMPLETE' or q.get('status')!='MEASURED':
            per_clip.append({'sample_id':r['sample_id'],'status':'ERROR','reason':r.get('error') or q.get('error')});continue
        metrics={k:r[k] for k in ('hinge_mae_deg','root_rmse_aligned_m','root_orientation_mean_deg')}
        metrics['torso_orientation_mean_deg']=q['torso_orientation_mean_deg']
        metrics.update({k+'_mean_m':v for k,v in q['extremity_world_aligned_mean_m'].items()})
        contacts=list(q['contact_agreement'].values())
        f1=[v['f1'] for v in contacts if v['f1'] is not None];ba=[v['balanced_accuracy'] for v in contacts if v['balanced_accuracy'] is not None]
        metrics['contact_f1_supported_macro']=float(np.mean(f1)) if f1 else None
        metrics['contact_balanced_accuracy_supported_macro']=float(np.mean(ba)) if ba else None
        failures=[]
        for k,threshold in [('hinge_mae_deg',20),('root_rmse_aligned_m',.15),('root_orientation_mean_deg',20),('torso_orientation_mean_deg',20),('left_foot_mean_m',.08),('right_foot_mean_m',.08),('left_wrist_mean_m',.15),('right_wrist_mean_m',.15)]:
            if metrics[k]>threshold:failures.append(k)
        if f1 and min(f1)<.7:failures.append('contact_f1_channel_below_0.70')
        per_clip.append({'sample_id':r['sample_id'],'status':'MEASURED','frames':r['frames'],'metrics':metrics,
                         'measured_pairwise_grace_failures':failures,'candidate_contract':q['candidate_contract']['status'],
                         'reference_contract':q['reference_contract']['status'],'source_props':q.get('source_props'),
                         'kinematic_release':'NOT_EVALUATED: collision allowlist and scene categories not frozen'})
    measured=[r for r in per_clip if r['status']=='MEASURED'];aggregate={}
    for key in measured[0]['metrics'] if measured else []:
        present=[r for r in measured if r['metrics'][key] is not None]
        aggregate[key]=distribution([r['metrics'][key] for r in present],[r['frames'] for r in present]) if present else None
        if aggregate[key] is not None:aggregate[key]['supported_clips']=len(present)
    complete_dev=run['run_config'].get('panel')=='dev' and len(measured)==run['fixed_panel_size']==200
    contact_channels={}
    for name in sorted({name for row in qa['results'] for name in row.get('contact_agreement',{})}):
        entries=[(row['contact_agreement'][name],row['frames']) for row in qa['results'] if row.get('status')=='MEASURED' and name in row.get('contact_agreement',{})]
        channel={key:sum(e[key] for e,_ in entries) for key in ('tp','tn','fp','fn')}
        for metric in ('f1','balanced_accuracy'):
            supported=[(e[metric],n) for e,n in entries if e[metric] is not None]
            channel[metric]=distribution([e for e,n in supported],[n for e,n in supported]) if supported else None
            channel[metric+'_supported_clips']=len(supported)
            channel[metric+'_unsupported_clips']=len(entries)-len(supported)
        contact_channels[name]=channel
    report={'status':'DEVELOPMENT_DIAGNOSTIC','namespace':namespace,'expected':run['fixed_panel_size'],
            'attempted':len(run['results']),'measured':len(measured),'frames':sum(r['frames'] for r in measured),
            'candidate_contract_pass':sum(r['candidate_contract']=='PASS' for r in measured),
            'reference_contract_pass':sum(r['reference_contract']=='PASS' for r in measured),
            'pairwise_grace_pass_count':sum(not r['measured_pairwise_grace_failures'] for r in measured),
            'fixed_dev200_complete':complete_dev,'contact_channels':contact_channels,
            'aggregate':aggregate,'clips':per_clip,'input_sha256':{str(runpath):sha(runpath),str(qapath):sha(qapath)},
            'release_gate':'BLOCKED','missing':([] if complete_dev else ['fixed dev200 completed'])+['configuration frozen','independent accept300',
                'scene and allowed-contact policy','validated controller envelope and dynamic QA',
                'complete per-body unweighted IK residuals and continuity baselines']}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_qa_summary.json',report)
    print({k:v for k,v in report.items() if k not in ('aggregate','clips','input_sha256')})
    return report
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--namespace',default='bones_soma_rest_smoke_v3');a=p.parse_args();summarize(a.namespace)
