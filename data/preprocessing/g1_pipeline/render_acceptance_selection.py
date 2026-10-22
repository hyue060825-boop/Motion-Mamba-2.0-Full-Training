"""Metadata-stratified acceptance videos, followed by measured worst/review clips."""
import argparse
import json
from collections import defaultdict
from .common import ROOT,read_jsonl,write_json,sha,digest
from .calibration_protocol import require_frozen,PATH
from .bones_scene_evidence import load


def run(*,complete=False):
    require_frozen();namespace='bones_frozen_accept_v25';manifest=ROOT/'manifests/bones_pairs_accept.jsonl'
    rows=read_jsonl(manifest);scenes={r['sample_id']:r['scene']['scene_class'] for r in load(manifest)}
    bins=defaultdict(list)
    for row in rows:bins[scenes[row['sample_id']]].append(row)
    for group in bins.values():group.sort(key=lambda r:digest([20260916,'acceptance_visual',r['sample_id']]))
    selected={}
    while len(selected)<12:
        for scene,group in sorted(bins.items()):
            if group and len(selected)<12:selected[group.pop(0)['sample_id']]='preselected_source_scene_'+scene
    namespacepath=ROOT/'work'/namespace;done={r['sample_id']:r for r in read_jsonl(namespacepath/'results.jsonl')}
    if complete:
        summarypath=ROOT/'qa/aggregate'/f'{namespace}_qa_summary.json';summary=json.loads(summarypath.read_text())
        denominators={'hinge_mae_deg':20,'root_rmse_aligned_m':.15,'root_orientation_mean_deg':20,
            'torso_orientation_mean_deg':20,'left_foot_mean_m':.08,'right_foot_mean_m':.08,'left_wrist_mean_m':.15,'right_wrist_mean_m':.15}
        ranked=sorted([r for r in summary['clips'] if r['status']=='MEASURED'],
            key=lambda r:max(r['metrics'][k]/v for k,v in denominators.items()),reverse=True)
        added=0
        for row in ranked:
            if row['sample_id'] not in selected:selected[row['sample_id']]='worst_frozen_pairwise_grace_ratio';added+=1
            if added==8:break
        evaluation=json.loads((ROOT/'qa/aggregate'/f'{namespace}_frozen_evaluation.json').read_text())
        for row in evaluation['results']:
            if row['review_pending'] or row['core_scope'] and row['rejections']:selected.setdefault(row['sample_id'],'core_anomaly_or_rejection_review')
    eligible=[sid for sid in selected if done.get(sid,{}).get('status')=='DIAGNOSTIC_COMPLETE']
    path=ROOT/'manifests'/f'{namespace}_video_selection.json'
    write_json(path,{'policy_sha256':sha(PATH),'manifest_sha256':sha(manifest),'selection':selected,'available':eligible,
        'metadata_selection_independent_of_trajectory_quality':True,'complete_worst_and_review_selection':complete})
    from .render_frozen_review import run as render
    return render(namespace,eligible,source='bones',panel='accept')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');a=p.parse_args();run(complete=a.complete)
