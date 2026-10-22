"""Twenty diagnostic videos: 8 worst paired errors plus 12 metadata-stratified clips."""
import json
from .common import ROOT,read_jsonl,write_json,digest,sha
from .scene_policy import classify


def run(namespace='bones_soma_rest_dev_v4'):
    if not namespace.replace('_','').isalnum():raise ValueError('INVALID_NAMESPACE')
    summary=ROOT/'qa/aggregate'/f'{namespace}_qa_summary.json'
    qa=json.loads(summary.read_text());manifest=ROOT/'manifests/bones_pairs_dev.jsonl';metadata=read_jsonl(manifest)
    # Normalized ratios use the task's existing grace values, not selected cutoffs.
    denominators={'hinge_mae_deg':20,'root_rmse_aligned_m':.15,'root_orientation_mean_deg':20,
        'torso_orientation_mean_deg':20,'left_foot_mean_m':.08,'right_foot_mean_m':.08,'left_wrist_mean_m':.15,'right_wrist_mean_m':.15}
    ranking=sorted([r for r in qa['clips'] if r['status']=='MEASURED'],key=lambda r:max(r['metrics'][k]/v for k,v in denominators.items()),reverse=True)
    selected={r['sample_id']:'worst_existing_pairwise_error' for r in ranking[:8]}
    groups={}
    for row in metadata:
        if row['sample_id'] in selected:continue
        category=classify(row)['scene_class'];groups.setdefault(category,[]).append(row)
    for group in groups.values():group.sort(key=lambda r:digest({'seed':20260916,'sample_id':r['sample_id']}))
    while len(selected)<20:
        for category,group in sorted(groups.items()):
            if group and len(selected)<20:selected[group.pop(0)['sample_id']]='metadata_stratified_'+category
    path=ROOT/'manifests'/f'{namespace}_video20.json'
    write_json(path,{'scope':'development visual diagnosis; not representative quality sample','seed':20260916,
        'input_manifest_sha256':sha(manifest),'qa_summary_sha256':sha(summary),'selection':selected})
    from .render_bones_pairs import run as render
    return render(namespace,panel='dev',selection=selected)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--namespace',default='bones_soma_rest_dev_v4')
    run(p.parse_args().namespace)
