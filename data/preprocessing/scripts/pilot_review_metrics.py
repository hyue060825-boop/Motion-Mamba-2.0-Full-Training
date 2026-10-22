"""Read-only metrics index for the two explicitly authorized pilots."""
import argparse
import json
import math
from g1_pipeline.common import ROOT, read_jsonl, sha, write_json


def run(namespace):
    assert namespace.startswith('pilot_')
    rows = read_jsonl(ROOT / 'work' / namespace / 'results.jsonl')
    results = []
    for r in rows:
        if not r.get('metrics_path'):
            results.append({'sample_id': r['sample_id'], 'execution_error': r.get('error')})
            continue
        p = ROOT / r['metrics_path']
        m = json.loads(p.read_text())
        targets = m['human_target_metrics']
        native = r.get('source_metadata', {}).get('source_paths', {}).get('native', '')
        results.append({
            'sample_id': r['sample_id'], 'metrics_sha256': sha(p),
            'source_native_path': native,
            'contract_status': r['contract_status'], 'hard_rejections': r['rejections'],
            'flags': r['review_flags'],
            'ground_lift_max_m': (m['ground_correction'] or {}).get('lift_max_m', 0),
            'rate_correction_max_rad': (m['rate_projection'] or {}).get('correction_max_rad', 0),
            'human_target_residual_change': m['human_target_residual_change'],
            'root_target_position_mean_m': targets['pelvis']['position_mean_m'],
            'feet_target_position_mean_max_m': max(targets[b]['position_mean_m'] for b in targets if 'ankle' in b),
            'source_high_band_displacement_rms_max_deg': math.degrees(m.get('native_spectrum', {}).get('high_band_displacement_rms_max', 0)),
            'continuity': m['continuity'],
            'relative_motion': m['relative_motion'],
        })
    write_json(ROOT / 'audit' / f'{namespace}_review_metrics.json', {
        'scope': 'Read-only full-series numerical review index; no added acceptance thresholds, no solver or source changes',
        'ledger_sha256': sha(ROOT / 'work' / namespace / 'results.jsonl'),
        'results': results,
    })
    print({'namespace': namespace, 'indexed': len(results)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--namespace', required=True)
    run(p.parse_args().namespace)
