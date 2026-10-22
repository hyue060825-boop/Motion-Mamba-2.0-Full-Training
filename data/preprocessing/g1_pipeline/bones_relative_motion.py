"""Fit relative continuity flags using official BONES development motions only.

These are empirical anomaly flags, never controller or motor specifications.
Action labels may overlap; a clip is checked against every supported label.
"""
import argparse
import json
from collections import defaultdict
import mujoco
import numpy as np
from .common import ROOT, BONES, read_jsonl, write_json, sha
from .robot import compile_target, read_csv, FORMAT
from .resample import qpos as resample


LABELS = {
    'locomotion': ('walking', 'running', 'jogging'),
    'turning': ('turning', 'spinning'),
    'crouching': ('crouching', 'lunging', 'bowing down'),
    'jumping': ('jumping', 'vaulting', 'flipping'),
    'upper_body': ('gesture', 'pointing', 'waving', 'communication',
                   'object manipulation', 'throwing', 'punching', 'pulling'),
    'standing_idle': ('standing', 'standing idle', 'idle', 'looking'),
    'dancing_exercising': ('dancing', 'exercising', 'kicking', 'sports'),
    'ground_body_support': ('sitting', 'sitting idle', 'lying down', 'rolling',
                            'crawling', 'kneeling', 'on hands and knees'),
}


def labels(row):
    tokens = {s.strip().lower() for s in str(row.get('category', '')).split(',')}
    return [name for name, values in LABELS.items() if tokens.intersection(values)]


def series(model, q, t):
    if len(q) < 3 or q.shape != (len(t), 36) or not np.isfinite(q).all():
        raise ValueError('INVALID_CONTINUITY_TRAJECTORY')
    dt = np.diff(t)
    if not np.allclose(dt, .04, atol=1e-6, rtol=0):
        raise ValueError('NON_25HZ_TIME')
    velocity = np.empty((len(q)-1, model.nv))
    for i, step in enumerate(dt):
        mujoco.mj_differentiatePos(model, velocity[i], step, q[i], q[i+1])
    acceleration = np.diff(velocity, axis=0) / ((dt[:-1]+dt[1:])*.5)[:, None]
    return {'joint_speed_rad_s': abs(velocity[:, 6:]),
            'joint_acceleration_rad_s2': abs(acceleration[:, 6:]),
            'root_speed_m_s': np.linalg.norm(velocity[:, :3], axis=1)[:, None],
            'root_angular_speed_rad_s': np.linalg.norm(velocity[:, 3:6], axis=1)[:, None],
            'root_acceleration_m_s2': np.linalg.norm(acceleration[:, :3], axis=1)[:, None]}


def threshold(maxima, floor):
    """One independent maximum per clip; equal clip weights, per coordinate."""
    values = np.asarray(maxima)
    q25, q75, q99 = np.percentile(values, [25, 75, 99], axis=0)
    return q99 + 3*np.maximum(q75-q25, floor)


def run(namespace):
    if not namespace.replace('_', '').isalnum():
        raise ValueError('INVALID_NAMESPACE')
    manifest = ROOT/'manifests/bones_pairs_dev.jsonl'
    rows = read_jsonl(manifest)
    if len(rows) != 200 or any(r['adapter_split'] != 'dev' or r['project_split'] != 'train' for r in rows):
        raise ValueError('ONLY_FIXED_TRAIN_DEV200_ALLOWED')
    model, asset = compile_target()
    root = ROOT/'work'/namespace
    floors = {'joint_speed_rad_s': .01, 'joint_acceleration_rad_s2': .1,
              'root_speed_m_s': .001, 'root_angular_speed_rad_s': .01,
              'root_acceleration_m_s2': .01}
    prepared = []; groups = defaultdict(list)
    for row in rows:
        source = BONES/row['source_paths']['g1']
        reference, t, _ = read_csv(source, format_id=FORMAT, fps=row['original_fps'])
        reference, t, _ = resample(reference, t, model.jnt_range[1:])
        values = series(model, reference, t)
        item = {'row': row, 'reference': values, 'q': reference, 't': t,
                'source_sha256': sha(source), 'labels': labels(row)}
        prepared.append(item)
        for label in item['labels']:
            groups[label].append(item)
    minimum_support = 5
    baseline = {}
    for label in LABELS:
        members = groups[label]
        supported = len(members) >= minimum_support
        baseline[label] = {
            'independent_clips': len(members), 'supported': supported,
            'thresholds': {key: threshold([m['reference'][key].max(axis=0) for m in members], floor).tolist()
                           for key, floor in floors.items()} if supported else None,
            'reason': None if supported else 'fewer than five independent development clips',
            'sample_ids': [m['row']['sample_id'] for m in members]}
    results = []
    for item in prepared:
        row = item['row']; mid = row['source_motion_id']
        qp = root/(mid+'.npy'); tp = root/(mid+'.time.json')
        record = {'sample_id': row['sample_id'], 'labels': item['labels'],
                  'source_sha256': item['source_sha256'], 'status': 'ERROR'}
        try:
            t = np.asarray(json.loads(tp.read_text())['source']['target_timestamps'])
            # Paired support ends at the earlier SOMA/CSV last sample. BVH's
            # rounded Frame Time can legitimately remove the final CSV sample.
            # The full official clip still defines the independent baseline.
            if len(t)>len(item['t']) or not np.allclose(t, item['t'][:len(t)], atol=1e-9, rtol=0):
                raise ValueError('CANDIDATE_REFERENCE_TIME_MISMATCH')
            candidate = series(model, np.load(qp), t)
            reference = series(model, item['q'][:len(t)], t)
            supported = [label for label in item['labels'] if baseline[label]['supported']]
            roles = {}
            for role, values in [('candidate', candidate), ('reference', reference)]:
                roles[role] = {}
                for label in supported:
                    roles[role][label] = {}
                    for key, value in values.items():
                        exceed = value > np.asarray(baseline[label]['thresholds'][key])
                        roles[role][label][key] = {
                            'samples': len(value), 'per_coordinate_max': value.max(axis=0).tolist(),
                            'per_coordinate_exceed_fraction': exceed.mean(axis=0).tolist(),
                            'any_coordinate_exceed_fraction': float(exceed.any(axis=1).mean())}
            record.update(status='MEASURED' if supported else 'UNSUPPORTED_CATEGORY',
                          reason=None if supported else 'no sufficiently supported action label',
                          candidate_sha256=sha(qp), time_sha256=sha(tp), roles=roles)
            record['reference_full_frames'] = len(item['t'])
            record['paired_frames'] = len(t)
        except Exception as exc:
            record['error'] = repr(exc)
        results.append(record)
    report = {'status': 'DEVELOPMENT_RELATIVE_FLAGS_NOT_HARDWARE_ACCEPTANCE',
              'namespace': namespace, 'expected': 200, 'attempted': len(results),
              'measured': sum(r['status']=='MEASURED' for r in results),
              'unsupported': sum(r['status']=='UNSUPPORTED_CATEGORY' for r in results),
              'errors': sum(r['status']=='ERROR' for r in results),
              'formula': 'P99 of official per-clip coordinate maxima + 3 * max(IQR, numerical floor)',
              'numerical_iqr_floors': floors, 'action_labels': LABELS,
              'overlap_policy': 'evaluate every supported metadata label; no label selected using motion quality',
              'minimum_support': minimum_support, 'baseline': baseline, 'results': results,
              'limitations': 'Small category samples give uncertain tails; flags require diagnosis, do not prove motor feasibility or automatically accept clips.',
              'hashes': {'manifest': sha(manifest), 'code': sha(__file__), 'scene_policy': sha(ROOT/'g1_pipeline/scene_policy.py')},
              'asset_hash': asset['asset_hash'], 'held_out_read': False}
    write_json(ROOT/'qa/aggregate'/f'{namespace}_relative_motion.json', report)
    print({k: report[k] for k in ('expected', 'attempted', 'measured', 'unsupported', 'errors')}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--namespace', required=True)
    run(parser.parse_args().namespace)
