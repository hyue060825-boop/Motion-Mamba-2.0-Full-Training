"""Three worst development core collisions: human-target refinement feasibility."""
import json
import numpy as np
import mujoco
from g1_pipeline.common import ROOT, read_jsonl, write_json, sha
from g1_pipeline.gates import require_pass
from g1_pipeline.locked_target import prepare, create_retargeter
from g1_pipeline.gmr_soma import parse
from g1_pipeline.gmr_collision_refinement import refine, collision_pairs, collision_terms
from g1_pipeline.scene_policy import classify


def run():
    require_pass('00_preflight')
    namespace = 'bones_collision_refinement_v16'
    out = ROOT/'work'/namespace
    if out.exists():
        raise ValueError('DIAGNOSTIC_NAMESPACE_EXISTS')
    out.mkdir()
    manifest = ROOT/'manifests/bones_pairs_dev.jsonl'
    metadata = {r['sample_id']: r for r in read_jsonl(manifest)}
    prior = ROOT/'work/bones_soma_weighted_dev_v8/pair_qa.json'
    rows = [r for r in json.loads(prior.read_text())['results']
            if r['status']=='MEASURED' and classify(metadata[r['sample_id']])['core_eligible']]
    selected = sorted(rows, key=lambda r: (-r['candidate_geometry']['raw_self_collision_max_depth_m'], r['sample_id']))[:3]
    cfgpath = ROOT/'configs/bones_temporal_arms_v15.json'; cfg = json.loads(cfgpath.read_text())
    xml, asset = prepare(); results = []
    design = {'status': 'DEVELOPMENT_SINGLE_FRAME_FEASIBILITY_ONLY',
              'selection': 'three largest v8 candidate maximum self penetrations in fixed dev200 metadata flat core; worst v15 frame per clip',
              'sample_ids': [r['sample_id'] for r in selected], 'full_clip_acceptance': False,
              'posture_weight': 5., 'collision_penalty': 2000., 'maximum_nfev': 150,
              'allowed_penetration_m': .002, 'held_out_read': False,
              'hashes': {str(p): sha(p) for p in [manifest, prior, cfgpath,
                         ROOT/'g1_pipeline/gmr_collision_refinement.py', ROOT/'scripts/diagnose_bones_collision_refinement.py']},
              'asset_hash': asset['asset_hash'], 'resources': {'workers': 1, 'cpu_threads': 1, 'gpu': False}}
    write_json(out/'design.json', design)
    for entry in selected:
        row = metadata[entry['sample_id']]; mid = row['source_motion_id']
        qp = ROOT/'work/bones_temporal_arms_dev_v15'/(mid+'.npy')
        tp = qp.with_suffix('.time.json'); q = np.load(qp)
        times = np.asarray(json.loads(tp.read_text())['source']['target_timestamps'])
        solver = create_retargeter(cfgpath, xml); m = solver.model; d = mujoco.MjData(m)
        worst = -1.; index = None
        for i, pose in enumerate(q):
            d.qpos[:] = pose; mujoco.mj_forward(m, d)
            depth = max([-c.dist for c in d.contact if m.geom_bodyid[c.geom1] and m.geom_bodyid[c.geom2]]+[0.])
            if depth > worst:
                worst = depth; index = i
        bvh = ROOT/'work/bones_pairs_dev'/(mid+'.bvh')
        frames, _ = parse(bvh, {n: n for n in cfg['human_scale_table']}, target_timestamps=times, rotation_basis='world_only')
        # Independent finite difference of the contact terms at the actual seed.
        pairs = collision_pairs(m); solver.configuration.update(q[index])
        _, analytic = collision_terms(m, solver.configuration.data, pairs)
        difference = []
        for joint in range(29):
            values = []
            for sign in (-1, 1):
                pose = q[index].copy(); pose[7+joint] += sign*1e-6
                solver.configuration.update(pose)
                values.append(collision_terms(m, solver.configuration.data, pairs)[0])
            difference.append((values[1]-values[0])/2e-6)
        numeric = np.array(difference).T
        refined, info = refine(solver, frames[index], q[index])
        np.savez(out/(mid+'.npz'), seed=q[index], refined=refined, frame=index, time=times[index])
        record = {'sample_id': row['sample_id'], 'frame': index, 'before_raw_depth_m': float(worst),
                  'collision_jacobian_max_abs_error': float(np.max(abs(numeric-analytic[:, 6:]))),
                  'candidate_sha256': sha(qp), 'source_sha256': sha(bvh), **info}
        results.append(record); print(record, flush=True)
        write_json(ROOT/'audit'/f'{namespace}.json', {'design': design, 'results': results})


if __name__ == '__main__':
    run()
