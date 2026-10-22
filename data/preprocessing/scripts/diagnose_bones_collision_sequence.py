"""Full three-clip development trial of bounded collision corrections."""
import io
import json
import time
import numpy as np
from g1_pipeline.common import ROOT, read_jsonl, write_json, sha, atomic_bytes
from g1_pipeline.gates import require_pass
from g1_pipeline.locked_target import prepare, create_retargeter
from g1_pipeline.gmr_soma import parse
from g1_pipeline.gmr_collision_refinement import refine
from g1_pipeline.bones_continuity_qa import metrics as continuity
from g1_pipeline.bones_pair_qa import run as pair_qa
from g1_pipeline.summarize_bones_qa import summarize


def run():
    require_pass('00_preflight')
    namespace = 'bones_collision_sequence_v17'; out = ROOT/'work'/namespace
    if out.exists():
        raise ValueError('DIAGNOSTIC_NAMESPACE_EXISTS')
    out.mkdir()
    previous = json.loads((ROOT/'audit/bones_collision_refinement_v16.json').read_text())
    selected = previous['design']['sample_ids']
    rows = {r['sample_id']: r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl')}
    source_run = ROOT/'qa/aggregate/bones_temporal_arms_dev_v15.json'
    source = {r['sample_id']: r for r in json.loads(source_run.read_text())['results']}
    cfgpath = ROOT/'configs/bones_temporal_arms_v15.json'; cfg = json.loads(cfgpath.read_text())
    xml, asset = prepare(); results = []
    settings = {'penalty': 2000., 'posture_weight': 5., 'max_nfev': 80, 'correction_velocity_weight': .4}
    design = {'status': 'DEVELOPMENT_SUBSET_NOT_FIXED_PANEL_PASS', 'sample_ids': selected,
              'selection': 'same three worst development collision clips as v16; all frames',
              'settings': settings, 'root_fixed': True, 'time_fixed': True, 'held_out_read': False,
              'expected_panel': 200, 'selected_clips': 3, 'reference_used_in_objective': False,
              'hashes': {str(p): sha(p) for p in [source_run, cfgpath,
                         ROOT/'g1_pipeline/gmr_collision_refinement.py', ROOT/'scripts/diagnose_bones_collision_sequence.py']},
              'asset_hash': asset['asset_hash'], 'resources': {'workers': 1, 'cpu_threads': 1, 'gpu': False}}
    write_json(out/'run.json', design)
    for sid in selected:
        started = time.monotonic(); row = rows[sid]; mid = row['source_motion_id']
        original = ROOT/'work/bones_temporal_arms_dev_v15'/(mid+'.npy')
        timepath = original.with_suffix('.time.json'); q = np.load(original)
        mapping = json.loads(timepath.read_text()); t = np.asarray(mapping['source']['target_timestamps'])
        bvh = ROOT/'work/bones_pairs_dev'/(mid+'.bvh')
        frames, _ = parse(bvh, {n: n for n in cfg['human_scale_table']}, target_timestamps=t, rotation_basis='world_only')
        solver = create_retargeter(cfgpath, xml); corrected = []; infos = []; delta = None
        for i, (frame, seed) in enumerate(zip(frames, q)):
            pose, info = refine(solver, frame, seed, previous_correction=delta, **settings)
            delta = pose[7:]-seed[7:]; corrected.append(pose); infos.append(info)
            if i % 25 == 0:
                print(sid, i, '/', len(q), info['maximum_penetration_excess_after_m'], flush=True)
        corrected = np.asarray(corrected)
        if not np.array_equal(corrected[:, :7], q[:, :7]):
            raise ValueError('ROOT_CHANGED_DURING_COLLISION_CORRECTION')
        buf = io.BytesIO(); np.save(buf, corrected); atomic_bytes(out/(mid+'.npy'), buf.getvalue())
        atomic_bytes(out/(mid+'.time.json'), timepath.read_bytes())
        write_json(out/(mid+'.solver.json'), {'frames': infos, 'settings': settings})
        # Root and time metrics are unchanged; recompute hinge errors with the
        # official reference only after all optimization has completed.
        from g1_pipeline.robot import read_csv, FORMAT
        from g1_pipeline.common import BONES
        from g1_pipeline.resample import qpos as resample
        reference, rt, _ = read_csv(BONES/row['source_paths']['g1'], format_id=FORMAT, fps=row['original_fps'])
        reference, _, _ = resample(reference, rt, solver.model.jnt_range[1:], end=np.nextafter(t[-1], np.inf))
        error = np.rad2deg(abs(corrected[:, 7:]-reference[:, 7:]))
        record = {**source[sid], 'hinge_mae_deg': float(error.mean()), 'per_joint_mae_deg': error.mean(axis=0).tolist(),
                  'output_sha256': sha(out/(mid+'.npy')), 'solver_sha256': sha(out/(mid+'.solver.json')),
                  'candidate_continuity': continuity(solver.model, corrected, t),
                  'correction_mae_deg': float(np.rad2deg(abs(corrected[:, 7:]-q[:, 7:])).mean()),
                  'optimizer_budget_frames': sum(not r['optimizer_success'] for r in infos),
                  'elapsed_sec': time.monotonic()-started}
        results.append(record)
        write_json(ROOT/'qa/aggregate'/f'{namespace}.json', {
            'status': 'DEVELOPMENT_SUBSET', 'fixed_panel_size': 200, 'actual_attempted': len(results),
            'run_config': {**design, 'panel': 'dev'}, 'results': results})
        print(sid, 'DONE', record['hinge_mae_deg'], record['elapsed_sec'], flush=True)
    pair_qa(namespace, 'dev', sample_ids=selected)
    summarize(namespace)


if __name__ == '__main__':
    run()
