"""Execute the original, hash-bound serial clip body in a fresh worker process.

AST extraction removes only the serial resume branch and shared ledger writes.
The computation is compiled directly from the original module, not reimplemented.
"""
import ast
import importlib
import time
from .common import ROOT, sha


def clip_code(module):
    tree = ast.parse((ROOT / 'g1_pipeline' / (module + '.py')).read_text())
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'row'
             and ast.unparse(n.iter) == 'rows[:max_clips]']
    if len(loops) != 1:
        raise ValueError('SERIAL_CLIP_BODY_NOT_UNIQUE')
    body = []
    for node in loops[0].body:
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith('resume and '):
            continue
        # Everything through previous[sid] = entry is the original per-clip path.
        body.append(node)
        if isinstance(node, ast.Assign) and any(ast.unparse(t) == 'previous[sid]' for t in node.targets):
            break
    else:
        raise ValueError('SERIAL_CLIP_COMMIT_BOUNDARY_NOT_FOUND')
    return compile(ast.Module(body=body, type_ignores=[]), str(ROOT/'g1_pipeline'/(module+'.py')), 'exec')


def execute(task, out, asset, expected_code):
    module = 'production_hml' if task['source'] == 'humanml3d' else 'production_hiphi'
    for name, bound in expected_code.items():
        if sha(ROOT/'g1_pipeline'/(name+'.py')) != bound:
            raise ValueError('SERIAL_CODE_CHANGED:' + name)
    original = importlib.import_module('g1_pipeline.' + module)
    from . import source_motion_processing as processing
    from pathlib import Path
    import torch
    # OMP_NUM_THREADS already constrains intra-op; the exporter has no parallel
    # inter-op graph. Record actual pools, rather than inventing more settings.
    timings = {}
    def timed(owner, name, label):
        fn = getattr(owner, name)
        def wrapped(*a, **kw):
            start = time.monotonic()
            try:
                return fn(*a, **kw)
            finally:
                timings[label] = timings.get(label, 0.) + time.monotonic() - start
        setattr(owner, name, wrapped)
    xml = Path(asset['path'])
    if sha(xml) != asset['sha256']:
        raise ValueError('ISOLATED_ASSET_CHANGED')
    # Coordinator ran the unchanged asset equivalence check once. This replaces
    # only redundant writes to a shared XML/diagnostic; solver construction stays.
    processing.prepare = lambda: (xml, asset['report'])
    for name, label in [('solve','ik'), ('refine','collision_refinement'),
                        ('correct','ground_correction'), ('project','rate_projection'),
                        ('replay','geometry_qa'), ('continuity','continuity_qa'),
                        ('series','relative_qa'), ('target_errors','target_qa')]:
        timed(processing, name, label)
    for name in (['native_parameters', 'native_pose_to_frames', 'time_grid']
                 if task['source']=='humanml3d' else ['frames_at_25hz', 'read_bvh']):
        timed(original, name, 'parse_fk_resample')
    timed(original, 'export_and_check', 'export_representation_roundtrip_qa')
    timed(original, 'local_rotation_spectrum', 'spectrum_qa')
    env = vars(original).copy()
    sid = task['row']['sample_id']
    env.update(row=task['row'], mappings={sid:task['mapping']},
               packages={sid:task.get('package')}, packagepath=ROOT/'audit/hiphi_package_evidence_v1.json',
               out=out, previous={}, pilot=True, resume=False)
    exec(clip_code(module), env)
    return env['entry'], {'phase_seconds':timings,
                         'torch_intraop_threads':torch.get_num_threads(),
                         'torch_interop_threads':torch.get_num_interop_threads()}
