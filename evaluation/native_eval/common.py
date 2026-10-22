from motion_mamba2.paths import artifact_path
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path(artifact_path('evaluator_native_v2_cc741fa946'))
CODE = Path(__file__).resolve().parents[1]
PROJECT = Path(artifact_path('mamba-transformer-keneral_64d2519211'))
UPSTREAM = Path(artifact_path('g1_fusion_20260915_e24ce84db0'))
RELEASE = UPSTREAM / 'releases/kinematic_training_20260917_v1'
GENERATOR = Path(artifact_path('g1_startpoint_and_learning_execution_20261004_d886325d67'))
INDEX = Path(artifact_path('data_index_jsonl_9966c1bc60'))
TEXT = Path(artifact_path('t5-3b_6e94c7579f'))
GPU_UUID = os.environ.get('EVALUATOR_V2_GPU_UUID', '')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = output_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)

def rows(path):
    with open(path) as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def jsonl(path, values):
    path = output_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        for r in values: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)

def ledger(stage, **values):
    p = ROOT / 'EXPERIMENT_LEDGER.json'
    history = read(p) if p.exists() else []
    history.append(dict(time=time.time(), stage=stage, **values)); write(p, history)

def status(phase, **values):
    write(ROOT/'STATUS.json', dict(phase=phase, updated_at=time.time(), **values))

def resource_guard():
    for path, gib in [('/data3',150),('/data4',100)]:
        if shutil.disk_usage(path).free < gib*2**30:
            raise RuntimeError(f'{path} below reserved {gib} GiB; stop evaluator only')
    # Files are small enough for this bounded, evaluator-only traversal.
    used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if used > 50*2**30: raise RuntimeError('Evaluator v2 disk budget exceeded')
    p=ROOT/'GPU_BUDGET.json'
    if p.exists():
        b=read(p); elapsed=b['seconds']
        if b.get('active_since'): elapsed += time.time()-b['active_since']
        lease=ROOT/'GPU_LEASE.json'
        if lease.exists():
            l=read(lease)
            if not l.get('released_at'):elapsed=max(elapsed,l.get('prior_charged_seconds',0)+time.time()-l['acquired_at'])
        if elapsed >= 24*3600: raise RuntimeError('24 GPU-hour v2 budget exhausted')

@contextlib.contextmanager
def gpu_budget(stage):
    p=ROOT/'GPU_BUDGET.json'; b=read(p) if p.exists() else dict(seconds=0, stages=[])
    if b.get('active_since'):
        b['seconds'] += time.time()-b['active_since']  # Conservative crash charge.
    start=time.time(); b.update(active_since=start,active_stage=stage); write(p,b)
    resource_guard()
    try: yield
    finally:
        b['seconds'] += time.time()-start
        b['stages'].append(dict(stage=stage,seconds=time.time()-start))
        b.update(active_since=None,active_stage=None); write(p,b)

def snapshot_code():
    return {str(p.relative_to(CODE)):sha(p) for p in sorted(CODE.rglob('*'))
            if p.is_file() and p.suffix in ['.py','.json','.sh','.md'] and '.venv' not in p.parts}

def output_path(path):
    path=Path(path).absolute()
    resolved=path.resolve()
    if not any(resolved.is_relative_to(root) for root in (ROOT,CODE)):
        raise ValueError('v2 writes must remain within v2 code/results roots: '+str(path))
    return path
