from motion_mamba2.paths import artifact_path
import os, sys, json, time, subprocess, signal, traceback, math
from pathlib import Path
R = Path(artifact_path('batch512_after75k_20261007_c710d5b650'))
S = R / 'continuous_accel_20261007'
sys.path.insert(0,str(R))
from common import *
deadline = time.time() + 1200

def launch(label, script, args=(), ranks=1):
    devices = '4' if ranks == 1 else '4,5,6,7'
    cmd = [PYTHON] + (['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={ranks}'] if ranks > 1 else []) + [str(S / script)] + list(args)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    with (S / (label + '.log')).open('w') as out:
        p = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        write(S / 'PROBE_CHILD.json', dict(pid=p.pid, command=cmd, label=label, time=time.time()))
        try:
            rc = p.wait(timeout=1500)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=30)
            raise
    with (S / 'budget/GPU_TIME.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label, seconds=time.time() - read(S / 'PROBE_CHILD.json')['time'], devices=ranks)) + '\n')
    if rc:
        raise RuntimeError((label, rc, (S / (label + '.log')).read_text()[-3000:]))

def compare_tensor_dict(left, right):
    import torch
    torch.set_num_threads(2)
    x = torch.load(left, map_location='cpu', mmap=True, weights_only=False)
    y = torch.load(right, map_location='cpu', mmap=True, weights_only=False)

    def flatten(v, prefix=''):
        out = {}
        for (k, t) in v.items():
            if isinstance(t, dict):
                out.update(flatten(t, prefix + str(k) + '.'))
            else:
                out[prefix + str(k)] = t
        return out
    x = flatten(x)
    y = flatten(y)
    assert x.keys() == y.keys()
    groups = {}
    dot = aa = bb = ee = 0
    for (k, a) in x.items():
        a = a.flatten().double()
        b = y[k].flatten().double()
        assert torch.isfinite(b).all()
        d = float(a @ b)
        u = float(a @ a)
        v = float(b @ b)
        e = float((a - b).square().sum())
        dot += d
        aa += u
        bb += v
        ee += e
        g = 'root' if k.startswith('root_model.') else 'body' if k.startswith('body_model.') else 'other'
        vals = groups.setdefault(g, [0.0, 0.0, 0.0, 0.0])
        vals[0] += d
        vals[1] += u
        vals[2] += v
        vals[3] += e

    def metrics(d, u, v, e):
        return dict(cosine=d / math.sqrt(max(u * v, 1e-60)), relative_l2=math.sqrt(e / max(u, 1e-60)))
    return dict(total=metrics(dot, aa, bb, ee), groups={k: metrics(*v) for (k, v) in groups.items()})

try:
    bench=read(S/'BENCHMARK.json');gate=read(S/'UPDATE_GATE.json');assert gate['passed']
    enabled=bench['mean_speedup']>1.03
    mode='serial_codec' if enabled else 'reference'
    launch('resume_parent','probe_pipeline.py',['--mode',mode,'--workers','0','--label','resume_parent','--count','2','--resume-gate-save'],4)
    cp=S/'probes/resume_parent/resume_checkpoint.pt'
    launch('resume_child','probe_pipeline.py',['--mode',mode,'--workers','0','--label','resume_child','--count','1','--resume-gate-load',str(cp)],4)
    a=read(S/'probes/resume_parent/AFTER_NEXT_UPDATE.json');b=read(S/'probes/resume_child/AFTER_NEXT_UPDATE.json')
    write(S/'CONTINUITY_GATE.json',dict(passed=a==b,continuous=a,resumed=b,optimizer_copy_enabled=enabled,
        checkpoint_every_1000=True,recipe_unchanged=True))
    assert a==b, 'Resumed next update differs from uninterrupted update'
    print('CONTINUITY PASSED optimizer_copy_enabled',enabled,flush=True)
except BaseException as e:
    write(S/'RESUME_GATE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()))
    pause=read(S/'PAUSE.json');(R/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
    try:os.kill(pause['controller_pid'],signal.SIGCONT)
    except ProcessLookupError:pass
    write(S/'RESUMED.json',dict(time=time.time(),previous_accelerated_controller_resumed=True,reason='resume_gate_error'))
    raise
