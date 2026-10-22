from motion_mamba2.paths import artifact_path
import os, sys, json, time, subprocess, signal, traceback, math
from pathlib import Path
R = Path(artifact_path('batch512_after75k_20261007_c710d5b650'))
S = R / 'codec_graph_accel_20261007'
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
    deadline=time.time()+1200
    while not (S/'BOUNDARY.json').exists():
        if time.time()>deadline:raise TimeoutError('Saved boundary was not ready')
        time.sleep(5)
    write(S/'budget/LIMITS.json',dict(caps=dict(checks=100,B=0,C=0,teacher=0),total_updates=100,gpu_hours=4))
    launch('input_gate','input_gate.py')
    assert read(S/'INPUT_GATE.json')['passed'], 'Codec graph changed input/target values'
    launch('reference_4r','probe_pipeline.py',['--mode','reference','--workers','0','--label','reference_4r','--save-gate'],4)
    launch('candidate_4r','probe_pipeline.py',['--mode','serial_codec','--workers','0','--label','candidate_4r','--save-gate'],4)
    ref=S/'probes/reference_4r';cand=S/'probes/candidate_4r'
    g=compare_tensor_dict(ref/'gradients.pt',cand/'gradients.pt')
    u=compare_tensor_dict(ref/'master_delta.pt',cand/'master_delta.pt')
    mom=compare_tensor_dict(ref/'moments.pt',cand/'moments.pt')
    a=read(ref/'RESULT.json');b=read(cand/'RESULT.json')
    eq=dict(gradient=g,actual_adam_master_delta=u,adam_moments=mom,
        starting_states_equal=read(ref/'RESTORED.json')==read(cand/'RESTORED.json'),
        same_samples=all(x['sampler_identity']==y['sampler_identity'] for x,y in zip(a['rows'],b['rows'])),
        maximum_loss_relative_difference=max(abs(x['loss']-y['loss'])/max(abs(x['loss']),1e-8) for x,y in zip(a['rows'],b['rows'])))
    eq['passed']=eq['starting_states_equal'] and eq['same_samples'] and eq['maximum_loss_relative_difference']<.001 and all(v['cosine']>.999 and v['relative_l2']<.005 for v in [g['total'],u['total'],*u['groups'].values(),mom['total']])
    write(S/'UPDATE_GATE.json',eq);assert eq['passed'], 'Actual update equivalence gate failed'
    launch('confirm_candidate_4r','probe_pipeline.py',['--mode','serial_codec','--workers','0','--label','confirm_candidate_4r'],4)
    launch('confirm_reference_4r','probe_pipeline.py',['--mode','reference','--workers','0','--label','confirm_reference_4r'],4)
    ca=read(S/'probes/confirm_candidate_4r/RESULT.json');ra=read(S/'probes/confirm_reference_4r/RESULT.json')
    result=dict(reference=a,candidate=b,confirm_reference=ra,confirm_candidate=ca,
        mean_speedup=ra['mean_seconds']/ca['mean_seconds'],
        time_reduction_fraction=1-ca['mean_seconds']/ra['mean_seconds'],
        selected=dict(mode='serial_codec',workers=0),all_probe_updates_discarded=True,
        gate_passed=True,baseline='already_accelerated_pipeline_v1')
    write(S/'BENCHMARK.json',result)
    print('COMPLETE',result['mean_speedup'],ca['mean_seconds'],flush=True)
except BaseException as e:
    write(S/'ERROR.json',dict(error=repr(e),traceback=traceback.format_exc()))
    pause=read(S/'PAUSE.json');(R/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
    try:os.kill(pause['controller_pid'],signal.SIGCONT)
    except ProcessLookupError:pass
    write(S/'RESUMED.json',dict(time=time.time(),previous_accelerated_controller_resumed=True,reason='probe_error'))
    raise
