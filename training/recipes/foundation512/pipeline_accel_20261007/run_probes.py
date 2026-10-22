from motion_mamba2.paths import artifact_path
import os,sys,json,time,subprocess,signal,traceback,math
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'pipeline_accel_20261007'
sys.path.insert(0,str(R));from common import *
deadline=time.time()+1200
while not (S/'BOUNDARY.json').exists():
    if time.time()>deadline:raise TimeoutError('Saved boundary was not ready')
    time.sleep(5)
write(S/'budget/LIMITS.json',dict(caps=dict(checks=200,B=0,C=0,teacher=0),total_updates=200,gpu_hours=8))
def launch(label,script,args=(),ranks=1):
    devices='4' if ranks==1 else '4,5,6,7'
    cmd=[PYTHON]+(['-m','torch.distributed.run','--standalone',f'--nproc_per_node={ranks}'] if ranks>1 else [])+[str(S/script)]+list(args)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    with (S/(label+'.log')).open('w') as out:
        p=subprocess.Popen(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        write(S/'PROBE_CHILD.json',dict(pid=p.pid,command=cmd,label=label,time=time.time()))
        try:rc=p.wait(timeout=1500)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);raise
    with (S/'budget/GPU_TIME.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,seconds=time.time()-read(S/'PROBE_CHILD.json')['time'],devices=ranks))+'\n')
    if rc:raise RuntimeError((label,rc,(S/(label+'.log')).read_text()[-3000:]))
def compare_tensor_dict(left,right):
    import torch
    torch.set_num_threads(2)
    x=torch.load(left,map_location='cpu',mmap=True,weights_only=False);y=torch.load(right,map_location='cpu',mmap=True,weights_only=False)
    def flatten(v,prefix=''):
        out={}
        for k,t in v.items():
            if isinstance(t,dict):out.update(flatten(t,prefix+str(k)+'.'))
            else:out[prefix+str(k)]=t
        return out
    x=flatten(x);y=flatten(y);assert x.keys()==y.keys();groups={};dot=aa=bb=ee=0
    for k,a in x.items():
        a=a.flatten().double();b=y[k].flatten().double();assert torch.isfinite(b).all()
        d=float(a@b);u=float(a@a);v=float(b@b);e=float((a-b).square().sum())
        dot+=d;aa+=u;bb+=v;ee+=e
        g='root' if k.startswith('root_model.') else 'body' if k.startswith('body_model.') else 'other'
        vals=groups.setdefault(g,[0.,0.,0.,0.]);vals[0]+=d;vals[1]+=u;vals[2]+=v;vals[3]+=e
    def metrics(d,u,v,e):return dict(cosine=d/math.sqrt(max(u*v,1e-60)),relative_l2=math.sqrt(e/max(u,1e-60)))
    return dict(total=metrics(dot,aa,bb,ee),groups={k:metrics(*v) for k,v in groups.items()})
try:
    launch('correctness','correctness.py')
    gate=read(S/'INPUT_LOSS_GATE.json');assert gate['complete']
    assert all(r['loss_gradient_pass'] for r in gate['rows']),'Batched decoder/loss does not preserve objective gradient'
    mode='batch_all' if gate['passed'] else 'loss_only'
    write(S/'SELECTED_MODE.json',dict(mode=mode,reason='exact all input tensors across six layouts' if mode=='batch_all' else 'retain serial encoder because batch target equality did not pass'))
    launch('reference_4r','probe_pipeline.py',['--mode','reference','--label','reference_4r','--save-gate'],4)
    launch('candidate_4r','probe_pipeline.py',['--mode',mode,'--workers','4','--label','candidate_4r','--save-gate'],4)
    ref=S/'probes/reference_4r';cand=S/'probes/candidate_4r'
    g=compare_tensor_dict(ref/'gradients.pt',cand/'gradients.pt')
    u=compare_tensor_dict(ref/'master_delta.pt',cand/'master_delta.pt')
    moments=compare_tensor_dict(ref/'moments.pt',cand/'moments.pt')
    a=read(ref/'RESULT.json');b=read(cand/'RESULT.json');ra=read(ref/'RESTORED.json');rb=read(cand/'RESTORED.json')
    eq=dict(gradient=g,actual_adam_master_delta=u,adam_moments=moments,starting_states_equal=ra==rb,
        loss_relative_difference=abs(a['rows'][0]['loss']-b['rows'][0]['loss'])/max(abs(a['rows'][0]['loss']),1e-8),
        same_samples=all(x['sampler_identity']==y['sampler_identity'] for x,y in zip(a['rows'],b['rows'])))
    eq['passed']=eq['starting_states_equal'] and eq['same_samples'] and eq['loss_relative_difference']<.001 and all(v['cosine']>.999 and v['relative_l2']<.05 for v in [g['total'],u['total'],*u['groups'].values()])
    write(S/'UPDATE_GATE.json',eq);assert eq['passed'],'Actual optimizer update gate failed'
    candidates=[b]
    if mode=='batch_all':
        for workers in [0,2,8]:
            label=f'candidate_workers{workers}_4r'
            launch(label,'probe_pipeline.py',['--mode',mode,'--workers',str(workers),'--label',label,'--count','12'],4)
            candidates.append(read(S/'probes'/label/'RESULT.json'))
    best=min(candidates,key=lambda x:x['mean_seconds'])
    launch('confirm_candidate_4r','probe_pipeline.py',['--mode',mode,'--workers',str(best['workers']),'--label','confirm_candidate_4r'],4)
    launch('confirm_reference_4r','probe_pipeline.py',['--mode','reference','--label','confirm_reference_4r'],4)
    ca=read(S/'probes/confirm_candidate_4r/RESULT.json');ra=read(S/'probes/confirm_reference_4r/RESULT.json')
    write(S/'BENCHMARK.json',dict(reference=a,candidates=candidates,selected=dict(mode=mode,workers=best['workers']),
        confirm_reference=ra,confirm_candidate=ca,
        mean_speedup=ra['mean_seconds']/ca['mean_seconds'],time_reduction_fraction=1-ca['mean_seconds']/ra['mean_seconds'],
        sample_seconds=ca['mean_seconds'],all_probe_updates_discarded=True,gate_passed=True))
    print('PROBES_COMPLETE',ra['mean_seconds'],ca['mean_seconds'],flush=True)
except BaseException as error:
    write(S/'ERROR.json',dict(error=repr(error),traceback=traceback.format_exc()))
    pause=read(S/'PAUSE.json');(R/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
    try:os.kill(pause['controller_pid'],signal.SIGCONT)
    except ProcessLookupError:pass
    write(S/'RESUMED.json',dict(time=time.time(),reference_controller_resumed=True,reason='probe_error'))
    raise
