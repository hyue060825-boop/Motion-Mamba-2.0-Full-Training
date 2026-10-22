"""Attach the already-admitted actual-mask backend to the HumanML3D timing entry."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,hashlib,time,shutil,subprocess,fcntl,importlib.util,traceback
R=Path(artifact_path('humanml3d_section1_20261020_c1c76adb30'))
O=R/'latency_masked_recheck'
OLD=Path(artifact_path('section1_masked_kernel_repair_20261020_47cd64f873'))
PY=artifact_path('python_9f8003379b')
MODELS=['native100','native110','native117','native122','native122long']
TASKS=['short52','aligned_w04']
def rd(p):return json.loads(p.read_text())
def wr(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,default=str)+'\n');os.replace(q,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def setup():
    O.mkdir(exist_ok=True)
    assert rd(OLD/'ADMISSION.json')['mode']=='masked'
    for model in MODELS:
        s=R/'models'/model;d=O/model;d.mkdir(exist_ok=True)
        assert rd(s/'COMPLETE.json')['passed']
        for p in s.glob('*.py'):shutil.copy2(p,d/p.name)
        shutil.copytree(s/'sampler',d/'sampler',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(s/'panels',d/'panels',dirs_exist_ok=True)
        for n in ['manifest.jsonl','text_cache','CHECKPOINTS.json','PREPARED.json','GATES.json','PLAN.json']:
            if not (d/n).exists():(d/n).symlink_to(s/n,target_is_directory=(s/n).is_dir())
        (d/'util.py').write_text((s/'util.py').read_text()+f'\nROOT=Path({str(d)!r})\n')
    wr(O/'PLAN.json',dict(existing_admitted_backend=str(OLD/'masked_graph_candidate.py'),backend_sha256=sha(OLD/'masked_graph_candidate.py'),
        input_cohort_sha256=rd(R/'COHORT_FROZEN.json')['cohort_sha256'],models=MODELS,tasks=TASKS,
        no_new_training_or_kernel=True,weights_goals_noise_and_quality_results_unchanged=True,
        gates='full output/geometry/final and per-window state/RNG bitwise; dynamic-mask replay; all64 B1 outputs bitwise',
        timing='same32 prespecified sources,2 whole warmups,10steps CFG2/2; exclude graph creation, require zero new graphs and counted fallbacks',time=time.time()))
def worker(model):
    d=O/model;sys.path[:0]=[str(d),str(R)]
    os.environ.update(EVAL_MODEL=model,EVAL_LABEL=model)
    import evaluate_suite as suite
    harness=suite.install(d)
    sp=importlib.util.spec_from_file_location('admitted_mask_driver',OLD/'driver.py');repair=importlib.util.module_from_spec(sp);sp.loader.exec_module(repair)
    repair.install(harness)
    try:
        import numpy as np,torch,gc
        gates=[]
        for task in TASKS:
            os.environ['REPAIR_MODE']='old'
            a=suite.run(d,model,task,0,'gate_mask_old','optimized',pilot=True)
            os.environ['REPAIR_MODE']='masked'
            b=suite.run(d,model,task,0,'gate_mask_fixed','optimized',pilot=True)
            c=repair.gate_pair(a,b);assert c['passed'],(model,task,c)
            gates.append(dict(task=task,**c))
        client=harness.CACHE[1];client.controller._repair_graph_caches.clear()
        for block in [client.controller.root_block,client.controller.body_block]:
            if hasattr(block.sequence_stack,'_inference_graphs'):del block.sequence_stack._inference_graphs
        gc.collect();torch.cuda.empty_cache()
        wr(d/'DYNAMIC_MASK_GATE.json',repair.dynamic_mask_gate(client))
        rows=[]
        for task in TASKS:
            os.environ['REPAIR_MODE']='masked'
            new=suite.run(d,model,task,0,'timing','optimized')
            old=R/'models'/model/'timing'/model/task/'shard00'
            with np.load(new/'geometry.npz') as x,np.load(old/'geometry.npz') as y:
                checks={k:bool(np.array_equal(x[k],y[k])) for k in x.files};assert all(checks.values()),(model,task,checks)
            times=[x for x in rd(new/'TIMING_DETAIL.json')['rows'] if not x['warmup']]
            assert len(times)==32 and all(t['batch']==1 and t['graph_delta']['builds']==t['graph_delta']['fallbacks']==0 for t in times)
            W=rd(d/'PLAN.json')['tasks'][task]['goal_window'];seconds=[x['elapsed_seconds'] for x in times]
            rows.append(dict(task=task,N=32,windows=W,ms_per_window=1000*float(np.median(seconds))/W,
                graph_replays=sum(x['graph_delta']['replays'] for x in times),new_timed_graphs=0,counted_fallbacks=0,
                all_geometry_and_outputs_bitwise_exact=True,old_geometry_sha256=sha(old/'geometry.npz'),new_geometry_sha256=sha(new/'geometry.npz')))
        wr(d/'COMPLETE.json',dict(passed=True,model=model,gates=gates,timings=rows,time=time.time()))
    except BaseException:
        wr(d/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
def controller():
    O.mkdir(exist_ok=True);lock=(O/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    wr(O/'CONTROLLER_PID.json',dict(pid=os.getpid(),cwd=str(O),time=time.time()))
    handles=[];jobs={};queue=MODELS[:]
    try:
        import evaluate_suite
        assert evaluate_suite.free_cards()==[0,1,2,3],'Must wait for genuine idle evaluation GPUs'
        for card in range(4):
            for root in [R,evaluate_suite.B3,Path(artifact_path('g1_external_matched100k_batch2_20261019_068631a7db')),Path(artifact_path('g1_external_matched100k_20261019_f77433d79c'))]:
                f=(root/f'gpu{card}.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
        while queue or jobs:
            for card in range(4):
                if card in jobs or not queue:continue
                m=queue.pop(0);d=O/m;env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(card),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                with (d/'worker.log').open('a') as log:p=subprocess.Popen([PY,__file__,'worker',m],cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
                jobs[card]=(m,p)
            for card,(m,p) in list(jobs.items()):
                if p.poll() is None:continue
                assert p.returncode==0,(m,p.returncode);del jobs[card]
            wr(O/'STATUS.json',dict(stage='CHECKING_MASKED_PATH_AND_TIMING',active={c:m for c,(m,p) in jobs.items()},queued=queue,time=time.time()))
            if queue or jobs:time.sleep(5)
        proofs={m:rd(O/m/'COMPLETE.json') for m in MODELS};assert all(p['passed'] for p in proofs.values())
        archive=R/'maintenance/20261021_unattached_mask_backend';archive.mkdir(parents=True,exist_ok=True)
        for m in MODELS:
            old=R/'models'/m/'timing';dest=archive/m/'timing';dest.parent.mkdir(exist_ok=True)
            assert not dest.exists(),'Do not overwrite archived timings';shutil.move(old,dest)
            shutil.copytree(O/m/'timing',old)
        wr(R/'EXECUTION_REPAIR_LATENCY.json',dict(passed=True,reason='HumanML3D entry cloned pre-mask-repair runner: optimized label alone left masked read-only calls outside actual-mask Graph coverage',
            fix='Attach same previously admitted mask-aware backend; no suffix cropping or condition removal',existing_admission=str(OLD/'ADMISSION.json'),
            backend_sha256=sha(OLD/'masked_graph_candidate.py'),full_new_cohort_gate=True,models=proofs,
            weights_training_sampling_quality_unchanged=True,old_timings_preserved=str(archive),time=time.time()))
        wr(O/'COMPLETE.json',dict(passed=True,models=MODELS,time=time.time()))
    except BaseException:
        wr(O/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
    finally:
        for m,p in jobs.values():p.wait()
        for f in handles:f.close()
if __name__=='__main__':
    if sys.argv[1]=='setup':setup()
    elif sys.argv[1]=='worker':worker(sys.argv[2])
    else:controller()
