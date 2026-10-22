"""Commit only a tested execution change, then inherit the saved live state."""
from motion_mamba2.paths import artifact_path
import os,sys,time,signal,subprocess,hashlib,traceback
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'codec_graph_accel_20261007'
sys.path.insert(0,str(R));from common import *
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pause=read(S/'PAUSE.json');controller=pause['controller_pid'];worker=R/'worker.py';original=worker.read_text()
committed=False;process=None
try:
    bench=read(S/'BENCHMARK.json');gate=read(S/'UPDATE_GATE.json')
    assert gate['passed'] and bench['mean_speedup']>1.05
    old=read(S/'BOUNDARY.json');state=read(R/'train/STATE.json');assert state['checkpoint']==old['checkpoint']
    assert Path(f'/proc/{controller}/cwd').resolve()==R
    assert b'admit64_final.py' in Path(f'/proc/{controller}/cmdline').read_bytes()
    write(S/'ENABLED.json',dict(version='frozen_serial_codec_graph_v2',mode=bench['selected']['mode'],workers=bench['selected']['workers'],
        source_sha256=digest(S/'pipeline_accel.py'),codec_source_sha256=digest(S/'codec_graph.py'),gate=str(S/'UPDATE_GATE.json'),input_gate=str(S/'INPUT_GATE.json'),
        benchmark=str(S/'BENCHMARK.json'),resume_checkpoint=state['checkpoint'],resume_stage_step=state['step'],
        global_batch=512,microbatch=64,loss_sampler_noise_optimizer_scheduler_unchanged=True,convergence_equivalence_not_claimed=True))
    (S/'worker_before.py').write_text(original)

    assert original.count("execution_root=ROOT/'pipeline_accel_20261007'")==1
    updated=original.replace("execution_root=ROOT/'pipeline_accel_20261007'","execution_root=ROOT/'codec_graph_accel_20261007'")
    needle="    assert read(execution['gate'])['passed']"
    updated=updated.replace(needle,needle+"\n    assert hashlib.sha256((execution_root/'codec_graph.py').read_bytes()).hexdigest()==execution['codec_source_sha256']")
    needle2="    engine.update=pipeline_accel.install(execution['mode'],execution['workers'])"
    updated=updated.replace(needle2,"    assert pipeline_accel.VERSION==execution['version']\n    assert Path(pipeline_accel.__file__).resolve()==(execution_root/'pipeline_accel.py').resolve()\n"+needle2)
    worker.write_text(updated)

    write(S/'WORKER_CHANGE.json',dict(before_sha256=digest(S/'worker_before.py'),after_sha256=digest(worker),time=time.time(),no_model_source_modified=True))
    (R/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
    until=state['step']+12
    args=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(worker),'--until',str(until),'--resume',state['checkpoint']]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    env['MOTION_MICROBATCH_GATE']=str(R/'MICROBATCH_GATE.json')
    with (S/'live_first12.log').open('w') as out:
        process=subprocess.Popen(args,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        write(S/'PROBE_CHILD.json',dict(pid=process.pid,command=args,label='live_first12',time=time.time()))
        rc=process.wait(timeout=1200)
    assert rc==0,('Live accelerated worker failed',rc)
    actual=read(R/'train/STATE.json');assert actual['step']==until
    import json,statistics
    rows=[json.loads(x) for x in (R/'train/train.jsonl').read_text().splitlines()][-12:]
    assert all(x.get('execution_backend')=='frozen_serial_codec_graph_v2' for x in rows)
    mean=statistics.mean(x['seconds'] for x in rows[2:])
    write(S/'LIVE_RESULT.json',dict(passed=True,rows=rows,mean_seconds=mean,samples_per_second=512/mean,stage_step=until,
        global_step=75000+until,checkpoint=actual['checkpoint'],remaining_training_hours=(25000-until)*mean/3600,
        initial_saved_checkpoint=state['checkpoint'],optimizer_state_and_schedule_inherited=True))
    committed=True
    os.kill(controller,signal.SIGCONT)
    write(S/'RESUMED.json',dict(time=time.time(),controller_pid=controller,accelerated_backend_enabled=True,checkpoint=actual['checkpoint']))
    print('ENABLED',actual['step'],mean,flush=True)
except BaseException as error:
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=30)
    write(S/'ENABLE_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),committed=committed))
    if not committed:
        worker.write_text(original);(S/'ENABLED.json').unlink(missing_ok=True)
        # Keep discarded receipts and budget charges as evidence; resume only
        # from the complete state. Usually no discarded updates exist.
        import json
        current=read(R/'train/STATE.json');lines=(R/'train/train.jsonl').read_text().splitlines()
        if len(lines)>current['step']:
            (S/'discarded_live_receipts.jsonl').write_text('\n'.join(lines[current['step']:])+'\n')
            (R/'train/train.jsonl').write_text('\n'.join(lines[:current['step']])+'\n')
        (R/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
        os.kill(controller,signal.SIGCONT)
        write(S/'RESUMED.json',dict(time=time.time(),reference_controller_resumed=True,reason='enable_error'))
    raise
