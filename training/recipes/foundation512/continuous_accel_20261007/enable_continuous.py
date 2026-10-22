"""Admit bounded, tested execution changes and resume the owned controller."""
from motion_mamba2.paths import artifact_path
import os, sys, signal, subprocess, hashlib, traceback, time, statistics
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'))
S=R/'continuous_accel_20261007'
sys.path[:0]=[str(R),str(S)]
from common import *
from continuous_runtime import make_train, VERSION
from motion_training import engine

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

controller=read(S/'PAUSE.json')['controller_pid']
worker=R/'worker.py';original=worker.read_text();process=None;committed=False
try:
    gate=read(S/'UPDATE_GATE.json');bench=read(S/'BENCHMARK.json');continuity=read(S/'CONTINUITY_GATE.json')
    assert gate['passed'] and continuity['passed']
    assert Path(f'/proc/{controller}/cwd').resolve()==R
    assert b'admit64_final.py' in Path(f'/proc/{controller}/cmdline').read_bytes()
    state=read(R/'train/STATE.json');assert state['checkpoint']==read(S/'BOUNDARY.json')['checkpoint']
    audit=make_train(engine)
    # Exhaustively check the additional save rule over this bounded training run.
    cfg=read(R/'config.json')
    periodic=[i for i in range(state['step']+1,cfg['max_updates']+1)
              if i in cfg['savepoints'] or i==cfg['max_updates'] or i%1000==0]
    assert all(i in periodic for i in range(((state['step']//1000)+1)*1000,cfg['max_updates']+1,1000))
    write(S/'STRUCTURAL_GATE.json',dict(passed=True,periodic_savepoints=periodic,**audit))
    (S/'generated_train.py').write_text(audit['generated_source']+'\n')
    enabled=dict(version=VERSION,optimizer_copy_enabled=continuity['optimizer_copy_enabled'],
        original_train_sha256=audit['original_train_sha256'],transformed_ast_sha256=audit['transformed_ast_sha256'],
        runtime_source_sha256=digest(S/'continuous_runtime.py'),optimizer_source_sha256=digest(S/'optimizer_copy.py'),
        gate=str(S/'UPDATE_GATE.json'),continuity_gate=str(S/'CONTINUITY_GATE.json'),
        structural_gate=str(S/'STRUCTURAL_GATE.json'),benchmark=str(S/'BENCHMARK.json'),
        controller_pid=controller,resume_checkpoint=state['checkpoint'],resume_stage_step=state['step'],
        global_batch=512,microbatch=64,save_every_updates=1000,automatic_world_switch_preserved=True,
        loss_inputs_noise_optimizer_schedule_sampler_precision_unchanged=True)
    write(S/'ENABLED.json',enabled)
    (S/'worker_before.py').write_text(original)
    needle="engine.train(str(ROOT/'config.json'),str(ROOT/'train'),until=a.until,resume=a.resume,category='B')"
    assert original.count(needle)==1
    replacement="""run_until=a.until
continuous_root=ROOT/'continuous_accel_20261007'
if (continuous_root/'ENABLED.json').exists():
    continuous=read(continuous_root/'ENABLED.json')
    import hashlib,os
    assert hashlib.sha256((continuous_root/'continuous_runtime.py').read_bytes()).hexdigest()==continuous['runtime_source_sha256']
    assert hashlib.sha256((continuous_root/'optimizer_copy.py').read_bytes()).hexdigest()==continuous['optimizer_source_sha256']
    assert all(read(continuous[k])['passed'] for k in ['gate','continuity_gate','structural_gate'])
    sys.path.insert(0,str(continuous_root))
    import continuous_runtime
    assert continuous_runtime.VERSION==continuous['version']
    assert Path(continuous_runtime.__file__).resolve()==(continuous_root/'continuous_runtime.py').resolve()
    continuous_runtime.install(engine,ROOT,continuous)
    if os.environ.get('MOTION_CONTINUOUS_BOUNDED_GATE')!='1':run_until=cfg['max_updates']
engine.train(str(ROOT/'config.json'),str(ROOT/'train'),until=run_until,resume=a.resume,category='B')"""
    worker.write_text(original.replace(needle,replacement))
    write(S/'WORKER_CHANGE.json',dict(before_sha256=digest(S/'worker_before.py'),after_sha256=digest(worker),
          no_model_recipe_snapshot_change=True,time=time.time()))
    (R/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
    until=state['step']+12
    args=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(worker),
          '--until',str(until),'--resume',state['checkpoint']]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             OPENBLAS_NUM_THREADS='1',MOTION_CONTINUOUS_BOUNDED_GATE='1',
             MOTION_MICROBATCH_GATE=str(R/'MICROBATCH_GATE.json'))
    with (S/'live_first12.log').open('w') as out:
        process=subprocess.Popen(args,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        write(S/'PROBE_CHILD.json',dict(pid=process.pid,command=args,label='live_first12',time=time.time()))
        rc=process.wait(timeout=1200)
    assert rc==0,('Live worker failed',rc)
    actual=read(R/'train/STATE.json');assert actual['step']==until
    rows=[json.loads(x) for x in (R/'train/train.jsonl').read_text().splitlines()][-12:]
    assert all(x['execution_backend']=='frozen_serial_codec_graph_v2' and x['batch_size']==512 for x in rows)
    mean=statistics.mean(x['seconds'] for x in rows[2:])
    write(S/'LIVE_RESULT.json',dict(passed=True,rows=rows,mean_seconds=mean,stage_step=until,
        global_step=75000+until,checkpoint=actual['checkpoint'],samples_per_second=512/mean,
        remaining_training_hours=(25000-until)*mean/3600,optimizer_copy_enabled=enabled['optimizer_copy_enabled']))
    committed=True
    os.kill(controller,signal.SIGCONT)
    write(S/'RESUMED.json',dict(time=time.time(),controller_pid=controller,continuous_execution_enabled=True,
                              checkpoint=actual['checkpoint']))
    print('ENABLED',actual['step'],mean,flush=True)
except BaseException as e:
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=30)
    write(S/'ENABLE_ERROR.json',dict(error=repr(e),traceback=traceback.format_exc(),committed=committed))
    if not committed:
        worker.write_text(original);(S/'ENABLED.json').unlink(missing_ok=True)
        current=read(R/'train/STATE.json');lines=(R/'train/train.jsonl').read_text().splitlines()
        if len(lines)>current['step']:
            (S/'discarded_live_receipts.jsonl').write_text('\n'.join(lines[current['step']:])+'\n')
            (R/'train/train.jsonl').write_text('\n'.join(lines[:current['step']])+'\n')
        (R/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
        os.kill(controller,signal.SIGCONT)
        write(S/'RESUMED.json',dict(time=time.time(),reference_controller_resumed=True,reason='enable_error'))
    raise
