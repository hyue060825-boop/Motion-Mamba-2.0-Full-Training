"""Pause only our coordinator at a committed boundary, evaluate, always resume."""
import ast,signal,subprocess,traceback,fcntl
from common import *
tree=ast.parse((ROOT/'controller.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[]),'controller_helpers','exec'),globals())
folder=ROOT/'evaluation/75000';folder.mkdir(parents=True,exist_ok=True)
lock=(folder/'immediate.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
coordinator=read(ROOT/'STATUS.json')['controller_pid'];paused=False;released=False
try:
    command=Path(f'/proc/{coordinator}/cmdline').read_bytes()
    assert b'admit64_final.py' in command or b'controller.py' in command
    assert Path(f'/proc/{coordinator}/cwd').resolve()==ROOT
    initial=read(ROOT/'train/STATE.json')['step']
    write(folder/'IMMEDIATE_STATUS.json',dict(stage='WAITING_OWN_SAVE_BOUNDARY',training_coordinator=coordinator,saved_stage_step=initial))
    os.kill(coordinator,signal.SIGSTOP);paused=True
    write(ROOT/'BOUNDARY_REQUEST.json',dict(reason='USER_REQUEST_MEASURE_EXACT75000_NOW',time=time.time()))
    deadline=time.time()+1800
    while True:
        state=read(ROOT/'train/STATE.json')
        if state['step']>initial and free('4,5,6,7'):break
        if time.time()>deadline:raise RuntimeError('No safe saved training boundary reached')
        time.sleep(15)
    released=True;publish(state)
    write(folder/'TRAINING_PAUSE.json',dict(saved_global_step=75000+state['step'],resume_checkpoint=state['checkpoint'],coordinator=coordinator,no_foreign_jobs_stopped=True,time=time.time()))
    write(ROOT/'STATUS.json',dict(stage='EVALUATING75K_TRAINING_PAUSED',controller_pid=coordinator,evaluation_driver_pid=os.getpid(),checkpoint=str(OLD/'train/checkpoint75000.pt'),saved_training_global_step=75000+state['step']))
    write(folder/'IMMEDIATE_STATUS.json',dict(stage='EVALUATING',checkpoint=str(OLD/'train/checkpoint75000.pt'),device='4'))
    rc,log=invoke('evaluation75000_requested_now',[ROOT/'evaluate_stage.py','--step','75000','--checkpoint',OLD/'train/checkpoint75000.pt'],devices='4',timeout=3600)
    assert rc==0,('75k evaluation failed',str(log),rc)
    rc,log=invoke('report75000_requested_now',[ROOT/'report75_now.py'],devices='',timeout=180)
    assert rc==0,('75k report failed',str(log),rc)
    write(folder/'IMMEDIATE_STATUS.json',dict(stage='COMPLETE',report=str(folder/'comparison/REPORT.md'),training_will_resume=True))
except BaseException as error:
    write(folder/'IMMEDIATE_ERROR.json',dict(error=repr(error),traceback=traceback.format_exc(),training_will_resume=paused))
    raise
finally:
    if paused:
        (ROOT/'BOUNDARY_REQUEST.json').unlink(missing_ok=True)
        os.kill(coordinator,signal.SIGCONT)
        write(folder/'TRAINING_RESUMED.json',dict(coordinator=coordinator,time=time.time(),saved_boundary_reused=released,per_rank_microbatch=64,global_batch=512))
        if released:write(ROOT/'STATUS.json',dict(stage='RESUMING_TRAINING_AFTER75K_EVALUATION',controller_pid=coordinator,global_batch=512,micro=64))
