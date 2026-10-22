from motion_mamba2.paths import artifact_path
import os,signal,time,json,subprocess
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'))
S=R/'continuous_accel_20261007';S.mkdir(exist_ok=True)
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
status=json.loads((R/'STATUS.json').read_text());pid=status['controller_pid']
assert b'admit64_final.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
assert Path(f'/proc/{pid}/cwd').resolve()==R
previous=json.loads((R/'train/STATE.json').read_text())['step']
write(S/'PAUSE.json',dict(controller_pid=pid,old_stage_step=previous,time=time.time(),deadline=time.time()+5400))
os.kill(pid,signal.SIGSTOP)
write(R/'BOUNDARY_REQUEST.json',dict(reason='USER_AUTHORIZED_PIPELINE_ACCEL_PROBE',time=time.time()))
# Independent safety process resumes the reference controller after the deadline.
with (S/'watchdog.log').open('w') as log:
    p=subprocess.Popen([artifact_path('python_9f8003379b'),str(S/'watchdog.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(S/'WATCHDOG.json',dict(pid=p.pid))
try:
    deadline=time.time()+1200
    while time.time()<deadline:
        state=json.loads((R/'train/STATE.json').read_text())
        out=subprocess.check_output(['nvidia-smi','-i','4,5,6,7','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        if state['step']>previous and all(int(x)<512 for x in out.splitlines()):
            write(S/'BOUNDARY.json',dict(state,controller_pid=pid,complete_optimizer_boundary=True))
            print(json.dumps(state),flush=True);break
        time.sleep(5)
    else:raise TimeoutError('Boundary checkpoint did not finish')
except BaseException:
    (R/'BOUNDARY_REQUEST.json').unlink(missing_ok=True);os.kill(pid,signal.SIGCONT);raise
