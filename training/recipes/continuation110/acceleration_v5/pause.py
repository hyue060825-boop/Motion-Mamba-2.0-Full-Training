from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,json,signal,time,subprocess
r=Path(artifact_path('g1_native112k_future_interaction_20261011_d06c37e3c2'))
controller=json.loads((r/'acceleration_v4/RESUMPTION.json').read_text())['pid']
cmd=Path(f'/proc/{controller}/cmdline').read_bytes().decode().replace('\0',' ')
assert 'controller_future.py' in cmd
os.kill(controller,signal.SIGSTOP)
request=r/'BOUNDARY_REQUEST.json';assert not request.exists()
request.write_text(json.dumps(dict(reason='user-authorized v5 exact-update performance maintenance',time=time.time(),controller_pid=controller))+'\n')
print(json.dumps(dict(controller_suspended=controller,boundary_requested=True)))
