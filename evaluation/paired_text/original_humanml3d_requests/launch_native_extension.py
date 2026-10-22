from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,json,time,subprocess
R=Path(artifact_path('unified_interactive_easy_v3_20261021_ebde76396c'));E=R/'extension_native_finetunes_20261021'
E.mkdir(exist_ok=True)
old=E/'CONTROLLER_PID.json'
if old.exists():
 pid=json.loads(old.read_text())['pid'];p=Path(f'/proc/{pid}/cmdline')
 if p.exists() and b'native_extension_controller.py' in p.read_bytes():raise SystemExit('Existing additional-model controller: not duplicated')
with (E/'controller.log').open('a') as log:
 p=subprocess.Popen([artifact_path('python_9f8003379b'),str(R/'native_extension_controller.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
data=dict(pid=p.pid,script=str(R/'native_extension_controller.py'),models=['native110','native117','native122'],regular122_not_long_distill=True,new_conditions=12,new_rollouts=4608,training_or_online_changes=False,time=time.time())
(E/'LAUNCH.json').write_text(json.dumps(data,indent=2)+'\n')
(R/'EXTENSION_ACTIVE.json').write_text(json.dumps(dict(authorization='User explicitly requested Native122k117k110k on2026-10-21',root=str(E),status=str(E/'STATUS.json'),original9_model_results_preserved=True,**data),indent=2)+'\n')
print(json.dumps(data))
