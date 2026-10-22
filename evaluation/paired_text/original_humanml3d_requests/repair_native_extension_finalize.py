"""Resume only final publication after an import-path execution repair."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,json,time,subprocess,shutil,fcntl,hashlib
R=Path(artifact_path('unified_interactive_easy_v3_20261021_ebde76396c'));E=R/'extension_native_finetunes_20261021'
lock=(E/'publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((E/'CONTROLLER_PID.json').read_text());p=Path(f'/proc/{old["pid"]}/cmdline')
assert not p.exists(),'Existing evaluation controller must not be duplicated'
assert not json.loads((E/'ACTIVE_CHILD.json').read_text())['generation']
for m in ['native110','native117','native122']:
 for task in ['short52','nominal9','nominal20','nominal40']:
  assert json.loads((R/'models'/m/task/'GENERATION_COMPLETE.json').read_text())['passed']
  assert json.loads((R/'scoring/models'/m/task/'COMPLETE.json').read_text())['passed']
a=E/'maintenance/20261021_finalizer_import_path';a.mkdir(parents=True,exist_ok=True)
if (E/'FAILURE.json').exists():shutil.move(E/'FAILURE.json',a/'FAILURE.json')
shutil.copy2(E/'finalize.log',a/'finalize_before_repair.log')
(a/'REPAIR.json').write_text(json.dumps(dict(reason='Finalizer ran from nested source directory without parent execution engine on sys.path; add parent source path before import',generation_changes=False,scoring_changes=False,training_changes=False,only_publication_restarted=True,time=time.time()),indent=2)+'\n')
(E/'FINALIZER_PID.json').write_text(json.dumps(dict(pid=os.getpid(),cwd=str(Path.cwd()),script=str(Path(__file__).resolve()),time=time.time()),indent=2)+'\n')
with (E/'finalize.log').open('a') as log:
 subprocess.run([artifact_path('python_9f8003379b'),str(E/'finalize_extension.py')],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Combined48-condition result bundle ready')
