from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,time,subprocess,json,fcntl
R=Path(artifact_path('g1_external_matched100k_evaluation_20261019_aa0c66c5b2'))
lock=(R/'delivery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
deadline=time.monotonic()+6*3600
while not (R/'NUMERICS_COMPLETE.json').exists():
 if (R/'FAILURE.json').exists():raise RuntimeError('Evaluation controller failed; preserve evidence and repair before delivery')
 if time.monotonic()>deadline:raise TimeoutError('No completed numerics within six hours; no extra jobs launched')
 time.sleep(10)
if not (R/'DELIVERY_COMPLETE.json').exists():
 subprocess.run([sys.executable,str(R/'make_report.py')],cwd=R,check=True)
for name in ['DiP_G1','DART_MLP_G1']:
 p=Path(artifact_path('g1_external_matched100k_20261019_f77433d79c'))/name/'EVALUATION_HANDOFF.json'
 d=json.loads(p.read_text());d.update(status='COMPLETE_FROZEN_ROOT_ENDPOINT_DIAGNOSTIC',delivery=str(R/'two_models_evaluation_20261019.zip'),full_future_conditioning_equivalent=False,time=time.time())
 p.write_text(json.dumps(d,indent=2))
print('DELIVERY_COMPLETE',R/'two_models_evaluation_20261019.zip',flush=True)
