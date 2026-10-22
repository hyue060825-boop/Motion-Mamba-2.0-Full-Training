"""Audit finished models in an isolated output directory without finalizing the suite."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import hashlib,json,time

R=Path(artifact_path('humanml3d_section1_20261020_c1c76adb30'))
O=R/'partial_12_20261021'
O.mkdir(exist_ok=True)
source=(R/'aggregate.py').read_text()
derived=source.replace('from evaluate_suite import R,MODELS,TASKS,rd,wr,sha',
    "from evaluate_suite import R,MODELS,TASKS,rd,wr,sha\nFULL_MODELS=MODELS[:]\nMODELS=[m for m in FULL_MODELS if (R/'models'/m/'COMPLETE.json').exists()]\nMISSING=[m for m in FULL_MODELS if m not in MODELS]\nassert len(MODELS)==12 and MISSING==['motionstreamer'],(MODELS,MISSING)\nO=R/'partial_12_20261021'\n")
derived=derived.replace('wr(R/','wr(O/').replace('save(R/','save(O/').replace("np.savez_compressed(R/f'","np.savez_compressed(O/f'")
derived=derived.replace('dict(passed=True,models=MODELS,N=N,tasks=TASKS,bindings=bindings,',
    'dict(passed=True,partial=True,all_models_complete=False,missing_models=MISSING,models=MODELS,N=N,tasks=TASKS,bindings=bindings,')
derived=derived.replace("import subprocess\nsubprocess.run([sys.executable,str(R/'render_results.py')],check=True)\n",'')
assert derived!=source and 'subprocess.run' not in derived
(O/'AUDIT_EXECUTED_SOURCE.py').write_text(derived)
(O/'DERIVATION.json').write_text(json.dumps(dict(purpose='Independent completion audit of existing12 models; canonical controller and13-model finalization unchanged',
    aggregate_sha256=hashlib.sha256(source.encode()).hexdigest(),executed_sha256=hashlib.sha256(derived.encode()).hexdigest(),output=str(O),time=time.time()),indent=2)+'\n')
exec(compile(derived,str(O/'AUDIT_EXECUTED_SOURCE.py'),'exec'),{'__name__':'__main__'})
print((O/'SUMMARY.csv').read_text())
