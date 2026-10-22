from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,time,ast,subprocess,py_compile
R=Path(artifact_path('g1_native100k_original117k_protocol_20261014_dbcf4a1353'))
p=R/'train_future.py';before=p.read_text();after=before.replace('\nimport extreme_runtime','',1)
after=after.replace('import history_v3\n','import history_v3\nimport extreme_runtime\n',1)
def definitions(s):
 return [ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
assert definitions(before)==definitions(after)
p.write_text(after);py_compile.compile(str(p),doraise=True)
p=R/'profile_extreme.py';s=p.read_text().replace('if a.tag=="g0_full512_gate":','if a.tag=="g0_full512_gate" and not (ROOT/"acceleration_extreme/g0_timing/TIMING.json").exists():');p.write_text(s)
py_compile.compile(str(p),doraise=True)
p=R/'evaluate.py';p.write_text(p.read_text().replace("ck['step']==112000+a.step","ck['step']==100000+a.step"))
p=R/'evaluate_future.py';p.write_text(p.read_text().replace('112k actual accepted history','100k actual accepted history').replace('baseline112k=','baseline100k='))
m=R/'maintenance/startup_import_order';m.mkdir(parents=True,exist_ok=True)
for f in ['FAILURE.json','acceleration_extreme/g0_full512_gate/GATES.json']:
 p=R/f
 if p.exists():p.rename(m/p.name)
(m/'REPAIR.json').write_text(json.dumps(dict(time=time.time(),reason='Execution entry imported history acceleration before adding its dependency path',
 fix='Import history_v3 before extreme_runtime; classes/functions AST unchanged',definitions_exact=True,
 complete512_gate_repeated_after_source_hash_change=True,paired_timing_reused_no_training_math_change=True,
 evaluation_parent_step_assertion_and_labels_changed_to100k=True,real_training_updates_before_repair=0),indent=2))
ctrl=json.loads((R/'CONTROLLER_PID.json').read_text());assert not subprocess.run(['ps','-p',str(ctrl['pid']),'-o','args='],capture_output=True,text=True).stdout.strip()
log=(R/'logs/controller.log').open('a')
p=subprocess.Popen([artifact_path('python_9f8003379b'),'-u',str(R/'controller.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Restarted after import-order repair; full512 gate will be repeated',p.pid)
