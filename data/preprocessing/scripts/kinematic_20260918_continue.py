"""Resume the existing queue with bound recovery and official BONES admission."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import ast,sys,subprocess
from g1_pipeline.common import ROOT,write_json,now
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline.kinematic_20260918_runner import bound_check,NoUnrelatedTrainingImport
import kinematic_release as original

def run():
    sys.meta_path.insert(0,NoUnrelatedTrainingImport())
    bound_check(load(BASE/'execution_release.json')['bindings'])
    tree=ast.parse((ROOT/'scripts/kinematic_release.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Entry(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='g1_pipeline.kinematic_runner':node.value='g1_pipeline.kinematic_20260918_runner'
            return node
    fn=Entry().visit(fn);env=vars(original).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'kinematic_recovery_release','exec'),env)
    write_json(BASE/'continuation_status.json',{'time':now(),'phase':'EXECUTING_AUTHORIZED_20260918_RECOVERY','pid':os.getpid(),'admission_policy':'bones_official_20260918/policy.json'})
    env['run']()
    if load(BASE/'status.json')['phase']=='CONVERSION_COMPLETE_FINALIZATION_REQUIRED':
        subprocess.run([sys.executable,'scripts/kinematic_postprocess.py'],cwd=ROOT,check=True)

if __name__=='__main__':run()
