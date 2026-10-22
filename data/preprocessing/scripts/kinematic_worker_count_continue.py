"""User-authorized 16/32-worker continuation using the unchanged clip scheduler."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,ast,sys,subprocess
from g1_pipeline.common import ROOT,write_json,now,sha
from g1_pipeline.kinematic_policy import BASE,load
from g1_pipeline.kinematic_20260918_runner import bound_check,NoUnrelatedTrainingImport
import kinematic_release as original


def compiled_run(workers):
    if workers not in (16,32):raise ValueError('ONLY_AUTHORIZED_16_OR_32_WORKERS')
    tree=ast.parse((ROOT/'scripts/kinematic_release.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Entry(ast.NodeTransformer):
        changed=0
        def visit_Constant(self,node):
            if node.value=='g1_pipeline.kinematic_runner':node.value='g1_pipeline.kinematic_20260918_runner'
            return node
        def visit_List(self,node):
            self.generic_visit(node)
            for index,item in enumerate(node.elts[:-1]):
                if isinstance(item,ast.Constant) and item.value=='--workers':
                    assert isinstance(node.elts[index+1],ast.Constant) and node.elts[index+1].value=='16'
                    node.elts[index+1].value=str(workers);self.changed+=1
            return node
    edit=Entry();fn=edit.visit(fn)
    if edit.changed!=1:raise ValueError('WORKER_COUNT_DISPATCH_NOT_UNIQUE')
    env=vars(original).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'authorized_worker_count_release','exec'),env)
    return env['run']


def run(workers):
    sys.meta_path.insert(0,NoUnrelatedTrainingImport())
    bound_check(load(BASE/'execution_release.json')['bindings'])
    write_json(BASE/'continuation_status.json',{'time':now(),'phase':'EXECUTING_AUTHORIZED_WORKER_COUNT','pid':os.getpid(),
               'workers':workers,'entry_sha256':sha(ROOT/'scripts/kinematic_worker_count_continue.py'),
               'admission_policy':'bones_official_20260918/policy.json','solver_and_clip_logic_changed':False})
    compiled_run(workers)()
    if load(BASE/'status.json')['phase']=='CONVERSION_COMPLETE_FINALIZATION_REQUIRED':
        subprocess.run([sys.executable,'scripts/kinematic_postprocess.py'],cwd=ROOT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=[16,32],required=True)
    run(p.parse_args().workers)
