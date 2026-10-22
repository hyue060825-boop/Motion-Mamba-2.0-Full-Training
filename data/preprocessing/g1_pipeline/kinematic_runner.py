"""Thin authorized branch of frozen v4 coordinator; original file stays untouched."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import ast,json
from .common import ROOT,sha,write_json
from .kinematic_policy import BASE,authorize,load

def initialize(directory,panel):
 policy=authorize();release=load(BASE/'execution_release.json');spec=load(panel)
 if spec['scope']!='kinematic_training' or spec['release']!=str(BASE.relative_to(ROOT)):raise ValueError('WRONG_KINEMATIC_PANEL')
 if release['panel_hashes'].get(str(panel.relative_to(ROOT)))!=sha(panel):raise ValueError('UNSEALED_KINEMATIC_PANEL')
 bindings=dict(release['bindings']);bindings[str(panel.relative_to(ROOT))]=sha(panel)
 from .parallel_runner import bound_check
 bound_check(bindings)
 return {'scope':'kinematic_training','execution_scope':'kinematic_training','namespace':directory.name,'panel':str(panel.relative_to(ROOT)),'expected':len(spec['tasks']),'seed':20260917,'threads':{n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},'bindings':bindings,'code':release['original_clip_code'],'asset':release['asset'],'training_release':False,'kinematic_policy_sha256':sha(BASE/'execution_policy.json'),'state_QA_PASS_means':'artifact committed; acceptance is separately recorded in kinematic decision'}

def compiled():
 tree=ast.parse((ROOT/'g1_pipeline/parallel_runner.py').read_text())
 class Entry(ast.NodeTransformer):
  def visit_Constant(self,node):
   if node.value=='g1_pipeline.parallel_runner':node.value='g1_pipeline.kinematic_runner'
   elif node.value in ('benchmark_','pilot_parallel_'):node.value='kinematic_training_'
   return node
  def visit_ImportFrom(self,node):
   if node.module=='parallel_clip':node.module='kinematic_clip'
   return node
 tree=Entry().visit(tree)
 # Stop dispatch on genuine integrity/resource faults after preserving the error.
 worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
 class FaultGuard(ast.NodeTransformer):
  def visit_Expr(self,node):
   if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='state.commit':
    return [node,*ast.parse("if entry['status']=='ERROR' and any(token in entry.get('error','') for token in ('CHANGED','MISMATCH','No space left','MemoryError','Input/output error')): raise RuntimeError('KINEMATIC_INTEGRITY_OR_RESOURCE_FAULT:'+entry['error'])").body]
   return node
 FaultGuard().visit(worker)
 runfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 class ReuseAccounting(ast.NodeTransformer):
  def visit_Expr(self,node):
   if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='write_json' and 'session_' in ast.unparse(node.value.args[0]):
    return [*ast.parse("inline_reused = sum(bool(r.get('reused')) for r in entries)\nreport['reused_clips'] = max(report['reused_clips'],inline_reused)\nreport['newly_processed_frames'] = sum(r.get('frames',0) for r in entries if not r.get('reused') and r['sample_id'] not in initial_complete)\nif inline_reused: report.update(throughput_valid=False,frames_per_sec=None,source_motion_hours_per_hour=None)").body,node]
   return node
 ReuseAccounting().visit(runfn)
 # initialize is the only policy branch; run/worker ownership, retry and commit stay v4.
 tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='initialize') and not (isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")]
 ns={'__name__':'g1_pipeline.kinematic_runner_impl','__package__':'g1_pipeline','initialize':initialize}
 exec(compile(ast.fix_missing_locations(tree),str(ROOT/'g1_pipeline/parallel_runner.py'),'exec'),ns)
 return ns
if __name__=='__main__':compiled()['main']()
