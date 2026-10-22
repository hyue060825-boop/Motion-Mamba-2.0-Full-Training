"""Reuse existing EGL renderer for fixed accepted-source strata; no solver calls."""
import ast,contextlib,json
from g1_pipeline.common import ROOT,read_jsonl,write_json,write_jsonl,sha
from g1_pipeline.kinematic_policy import BASE,load,authorize

def renderer():
 from g1_pipeline import render_frozen_review as original
 tree=ast.parse((ROOT/'g1_pipeline/render_frozen_review.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 class Official(ast.NodeTransformer):
  def visit_If(self,node):
   if ast.unparse(node.test)=="source == 'bones'":
    if any(isinstance(n,ast.Assign) and any(ast.unparse(t)=='metadata' for t in n.targets) for n in node.body):
     node.body=ast.parse("metadata={sid:r['source_metadata'] for sid,r in records.items()}").body
    else:
     node.body=ast.parse("""meta=metadata[sid]
qp=ROOT/row['output_paths']['qpos_path'];reference=BONES/meta['source_paths']['g1']
candidate=np.load(qp)
ref,rt,_=read_csv(reference,format_id=FORMAT,fps=meta['original_fps'])
ref,times,_=resample(ref,rt,m.jnt_range[1:])
files=[qp,reference,ROOT/row['output_paths']['contract_path']]
labels=('Official native resampled','Exported G1')
""").body
    return node
   return self.generic_visit(node)
 fn=Official().visit(fn);ns=vars(original).copy();exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'kinematic_full_clip_renderer','exec'),ns)
 return ns['run']

def run():
 authorize();panel=load(BASE/'final_review/selection.json');render=renderer();manifest=[]
 for source in ['bones','humanml3d','hiphi']:
  rows=[i['record'] for i in panel['records'] if i['record']['dataset']==source]
  ns='kinematic_training_20260917_final_review_'+source;ledger=[]
  for r in rows:
   ledger.append({'sample_id':r['sample_id'],'source_metadata':r,'metrics_path':r.get('metrics_path'),'output_paths':{k:r[k] for k in ['qpos_path','features_path','csv_path','sidecar_path']}})
   ledger[-1]['output_paths']['contract_path']=str((ROOT/r['qpos_path']).parent.relative_to(ROOT)/'contract.json')
  write_jsonl(ROOT/'work'/ns/'results.jsonl',ledger)
  with (BASE/'final_review/render.log').open('a') as log,contextlib.redirect_stdout(log):
   records=render(ns,[r['sample_id'] for r in rows],source=source)
  manifest.extend(records)
 write_json(BASE/'final_review/videos.json',{'scope':'full videos for fixed stratified accepted panel, not dynamic tracking','selected':len(panel['records']),'rendered':len(manifest),'videos':manifest,'selection_sha256':sha(BASE/'final_review/selection.json'),'renderer_workers':1})
 print({'status':'STRATIFIED_VIDEOS_COMPLETE','clips':len(manifest)})
if __name__=='__main__':run()
