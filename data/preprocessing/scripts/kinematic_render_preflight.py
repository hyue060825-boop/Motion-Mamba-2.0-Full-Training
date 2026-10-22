import contextlib
from g1_pipeline.common import ROOT,write_jsonl,write_json,sha
from g1_pipeline.kinematic_policy import BASE,load
from kinematic_render import renderer
rows=load(BASE/'runtime_preflight/source_results.json')['results'];render=renderer();allvideos=[]
for row in rows:
 source=row['dataset'];ns='kinematic_training_frontend_preflight_'+source
 write_jsonl(ROOT/'work'/ns/'results.jsonl',[row])
 with (BASE/'runtime_preflight/render.log').open('a') as stream,contextlib.redirect_stdout(stream):allvideos.extend(render(ns,[row['sample_id']],source=source))
write_json(BASE/'runtime_preflight/videos.json',{'scope':'new renderer entry check only, not final release stratified review','videos':allvideos,'renderer_sha256':sha(ROOT/'scripts/kinematic_render.py')})
print({'status':'PASS','full_clips':len(allvideos)})
