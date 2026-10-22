from motion_mamba2.paths import artifact_path
import sys, importlib.util, gc
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'codec_graph_accel_20261007'
sys.path.insert(0,str(R));from common import *
import torch
from motion_training import runtime
from lazy_plan import Plan
import pipeline_accel as candidate
spec=importlib.util.spec_from_file_location('previous_pipeline',str(R/'pipeline_accel_20261007/pipeline_accel.py'))
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
m,c,ck=runtime.load(read(S/'BOUNDARY.json')['checkpoint'],True);m.autoencoder.eval()
cfg=read(R/'config.json');plan=Plan(read(R/'MANIFEST_TEMPLATE.json'));rows=[]
for repeat in range(2):
    for h,f in [(0,0),(52,0),(196,0),(0,196),(52,144),(144,52)]:
        records=plan.records_at(71000+len(rows),64,layout=(h,f))
        prepared=previous.submit_reads(c,records,cfg,0)
        ref,_=previous.build_group(m,c,prepared,cfg,serial_codec=True)
        cand,_=candidate.build_group(m,c,prepared,cfg,serial_codec=True)
        fields={k:dict(exact=torch.equal(v,cand[k]),max_abs=float((v.float()-cand[k].float()).abs().max())) for k,v in ref.items() if torch.is_tensor(v)}
        passed=all(v['exact'] for v in fields.values())
        rows.append(dict(H=h,F=f,repeat=repeat,batch=64,passed=passed,fields=fields))
        print('INPUT_GATE',h,f,repeat,passed,{k:v for k,v in fields.items() if not v['exact']},flush=True)
        del ref,cand;gc.collect()
coder=c._frozen_codec_graphs
write(S/'INPUT_GATE.json',dict(passed=all(x['passed'] for x in rows),rows=rows,
    graph_shapes=[str(x) for x in coder.graphs],replays=coder.replays,checkpoint=read(S/'BOUNDARY.json')['checkpoint'],
    previous_execution_backend='batch_codec_geometry_and_cpu_prefetch_v1',all_original_tensor_fields_checked=True))
