from motion_mamba2.paths import artifact_path
import sys
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'pipeline_accel_20261007'
sys.path.insert(0,str(R));from common import *
import torch,gc
from motion_training import runtime,batching
from lazy_plan import Plan
import pipeline_accel as acc
m,c,ck=runtime.load(read(S/'BOUNDARY.json')['checkpoint'],True);m.autoencoder.eval()
cfg=read(R/'config.json');plan=Plan(read(R/'MANIFEST_TEMPLATE.json'));rows=[]
for hf in [(0,0),(52,0),(196,0),(0,196),(52,144),(144,52)]:
    records=plan.records_at(63000+len(rows),64,layout=hf)
    ref=batching.collate([batching.corrected_build(m,c,r,cfg,audit=False)[0] for r in records])
    cand,_=acc.build_group(m,c,acc.materialize(acc.submit_reads(c,records,cfg,4)),cfg,serial_codec=True)
    fields={k:dict(exact=torch.equal(v,cand[k]),max_abs=float((v.float()-cand[k].float()).abs().max())) for k,v in ref.items() if torch.is_tensor(v)}
    passed=all(v['exact'] for v in fields.values())
    rows.append(dict(H=hf[0],F=hf[1],batch=64,passed=passed,fields=fields))
    print('SERIAL_GATE',hf,passed,{k:v for k,v in fields.items() if not v['exact']},flush=True)
    del ref,cand;gc.collect();torch.cuda.empty_cache()
passed=all(v['passed'] for v in rows)
write(S/'SERIAL_INPUT_GATE.json',dict(passed=passed,rows=rows,checkpoint=read(S/'BOUNDARY.json')['checkpoint'],exact_targets_preserved=passed))
