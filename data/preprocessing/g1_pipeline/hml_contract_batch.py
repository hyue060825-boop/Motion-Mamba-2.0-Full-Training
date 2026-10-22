from pathlib import Path
import json,numpy as np,torch
from .common import ROOT,write_jsonl,write_json
from .robot import export_csv,encode_csv

def run():
 from .gates import require_pass
 require_pass('01_bones_calibration')
 rows=[json.loads(x) for x in (ROOT/'manifests/hml_smoke100_results.jsonl').read_text().splitlines()];out=ROOT/'work/hml_smoke_csv';out.mkdir(exist_ok=True);res=[]
 for r in rows:
  try:
   q=np.load(ROOT/r['feature_source']);t=np.arange(len(q))/25.;p=out/(r['sample_id'].replace(':','_')+'.csv');export_csv(p,q,t,{'sample_id':r['sample_id'],'source':'native_amass_smoke100','gmr_commit':'bb1bbe40774794fceb2a7c579a3464a28e68c844'});f,rep,tr=encode_csv(p);res.append({'sample_id':r['sample_id'],'status':'PASS','frames':len(q),'features_shape':list(f.shape),'finite':bool(torch.isfinite(f).all()),'csv':str(p.relative_to(ROOT))})
  except Exception as e:res.append({'sample_id':r['sample_id'],'status':'ERROR','error':repr(e)})
 write_jsonl(ROOT/'manifests/hml_smoke100_contract_results.jsonl',res);write_json(ROOT/'qa/aggregate/hml_smoke100_contract.json',{'attempted':len(res),'passed':sum(x['status']=='PASS' for x in res),'errors':sum(x['status']=='ERROR' for x in res),'status':'PASS' if len(res)==100 and all(x['status']=='PASS' for x in res) else 'FAIL'});return res
if __name__=='__main__':print(run()[-1])
