"""Compare reconstructed initialization against a trusted original zero-step save, on CPU."""
from pathlib import Path
import argparse,json,tempfile,sys
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'training'),str(R/'data')]
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
import torch
from motion_training.runtime import load
from motion_training._compat.legacy_loss import _denoiser_aux_state
from motion_mamba2.legacy_compat import checkpoint_metadata
torch.set_num_threads(2)
x=torch.load(a.reference,map_location='cpu',mmap=True,weights_only=False);assert x['step']==0 and not x['optimizer']['state']
meta=checkpoint_metadata(json.loads((R/'configs/initialization/metadata.json').read_text()))
with tempfile.TemporaryDirectory() as d:
 q=Path(d)/'metadata.pt';torch.save(meta,q);m,c,k=load(q,device=a.device,fresh_seed=123,official_frontend=True)
 raw=m.denoiser.model;checks={}
 for name,state in [('root_model',raw.root_model.state_dict()),('body_model',raw.body_model.state_dict()),('denoiser_aux',_denoiser_aux_state(raw))]:
  assert state.keys()==x[name].keys(),name
  bad=[dict(key=key,max_abs=float((v.cpu().float()-x[name][key].cpu().float()).abs().max())) for key,v in state.items() if not torch.equal(v.cpu(),x[name][key].cpu())]
  assert not bad,(name,bad[:8])
  checks[name]=len(state)
print(json.dumps(dict(passed=True,exact_initial_tensors=checks,optimizer_updates=0,scope='Initialization identity on '+a.device+'; not a training rerun'),indent=2))
