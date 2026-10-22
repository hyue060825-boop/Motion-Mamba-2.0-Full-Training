"""Independent CPU save integrity check; never treats a filename as a training step."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--expected-step',type=int);p.add_argument('--expected-sha256');a=p.parse_args()
import torch
torch.set_num_threads(2);path=Path(a.checkpoint);h=hashlib.sha256()
with path.open('rb') as f:
 for b in iter(lambda:f.read(8388608),b''):h.update(b)
digest=h.hexdigest()
if a.expected_sha256:assert digest==a.expected_sha256
x=torch.load(path,map_location='cpu',mmap=True,weights_only=False);step=x.get('step');assert isinstance(step,int)
if a.expected_step is not None:assert step==a.expected_step,(step,a.expected_step)
opt=x.get('optimizer');assert opt is not None
states=opt['state']
if step==0:assert not states
for state in states.values():
 assert torch.isfinite(torch.as_tensor(state['step'])).all()
 for key in ['exp_avg','exp_avg_sq']:assert state[key].dtype==torch.float32 and torch.isfinite(state[key]).all()
schedule=x.get('scheduler',x.get('framework',{}).get('scheduler'))
rng=x.get('rng_by_rank',x.get('rng'))
assert rng is not None
print(json.dumps(dict(passed=True,actual_step=step,sha256=digest,Adam_states=len(states),FP32_moments_finite=True,scheduler=schedule,RNG_saved=True,not_a_quality_claim=True),default=str,indent=2))
