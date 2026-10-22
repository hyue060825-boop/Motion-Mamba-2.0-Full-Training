"""CPU-only trained-forward probe, independent of the rolling evaluator."""
from motion_mamba2.paths import artifact_path
import os,sys,json,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
S=Path(artifact_path('g1_external_matched100k_20261019_f77433d79c'))
R=Path(artifact_path('g1_external_matched100k_evaluation_20261019_aa0c66c5b2'))
sys.path.insert(0,str(S/'scripts'))
import torch
from external_model import ExternalDenoiser
torch.set_num_threads(2);torch.manual_seed(7361)
results=[]
for method,name in [('dip','DiP_G1'),('dart','DART_MLP_G1')]:
 ck=torch.load(S/name/'train/checkpoint100000.pt',map_location='cpu',weights_only=False,mmap=True)
 raw=ExternalDenoiser(method,S/'repos'/('DiP' if method=='dip' else 'DART'),None,None)
 raw.load_state_dict(ck['model'],strict=True);raw.eval();raw.requires_grad_(False)
 H,G,F=52,52,144;nf=H+G+F;ix=torch.arange(nf)[None];ti=torch.arange(nf//4)[None]
 mm=torch.zeros(1,nf,414);obs=torch.zeros_like(mm)
 mm[:,H+G-1,[0,2]]=1
 args=dict(x=torch.randn(1,nf//4,148),history_len=torch.tensor([H]),generation_len=torch.tensor([G]),future_len=torch.tensor([F]),
  history_mask=ix<H,generation_mask=(ix>=H)&(ix<H+G),future_mask=ix>=H+G,
  history_token_mask=ti<H//4,generation_token_mask=(ti>=H//4)&(ti<(H+G)//4),future_token_mask=ti>=(H+G)//4,
  text_feat=torch.randn(1,1,4096),text_pad_mask=torch.ones(1,1,dtype=torch.bool),t=torch.tensor([500]),
  first_heading_angle=torch.zeros(1),motion_mask=mm,observed_motion=obs)
 with torch.inference_mode():
  reference=raw(**args)
  changed=obs.clone();changed[:,H+G-1,[0,2]]=1
  goal=raw(**dict(args,observed_motion=changed))
  text=raw(**dict(args,text_feat=torch.zeros_like(args['text_feat'])))
  hidden=obs.clone();hidden[:,0,50]=900
  hidden_out=raw(**dict(args,observed_motion=hidden))
 gen=args['generation_token_mask'];gd=float((goal-reference)[gen].abs().max());td=float((text-reference)[gen].abs().max())
 assert torch.isfinite(reference).all() and torch.isfinite(goal).all() and gd>0 and td>0
 assert torch.equal(reference,hidden_out)
 results.append(dict(method=method,status='PASS_TRAINED_CANONICAL_FORWARD_CONDITION_SENSITIVITY_AND_MASK',visible_goal_max_output_delta=gd,text_max_output_delta=td,masked_unobserved_value_delta=0.,test_scope='finite short native input; not evidence of successful goal completion'))
 del raw,ck
(R/'CONDITIONING_FORWARD_PROBE.json').write_text(json.dumps(dict(status='PASS',device='CPU',no_training_updates=True,results=results,time=time.time()),indent=2))
print(json.dumps(results,indent=2))
