"""Exact CPU forward/gradient check against an explicitly supplied original layer source."""
from pathlib import Path
import argparse,ast,json,typing
import torch
from torch import nn,Tensor
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--original',required=True);a=p.parse_args();R=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
def cls(path):
 tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['_get_activation_fn','SDPATransformerEncoderLayer']]
 env=dict(torch=torch,nn=nn,F=F,Optional=typing.Optional,Union=typing.Union,Tensor=Tensor);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env);return env['SDPATransformerEncoderLayer']
old=cls(a.original);new=cls(R/'src/motion_mamba2/models/backbone.py');checks=[]
for norm_first in [False,True]:
 torch.manual_seed(123);m=old(32,4,64,dropout=0,batch_first=True,norm_first=norm_first);n=new(32,4,64,dropout=0,batch_first=True,norm_first=norm_first);n.load_state_dict(m.state_dict(),strict=True)
 for masked in [False,True]:
  x=torch.randn(3,13,32,requires_grad=True);y=x.detach().clone().requires_grad_();mask=torch.zeros(3,13,dtype=torch.bool)
  if masked:mask[0,-3:]=True;mask[1,-1:]=True
  o=m(x,src_key_padding_mask=mask);v=n(y,src_key_padding_mask=mask);assert torch.equal(o,v)
  o.square().mean().backward();v.square().mean().backward();assert torch.equal(x.grad,y.grad)
  for (ka,pa),(kb,pb) in zip(m.named_parameters(),n.named_parameters()):assert ka==kb and torch.equal(pa.grad,pb.grad)
  checks.append(dict(norm_first=norm_first,masked=masked,forward_exact=True,input_gradient_exact=True,parameter_gradients_exact=True));m.zero_grad();n.zero_grad()
print(json.dumps(dict(passed=True,checks=checks,scope='Exact layer rename smoke, not full model/GPU/100k training reproduction'),indent=2))
