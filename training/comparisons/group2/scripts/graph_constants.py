"""Cache literal index/metadata GPU constants during isolated geometry capture.

No masks or training tensors are declared valid or replaced. Constants must
already have been encountered in warmup; an unseen capture constant is rejected.
The context is removed before returning to denoiser/optimizer execution.
"""
import contextlib,numbers
import torch

class Constants:
    def __init__(self):self.values={}
    @contextlib.contextmanager
    def use(self):
        import motion_mamba2.skeleton.kinematics as kin
        get=torch.Tensor.__getitem__;ctor=torch.tensor;asctor=torch.as_tensor;to=torch.Tensor.to
        rigid=kin.batch_rigid_transform
        owner=self
        def cached(key,make):
            if key not in owner.values:
                if torch.cuda.is_current_stream_capturing():raise RuntimeError('Unwarmed geometry constant: '+str(key)[:200])
                owner.values[key]=make()
            return owner.values[key]
        def index(x,idx):
            if not x.is_cuda:return get(x,idx)
            def one(v):
                if isinstance(v,list) and all(isinstance(a,numbers.Integral) for a in v):
                    key=('index',tuple(map(int,v)),str(x.device))
                    return cached(key,lambda:ctor(v,dtype=torch.long,device=x.device))
                return v
            new=tuple(one(v) for v in idx) if isinstance(idx,tuple) else one(idx)
            return get(x,new)
        def make_ctor(fn):
            def construct(data,*args,**kw):
                dev=kw.get('device')
                if dev is None or torch.device(dev).type!='cuda' or torch.is_tensor(data):return fn(data,*args,**kw)
                import numpy as np
                arr=np.asarray(data)
                key=('literal',arr.shape,str(arr.dtype),arr.tobytes(),str(torch.device(dev)),str(kw.get('dtype')),str(torch.get_default_dtype()),fn.__name__)
                return cached(key,lambda:fn(data,*args,**kw))
            return construct
        def transfer(x,*args,**kw):
            dev=kw.get('device');dtype=kw.get('dtype')
            if args:
                if torch.is_tensor(args[0]):dev=args[0].device;dtype=args[0].dtype
                elif isinstance(args[0],(str,torch.device,int)):dev=args[0]
            if x.device.type=='cpu' and dev is not None and torch.device(dev).type=='cuda' and not x.requires_grad:
                assert x.numel()<100000,'Geometry constant unexpectedly large'
                key=('CPU_to_GPU',tuple(x.shape),str(x.dtype),x.contiguous().numpy().tobytes(),str(torch.device(dev)),str(dtype))
                return cached(key,lambda:to(x,*args,**kw))
            return to(x,*args,**kw)
        def capture_rigid(rot_mats,joints,parents,root_idx):
            key=('skeleton',parents.data_ptr(),parents._version,tuple(parents.shape),root_idx,str(joints.device))
            def metadata():
                cpu=parents.detach().cpu()
                levels=kin.compute_idx_levels(cpu)
                nonroot=[i for i in range(len(cpu)) if i!=root_idx]
                def gpu(v):return ctor(v,dtype=torch.long,device=joints.device)
                return gpu(nonroot),gpu(cpu[nonroot].tolist()),[(gpu(v.tolist()),gpu(cpu[v].tolist())) for v in levels]
            nonroot,parent_nonroot,levels=cached(key,metadata)
            joints=joints.unsqueeze(-1)
            relative=joints.clone()
            relative[:,nonroot]-=joints[:,parent_nonroot].clone()
            matrices=kin.transform_mat(rot_mats.reshape(-1,3,3),relative.reshape(-1,3,1)).reshape(-1,joints.shape[1],4,4)
            transforms=torch.zeros_like(matrices)
            transforms[:,root_idx]=matrices[:,root_idx]
            for indices,parent_indices in levels:
                result=torch.matmul(transforms[:,parent_indices],matrices[:,indices])
                transforms[:,indices]=result
            return transforms[:,:,:3,3],transforms[:,:,:3,:3]
        torch.Tensor.__getitem__=index;torch.tensor=make_ctor(ctor);torch.as_tensor=make_ctor(asctor);torch.Tensor.to=transfer
        kin.batch_rigid_transform=capture_rigid
        try:yield
        finally:
            torch.Tensor.__getitem__=get;torch.tensor=ctor;torch.as_tensor=asctor;torch.Tensor.to=to
            kin.batch_rigid_transform=rigid
