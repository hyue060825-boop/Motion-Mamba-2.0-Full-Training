from common import *
import random,gc,copy
import numpy as np
import torch
from external_model import ExternalDenoiser
from motion_training import runtime
from motion_training._compat import legacy_loss
from g1_optimizer_precision import create_adamw
from motion_training.io import identity,sha

class Client:
    def reset_memory(self):pass

def load(checkpoint=None):
    cfg=read(ROOT/'configs/batch32.json');method=read(ROOT/'PLAN.json')['method'];seed=123
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=runtime.load_model('g1',device='cuda',text_encoder=False,fresh_denoiser=True)
    legacy_loss._replace_stats(m,cfg['stats_dir'])
    m.training_codec_contract=cfg['training_codec_contract'];m.generation_contract=cfg['generation_contract']
    raw=ExternalDenoiser(method,SUITE/'repos'/('CMDM' if method=='cmdm' else 'MotionStreamer'),m.motion_rep,m.hybrid).cuda()
    m.denoiser.model=raw;gc.collect();torch.cuda.empty_cache()
    m.eval();m.autoencoder.eval()
    for p in m.parameters():p.requires_grad_(False)
    raw.apply_trainable_policy()
    trainable=[(n,p) for n,p in raw.named_parameters() if p.requires_grad]
    opt=create_adamw([p for n,p in trainable],precision='fp32_master_v1',parameter_names=['module.'+n for n,p in trainable],
        lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
    opt.training_schedule=dict(spec=cfg['scheduler'],completed=0)
    ck=dict(step=0,rng_by_rank=None)
    if checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert ck['method']==method
        raw.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
        opt.training_schedule=copy.deepcopy(ck['scheduler'])
        assert opt.training_schedule['completed']==ck['step']
        runtime.restore_rng(ck['rng_by_rank'][torch.distributed.get_rank()])
    return m,Client(),opt,ck

def save(m,opt,step,path):
    from motion_training import distributed as parallel
    rng=runtime.rng();states=parallel.gather(rng)
    digest=identity([m.denoiser.model.state_dict(),opt.state_dict()])
    hashes=parallel.gather(digest);assert len(set(hashes))==1,'rank model/Adam divergence'
    if parallel.leader():
        path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix('.tmp')
        payload=dict(method=read(ROOT/'PLAN.json')['method'],variant='common_G1_variable_length_architecture_adapter_v2',step=step,
            model=m.denoiser.model.state_dict(),optimizer=opt.state_dict(),scheduler=opt.training_schedule,
            rng_by_rank=states,rank_state_hashes=hashes,config=read(ROOT/f'configs/batch{32 if step<=75000 else 512}.json'),
            source_hashes=read(ROOT/'SOURCE_HASHES.json'),exposures=min(step,75000)*32+max(step-75000,0)*512)
        torch.save(payload,tmp);tmp.replace(path)
        write(ROOT/'LATEST.json',dict(step=step,checkpoint=str(path),exposures=payload['exposures'],complete_Adam=True,
            rank_states_equal=True,four_rank_RNG=True,scheduler_completed=opt.training_schedule['completed'],time=time.time()))
    parallel.barrier()
