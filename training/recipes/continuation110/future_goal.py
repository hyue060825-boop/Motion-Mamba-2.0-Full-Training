"""Isolated timed-goal collection and training; official constraint encoding."""
from common import *
import math, types
import numpy as np
import torch
from motion_mamba2.constraints import Root2DConstraintSet
from motion_mamba2.models.motion_model import translate_normalized_root_motion, get_three_mask_from_len
from g1_motion_contract import CONTRACT, decode_committed_window, decode_training_prediction
from motion_training import runtime, sampling, history
from motion_training.schema import validate_record

VERSION = 'timed_root_waypoint_248_v1'

def observation(model, goal, relative_frame, extent=None):
    assert type(relative_frame) is int and 0 <= relative_frame < 248
    n = max(52, 4 * math.ceil((relative_frame + 1) / 4)) if extent is None else extent
    assert 52 <= n <= 248 and n % 4 == 0 and relative_frame < n
    constraint = Root2DConstraintSet(model.skeleton,
        torch.tensor([relative_frame], device=model.device),
        torch.tensor([goal], device=model.device, dtype=torch.float32))
    obs, mask = model.motion_rep.create_conditions_from_constraints(
        [constraint], length=n, to_normalize=False, device=model.device)
    return (model.motion_rep.normalize(obs) * mask)[None], mask[None].bool()

def official_proposal(official, probe, runner, noise, goal=None, relative_frame=None):
    """Only G is decoded; future placeholders never become decoder frames."""
    o, m = official, runner.m
    if goal is None:
        obs = torch.zeros(1,52,414,device=o.device)
        mask = torch.zeros_like(obs,dtype=torch.bool)
    else:
        obs, mask = observation(o, goal, relative_frame)
    n = obs.shape[1]
    h = min(196, 248-n, runner.physical.shape[1])
    h -= h % 4
    if h:
        tokens, origin, heading = o._encode_init_history(
            o.motion_rep.normalize(runner.physical[:,-h:]), 1)
        assert float((origin-runner.translation).abs().max()) < 1e-5
    else:
        tokens, origin, heading = None, runner.translation.clone(), runner.heading.clone()
    assert h+n <= 248
    obs = torch.cat([torch.zeros(1,h,414,device=o.device),obs],1)
    mask = torch.cat([torch.zeros(1,h,414,device=o.device,dtype=torch.bool),mask],1)
    o.diffusion.calc_diffusion_vars(o.diffusion.space_timesteps(10)[0])
    probe.begin(noise)
    seq = o._generate_window(tokens,origin,0,h,h+n,runner.text,runner.mask,heading,
        mask,obs,torch.tensor([10],device=o.device),(2.,2.),list(range(9,-1,-1)),
        progress_bar=lambda x:x,target_motion=None,cfg_type=None)
    current = seq[:,h//4:h//4+13]
    assert current.shape[1] == 13
    world = sampling.world_motion(o,current,origin)
    root = world[:,:,:20].reshape(1,13,4,5)
    a,b = o.motion_rep.global_root_stats,m.motion_rep.global_root_stats
    world = world.clone()
    world[:,:,:20] = ((root*a.std_eps.to(root)+a.mean.to(root)-b.mean.to(root))
                          /b.std_eps.to(root)).reshape(1,13,20)
    decoded,_ = decode_committed_window(m,world,runner.previous)
    _,q = sampling.measure(m,decoded)
    return q,decoded,world,dict(H=h,G=52,F=n-52,relative_goal=relative_frame,
                               goal_visible=goal is not None,official_extent=h+n)

def shadow_of(runner):
    return types.SimpleNamespace(m=runner.m,physical=runner.physical.clone(),
        previous=runner.previous.clone(),translation=runner.translation.clone(),
        heading=runner.heading.clone(),text=runner.text.clone(),mask=runner.mask.clone(),i=runner.i)

def shadow_accept(shadow, decoded, world):
    ph = shadow.m.motion_rep.unnormalize(decoded)
    shadow.physical = torch.cat([shadow.physical,ph],1)
    shadow.previous = world.clone()
    shadow.translation = torch.zeros_like(shadow.translation)
    shadow.translation[:,0] = ph[:,-1,0]
    shadow.translation[:,2] = ph[:,-1,2]
    shadow.heading = torch.atan2(ph[:,-1,4],ph[:,-1,3])
    shadow.i += 1

def teacher_batch(m,c,r,cfg,old_cfg):
    if r.get('pool_kind') != 'future':
        return history.teacher_batch(m,c,r,cfg,old_cfg)
    validate_record(r)
    from speed_backend import packed_item, diagnostic_identity
    item = packed_item(r['label'],m.device)
    context=history.HistoryContext(item['chunks'],item['physical'],r['collector_checkpoint_sha256'])
    # Reference and canonical are selected by the checked training entry point.
    # Both far and arrival histories now use the admitted current-weight path.
    receipt=context.prime(m,c,item['target']['text'])
    receipt['history_backend']='selected_and_admitted_history_path'
    target = item['target']['target_latent']
    goal = item['target']['future_goal']
    obs,mask = observation(m,goal['xz'],goal['relative_frame'])
    root = translate_normalized_root_motion(m.motion_rep.extract_root(obs),-item['translation'],m.motion_rep)
    obs = m.motion_rep.concat_root_body(root,m.motion_rep.extract_body(obs))*mask
    before = bool(mask[:,52:].any())
    rng = np.random.default_rng(r['seed'])
    td = bool(rng.random()<cfg['text_dropout']); cd = bool(rng.random()<cfg['constraint_dropout'])
    text = item['target']['text'].clone()
    if td: text.zero_()
    if cd: obs.zero_(); mask.zero_()
    n=obs.shape[1];z=torch.zeros(1,device=m.device,dtype=torch.long)
    hl=z;gl=z+52;fl=z+(n-52 if bool(mask[:,52:].any()) else 0)
    hm,gm,fm=get_three_mask_from_len(hl,gl,z+n-52,n,m.device)
    fm &= bool(mask[:,52:].any())
    htm,gtm,ftm=m.hybrid.convert_frame_mask_to_token_mask(hm,gm,fm,mask)
    gen=torch.Generator(device=m.device).manual_seed(r['seed'])
    t=torch.randint(m.diffusion.num_base_steps,(1,),generator=gen,device=m.device)
    noise=torch.randn(target.shape,generator=gen,device=m.device,dtype=target.dtype)
    clean=target.new_zeros(1,n//4,148);clean[:,:13]=target
    m.diffusion.calc_diffusion_vars(torch.arange(m.diffusion.num_base_steps,device=m.device))
    x=clean.clone();x[:,:13]=m.diffusion.q_sample(target,t,noise)
    valid=hm|gm
    f=dict(x=x,target=clean,decoder_history_hybrid=item['target']['decoder_prefix'],
        history_len=hl,generation_len=gl,future_len=fl,history_mask=hm,generation_mask=gm,
        future_mask=fm,history_token_mask=htm,generation_token_mask=gtm,future_token_mask=ftm,
        text=text,text_mask=torch.tensor([[not td]],device=m.device),heading=item['heading'],
        motion_mask=mask.float(),observed=obs,pad_mask=valid,lengths=gl,timestep=t,noise=noise,
        generated_history_state_ready=True,sequential_cache=True,motion_contract=CONTRACT,
        hybrid_converter=m.hybrid)
    # The codec's output extent follows the supplied token tensor, even when
    # valid lengths are shorter. Never pass future placeholders to this decode.
    validG=torch.ones(1,52,device=m.device,dtype=torch.bool)
    decode_fixed=dict(motion_contract=CONTRACT,hybrid_converter=m.hybrid,
        history_mask=~validG,generation_mask=validG,
        decoder_history_hybrid=item['target']['decoder_prefix'])
    with torch.no_grad():f['supervised_motion']=decode_training_prediction(target,decode_fixed).detach()
    assert f['supervised_motion'].shape[1]==52
    assert int(gm.sum())==52 and int(gtm.sum())==13
    assert bool(ftm.any()) == bool(mask[:,52:].any())
    if not cd and goal['relative_frame']>=52:
        assert int(mask.sum())==2 and bool(ftm[0,goal['relative_frame']//4])
    return f,dict(r,history=receipt,F_before_drop=before,F_visible_after_drop=bool(ftm.any()),
        effective_G_frames=52,F_frames=n-52,goal_relative_frame=goal['relative_frame'],
        observed_scalars=int(mask.sum()),text_dropped=td,condition_dropped=cd,
        future_target_latents_zero=bool((clean[:,13:]==0).all()),
        input_hash=diagnostic_identity([x,text,obs,mask]),noise_hash=diagnostic_identity(noise))
