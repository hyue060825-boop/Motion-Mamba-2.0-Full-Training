import math
import numpy as np
import torch
from motion_training import sampling
def _official_proposal(official,noise_probe,runner,noise,control):
    """Generate a proposal without committing anything to the student."""
    from g1_motion_contract import decode_committed_window
    from g1_online_controls import conditions
    o=official;m=runner.m;physical=runner.physical[:,-196:]
    history,origin,heading=o._encode_init_history(o.motion_rep.normalize(physical),1)
    assert float((origin-runner.translation).abs().max())<1e-5
    n=physical.shape[1];obs,mask=conditions(o,control,origin)
    obs=torch.cat((torch.zeros(1,n,414,device=m.device),obs),1)
    mask=torch.cat((torch.zeros(1,n,414,device=m.device,dtype=torch.bool),mask),1)
    o.diffusion.calc_diffusion_vars(o.diffusion.space_timesteps(10)[0]);noise_probe.begin(noise)
    seq=o._generate_window(history,origin,0,n,n+52,runner.text,runner.mask,heading,mask,obs,
         torch.tensor([10],device=m.device),(2.,2.),list(range(9,-1,-1)),progress_bar=lambda x:x,target_motion=None,cfg_type=None)
    world=sampling.world_motion(o,seq[:,-13:],origin);shape=world[:,:,:20].shape
    root=world[:,:,:20].reshape(1,13,4,5);a=o.motion_rep.global_root_stats;b=m.motion_rep.global_root_stats
    world=world.clone();world[:,:,:20]=((root*a.std_eps.to(world.device)+a.mean.to(world.device)-b.mean.to(world.device))/b.std_eps.to(world.device)).reshape(shape)
    dec,_=decode_committed_window(m,world,runner.previous)
    _,q=sampling.measure(m,dec)
    return q,dec,world
def phase_quality(q, ph, windows, action, kind, j, heading, controls):
    """Two-window response checks at transitions; one-window stage checks later."""
    if j == 0 and kind != 'continue':
        score = event_score(np.concatenate(q), np.concatenate(ph), windows,
                            action, kind, heading, controls)
        return score['function_pass'], score
    physical = ph[0]
    direction = np.array([math.sin(heading), math.cos(heading)])
    progress = float((physical[-1, [0, 2]] - physical[0, [0, 2]]) @ direction)
    if action == 'backwalk':
        progress *= -1
    legs = float(np.rad2deg(np.ptp(q[0][:, 7:19], axis=0)).mean())
    speed = np.linalg.norm(np.diff(physical[:, [0, 2]], axis=0), axis=1) * 25
    checks = {'actual_controls': all(w['control_pass'] for w in windows)}
    if kind in ('stop', 'release_hold'):
        checks['hold_after_buffer'] = bool((speed[-25:] if j==1 else speed).max() <= .05)
    elif kind == 'multi':
        checks['target_or_active_legs'] = windows[0]['path_m'] <= .2 and (progress <= .1 or legs >= 5)
    elif action in ('walk', 'backwalk'):
        checks['progress_and_legs'] = progress > .05 and legs >= 5
    elif action == 'wave':
        r = np.rad2deg(np.ptp(q[0][:, 7:], axis=0))
        checks['both_arms_execute'] = bool(r[15:22].max() >= 30 and r[22:29].max() >= 30)
    elif action == 'turn':
        target = controls[0].get('heading', heading)
        requested = math.atan2(math.sin(target-heading), math.cos(target-heading))
        actual_h = math.atan2(physical[-1, 4], physical[-1, 3])
        actual = math.atan2(math.sin(actual_h-heading), math.cos(actual_h-heading))
        checks['actual_turn_response'] = abs(requested) < 1e-3 or actual*np.sign(requested) >= .5*abs(requested)
    checks = {k: bool(v) for k, v in checks.items()}
    return all(checks.values()), dict(checks=checks, progress_m=progress, leg_range_deg=legs)

def base_record(spec, path, parent_hash, model_hash, key, action, kind, j, seed, generation):
    return dict(schema=VERSION, id=key, source_family_id=spec['source_family_id'],
        dataset=spec['dataset'], split='TRAIN', crop_start=spec['crop_start'],
        crop_end=spec['crop_start']+104, feature_path=spec['feature_path'], fps=25,
        skeleton='g1', representation='physical414', codec_version='g1_history_codec_104_v1',
        normalizer_version='g1_committed_v3', H=0, G=52, F=0,
        supervision='same_state_teacher', label=str(path),
        collector_checkpoint_sha256=parent_hash, teacher_model_hash=model_hash,
        teacher_extra_conditions=[], generation=generation, group=spec['group'],
        action=action, kind=kind, phase=phase(action,kind,j),
        stage_key=stage_key(spec['group'],action,kind,j), relative_event_window=j,
        seed=seed, history_producer='student_after_gt_initialization',
        teacher_submitted_to_student=False, qualified=False)
from motion_training._compat.event_scores import event_score, phase
from motion_training.collectors import stage_key
from motion_training.schema import VERSION
