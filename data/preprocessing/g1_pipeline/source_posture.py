"""Human-only support-posture evidence, independent of robot QA or solver error."""
import numpy as np


def support_evidence(hips,chest,knees,feet,height_m,dt=.04):
    hips,chest,knees,feet=map(np.asarray,(hips,chest,knees,feet))
    n=len(hips)
    if hips.shape!=(n,3) or chest.shape!=(n,3) or knees.shape!=(n,2,3) or feet.shape!=(n,2,3):
        raise ValueError('INVALID_POSTURE_LANDMARK_SHAPES')
    if not all(np.isfinite(x).all() for x in (hips,chest,knees,feet)) or height_m<=0 or dt<=0 or n<1:
        raise ValueError('INVALID_POSTURE_INPUT')
    torso=chest-hips;length=np.linalg.norm(torso,axis=1)
    if np.any(length<1e-6):raise ValueError('DEGENERATE_HUMAN_TORSO')
    relative_height=(hips[:,2]-feet[:,:,2].min(axis=1))/height_m
    knee_elevation=(knees[:,:,2].max(axis=1)-hips[:,2])/height_m
    up_cosine=torso[:,2]/length
    low=relative_height<.2
    evidence=low&((knee_elevation>.025)|(up_cosine<.5))
    # Require both meaningful coverage and a continuous interval, avoiding a
    # single bent-torso transition being called a new supported scene.
    longest=current=0
    for value in evidence:
        current=current+1 if value else 0;longest=max(longest,current)
    indicated=bool(evidence.mean()>=.1 and longest*dt>=.2)
    return {'ground_multisupport_review':indicated,'flagged_frame_fraction':float(evidence.mean()),
            'longest_flagged_interval_s':longest*dt,'pelvis_above_lowest_foot_height_ratio_min':float(relative_height.min()),
            'knee_above_pelvis_height_ratio_max':float(knee_elevation.max()),
            'torso_up_cosine_min':float(up_cosine.min()),'frames':n,
            'rule':'pelvis above lowest foot <0.20 body heights AND (knee above pelvis >0.025 body heights OR torso up cosine <0.5), coverage >=10%, continuous interval >=0.2s',
            'interpretation':'human posture indicates non-standing support review; does not prove contacts, floor height or quality; no robot outputs used'}
