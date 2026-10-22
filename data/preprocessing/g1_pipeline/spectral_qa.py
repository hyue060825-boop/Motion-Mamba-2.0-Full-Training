"""Native-rate spectral evidence before 25Hz resampling; no implicit filtering."""
import numpy as np
from scipy.signal import welch,detrend
from scipy.spatial.transform import Rotation as R


def velocity_spectrum(values,fps,*,target_fps=25.,rms_floor=.1,review_fraction=.05):
    values=np.asarray(values,float)
    if values.ndim!=2 or not np.isfinite(values).all() or fps<=0:raise ValueError('INVALID_SPECTRAL_INPUT')
    if len(values)<8:return {'status':'UNSUPPORTED_SHORT_SPECTRUM','reason':'fewer than8 velocity samples','review_required':False,'measured':False}
    if fps<=target_fps:return {'status':'SOURCE_AT_OR_BELOW_TARGET_RATE','source_fps':fps,'review_required':False,'measured':False,
        'reason':'native Nyquist does not exceed target Nyquist; no new high-band alias introduced by downsampling'}
    x=detrend(values,axis=0,type='linear');frequency,power=welch(x,fs=fps,nperseg=min(len(x),256),axis=0,detrend=False)
    total=power.sum(axis=0);high=power[frequency>target_fps/2].sum(axis=0)
    high_mask=frequency>target_fps/2
    frequency_step=frequency[1]-frequency[0]
    displacement=np.sqrt(np.sum(power[high_mask]/(2*np.pi*frequency[high_mask,None])**2,axis=0)*frequency_step)
    fraction=np.divide(high,total,out=np.zeros_like(high),where=total>1e-20);rms=np.sqrt(np.mean(values**2,axis=0))
    flagged=(fraction>review_fraction)&(rms>rms_floor)
    return {'status':'NATIVE_SPECTRUM_MEASURED','measured':True,'source_fps':fps,'target_nyquist_hz':target_fps/2,
        'velocity_samples':len(values),'welch_segment_samples':min(len(x),256),'high_band_power_fraction':fraction.tolist(),
        'velocity_rms':rms.tolist(),'max_high_band_fraction':float(fraction.max()),'flagged_coordinates':np.flatnonzero(flagged).tolist(),
        'high_band_displacement_rms':displacement.tolist(),'high_band_displacement_rms_max':float(displacement.max()),
        'displacement_definition':'sqrt(integral PSD_velocity(f)/(2*pi*f)^2 df over f>12.5Hz); radians for angular velocity, meters for translation',
        'review_required':bool(flagged.any()),'review_rule':{'high_band_power_fraction_above':review_fraction,'velocity_rms_above':rms_floor},
        'filter_applied':False,'note':'SLERP is interpolation, not an anti-alias low-pass filter; spectral flags require content/continuity review, not per-clip parameter fitting'}


def local_rotation_spectrum(local_rotations,fps):
    rotations=np.asarray(local_rotations)
    if rotations.ndim!=4 or rotations.shape[-2:]!=(3,3):raise ValueError('LOCAL_ROTATION_SPECTRUM_SHAPE')
    relative=rotations[:-1].swapaxes(-1,-2)@rotations[1:]
    velocity=R.from_matrix(relative.reshape(-1,3,3)).as_rotvec().reshape(len(relative),-1)*fps
    return velocity_spectrum(velocity,fps)
