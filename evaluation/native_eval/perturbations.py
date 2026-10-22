"""Controlled temporal edits preserve physical position/velocity consistency."""
import numpy as np

def refresh_velocities(a):
    a=np.array(a,copy=True)
    positions=np.concatenate((a[:,:3,None].transpose(0,2,1),a[:,5:104].reshape(-1,33,3).copy()),axis=1)
    positions[:,1:,0]+=a[:,None,0];positions[:,1:,2]+=a[:,None,2]
    velocity=np.zeros_like(positions)
    if len(a)>1:
        velocity[:-1]=np.diff(positions,axis=0)*25
        velocity[-1]=velocity[-2]
    a[:,308:410]=velocity.reshape(len(a),102)
    return a

def reverse_motion(a):
    return refresh_velocities(a[::-1])

def static_motion(a,severity=1.):
    return refresh_velocities((1-severity)*a+severity*np.repeat(a[:1],len(a),0))
