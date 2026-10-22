"""Pose-row edits preserve per-frame FK, refresh velocities, disclose contacts."""
import numpy as np
from .common import identity
from .perturbations import refresh_velocities

EDITS={'static':1,'repeat4':4,'repeat13':13,'shuffle4':4,'shuffle52':52,'reverse':1,'seam':1}

def edit(a,kind,sample_id):
    n=len(a);indices=np.arange(n);rng=np.random.default_rng(int(identity(['v2-edit',kind,sample_id])[:16],16))
    if kind=='static':indices[:]=n//2
    elif kind.startswith('repeat'):indices%=int(kind[6:])
    elif kind.startswith('shuffle'):
        width=int(kind[7:])
        for start in range(0,n,width):indices[start:start+width]=rng.permutation(indices[start:start+width])
    elif kind=='reverse':indices=indices[::-1]
    elif kind=='seam':indices[n//2:]=indices[n//2:][::-1]
    else:raise ValueError(kind)
    return refresh_velocities(a[indices])

def bad_data(a,kind):
    b=np.array(a,copy=True)
    if kind=='nan':b[0,0]=np.nan
    elif kind=='channel_shuffle':b=b[:,np.random.default_rng(4).permutation(414)]
    elif kind=='velocity_inconsistent':b[:,308:410]+=10
    elif kind=='invalid_rotation6d':b[:,104:308]=0
    elif kind=='wrong_fps':return b,20
    else:raise ValueError(kind)
    return b,25
