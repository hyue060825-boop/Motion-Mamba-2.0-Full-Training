"""Physical-time interpolation. Never interpolate motion features or Euler values."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def time_grid(timestamps, start=None, end=None, fps=25.):
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('INVALID_TIMESTAMPS')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('INVALID_FPS')
    start = float(t[0]) if start is None else float(start)
    end = np.nextafter(t[-1], np.inf) if end is None else float(end)
    if not np.isfinite([start, end]).all() or start < t[0] or end <= start:
        raise ValueError('INVALID_INTERVAL')
    count = max(0, int(np.floor((min(t[-1], end) - start) * fps + 1e-10)) + 1)
    out = start + np.arange(count, dtype=np.float64) / fps
    out = out[(out < end) & (out <= t[-1] + 1e-12)]
    # Tiny arithmetic rounding is permitted, extrapolation is not.
    out[out > t[-1]] = t[-1]
    if not len(out):
        raise ValueError('EMPTY_INTERVAL')
    right = np.searchsorted(t, out, side='left').clip(0, len(t)-1)
    left = np.maximum(right-1, 0)
    width = t[right] - t[left]
    weight = np.divide(out-t[left], width, out=np.zeros_like(out), where=width > 0)
    return out, {'left': left.tolist(), 'right': right.tolist(), 'weight': weight.tolist(),
                 'source_timestamps': t.tolist(), 'target_timestamps': out.tolist(),
                 'fps': fps, 'interval': [start, end], 'interval_convention': '[start,end)',
                 'filter': None, 'filter_reason': 'no filter selected before development audit'}

def linear(values, mapping):
    v = np.asarray(values)
    a, b = np.asarray(mapping['left']), np.asarray(mapping['right'])
    w = np.asarray(mapping['weight']).reshape((-1,) + (1,) * (v.ndim-1))
    return v[a] * (1-w) + v[b] * w

def rotations(local_matrices, mapping):
    matrices = np.asarray(local_matrices)
    if matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError('INVALID_ROTATIONS')
    if np.max(np.abs(matrices @ matrices.swapaxes(-1,-2) - np.eye(3))) > 1e-5 or np.min(np.linalg.det(matrices)) < .99999:
        raise ValueError('NON_SO3_ROTATIONS')
    t, out = mapping['source_timestamps'], mapping['target_timestamps']
    if len(t) == 1:
        return matrices[:1].copy()
    flat = matrices.reshape(len(t), -1, 3, 3)
    result = np.stack([Slerp(t, Rotation.from_matrix(flat[:, j]))(out).as_matrix() for j in range(flat.shape[1])], axis=1)
    return result.reshape((len(out),) + matrices.shape[1:])

def qpos(q, timestamps, ranges, start=None, end=None, fps=25.):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 36 or len(q) != len(timestamps) or not np.isfinite(q).all():
        raise ValueError('INVALID_QPOS')
    if np.max(np.abs(np.linalg.norm(q[:,3:7], axis=-1)-1)) > 1e-5:
        raise ValueError('NON_UNIT_QUATERNION')
    ranges = np.asarray(ranges)
    if ((q[:,7:] < ranges[:,0]-1e-4) | (q[:,7:] > ranges[:,1]+1e-4)).any():
        raise ValueError('SOURCE_HINGE_LIMIT')
    out, mapping = time_grid(timestamps, start, end, fps)
    rot = rotations(Rotation.from_quat(q[:,3:7], scalar_first=True).as_matrix(), mapping)
    result = np.concatenate([linear(q[:,:3],mapping), Rotation.from_matrix(rot).as_quat(scalar_first=True), linear(q[:,7:],mapping)],axis=1)
    mapping['resample_count'] = int(not np.array_equal(np.asarray(timestamps), out))
    return result, out, mapping
