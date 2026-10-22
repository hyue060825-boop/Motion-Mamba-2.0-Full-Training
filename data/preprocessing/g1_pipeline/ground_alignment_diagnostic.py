"""Diagnostic-only ground alignment; never mutates production qpos."""
import glob, json
from pathlib import Path
import numpy as np
import mujoco
from .common import ROOT, write_json

def run(limit_clips=10, stride=20):
    model = mujoco.MjModel.from_xml_path(str(ROOT/'deps/GMR/assets/unitree_g1/g1_mocap_29dof.xml'))
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    out = {}
    for dataset in ('hml_smoke_qpos', 'hiphi_smoke_qpos'):
        data = mujoco.MjData(model); files = sorted((ROOT/'work'/dataset).glob('*.npy'))[:limit_clips]
        before_floor = before_self = after_floor = after_self = frames = 0; shifts=[]
        for path in files:
            for q in np.load(path)[::stride]:
                def counts(x):
                    data.qpos[:] = x; mujoco.mj_forward(model, data)
                    floor_min = 0.0; has_floor = False; self_bad = False
                    for i in range(data.ncon):
                        c = data.contact[i]
                        if c.geom1 == floor or c.geom2 == floor:
                            has_floor = True; floor_min = min(floor_min, float(c.dist))
                        elif c.dist < -0.05: self_bad = True
                    return int(has_floor and floor_min < 0), int(self_bad), floor_min
                bf, bs, floor_min = counts(q)
                aligned = q.copy(); shift = max(0.0, -floor_min) + 0.005
                aligned[2] += shift
                af, ass, _ = counts(aligned)
                before_floor += bf; before_self += bs; after_floor += af; after_self += ass
                shifts.append(shift); frames += 1
        out[dataset] = {'clips': len(files), 'sampled_frames': frames,
            'before_floor_bad': before_floor, 'after_floor_bad': after_floor,
            'before_self_bad': before_self, 'after_self_bad': after_self,
            'root_shift_m_percentiles': np.percentile(shifts,[50,90,99]).tolist() if shifts else None,
            'strategy': 'diagnostic per-frame root z lift to clear floor plus 5 mm; production qpos unchanged'}
    write_json(ROOT/'qa/aggregate/ground_alignment_diagnostic.json', out)
    return out

if __name__ == '__main__': print(json.dumps(run(), indent=2))
