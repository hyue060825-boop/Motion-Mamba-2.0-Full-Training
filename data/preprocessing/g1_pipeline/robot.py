import csv
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .common import XML, MESHES, GOLDEN, ROOT, sha, digest, write_json, atomic_bytes, setup_imports

FORMAT = 'bones_g1_frame_cm_euler_xyz_deg_hinge_deg_v1'
BASIS = np.array([[0.,1,0],[0,0,1],[1,0,0]])

def compile_target():
    import mujoco
    tree = ET.parse(XML)
    assets = {}
    for mesh in tree.findall('.//asset/mesh'):
        p = MESHES / mesh.attrib['file']
        if not p.is_file():
            raise FileNotFoundError(f'MISSING_TARGET_MESH: {p}')
        assets[str(p)] = sha(p)
    # Only path resolution changes. All bodies, inertials and collision geoms remain exact.
    tree.find('compiler').set('meshdir', str(MESHES))
    source = ET.tostring(tree.getroot(), encoding='unicode')
    model = mujoco.MjModel.from_xml_string(source)
    if (model.nq,model.nv) != (36,35) or list(model.jnt_type).count(mujoco.mjtJoint.mjJNT_HINGE) != 29:
        raise ValueError('TARGET_DOF_MISMATCH')
    return model, {'xml':str(XML),'xml_hash':sha(XML),'resolved_meshes':assets,'asset_hash':digest({'xml':sha(XML),'meshes':assets})}

def header():
    with GOLDEN.open(newline='') as f:
        return next(csv.reader(f))

def read_csv(path, *, format_id, fps=None):
    if format_id != FORMAT:
        raise ValueError('AMBIGUOUS_OR_UNSUPPORTED_CSV_FORMAT')
    path = Path(path)
    with path.open(newline='') as f:
        rows = csv.reader(f)
        if next(rows) != header():
            raise ValueError('GOLDEN_HEADER_MISMATCH')
        records = list(rows)
    if any(len(row) != 36 for row in records):
        raise ValueError('INVALID_CSV_ROWS_OR_BLANK_LINE')
    a = np.asarray(records, dtype=np.float64)
    if not len(a) or not np.isfinite(a).all() or not np.array_equal(a[:,0], np.arange(len(a))):
        raise ValueError('INVALID_CSV_FRAME_INDEX_OR_FINITE')
    sidecar = path.with_suffix('.json')
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text())
        if meta['format_id'] != FORMAT or meta['csv_sha256'] != sha(path) or meta['frames'] != len(a) or meta['schema_hash'] != digest(header()):
            raise ValueError('SIDECAR_MISMATCH')
        if fps is not None and fps != meta['fps']:
            raise ValueError('FPS_CONFLICT')
        fps = meta['fps']
        times = np.asarray(meta['timestamps'])
        if times.shape != (len(a),) or not np.isfinite(times).all() or not np.allclose(np.diff(times),1/fps,atol=1e-6,rtol=0):
            raise ValueError('SIDECAR_TIME_MISMATCH')
    else:
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError('MISSING_FPS_METADATA')
        times = np.arange(len(a))/fps
    q = np.concatenate([a[:,1:4]*.01,Rotation.from_euler('xyz',a[:,4:7],degrees=True).as_quat(scalar_first=True),np.deg2rad(a[:,7:])],axis=1)
    return q, times, float(fps)

def export_csv(path, q, timestamps, provenance):
    q, t = np.asarray(q), np.asarray(timestamps)
    if q.shape != (len(t),36) or not len(t) or not np.isfinite(q).all() or not np.isfinite(t).all():
        raise ValueError('INVALID_EXPORT')
    if not np.allclose(np.diff(t),.04,atol=1e-6,rtol=0) or np.max(np.abs(np.linalg.norm(q[:,3:7],axis=1)-1)) > 1e-5:
        raise ValueError('INVALID_EXPORT_TIME_OR_QUATERNION')
    values = np.concatenate([np.arange(len(q))[:,None],100*q[:,:3],Rotation.from_quat(q[:,3:7],scalar_first=True).as_euler('xyz',degrees=True),np.rad2deg(q[:,7:])],axis=1)
    out = io.StringIO()
    np.savetxt(out, values, delimiter=',', header=','.join(header()), comments='', fmt=['%d']+['%.12g']*35)
    atomic_bytes(path,out.getvalue().encode())
    write_json(Path(path).with_suffix('.json'),{'format_id':FORMAT,'fps':25,'frames':len(q),'timestamps':t.tolist(),'csv_sha256':sha(path),'schema_hash':digest(header()),'provenance':provenance})

def official_load(path):
    setup_imports()
    from motion_mamba2.data_processing.skeleton import load_csv_g1_animation
    from motion_mamba2.skeleton import G1Skeleton34
    # The official parser requires names/bones only, not a distinct physical skeleton.
    skeleton = G1Skeleton34(load=True)
    class Names:
        bones = skeleton.bone_order_names
        def get_bones_names(self): return self.bones
    p, r = load_csv_g1_animation(str(path), Names(), csv_format='retargeted_from_gmr',xml_path=str(XML))
    return p * .01, r, skeleton

def encode_csv(path, *, canonicalize=False):
    import torch
    from motion_mamba2.representation import MotionMamba2MotionRep
    # Enforce sidecar FPS; do not route a 25 Hz export into legacy 120 Hz converter.
    _, t, fps = read_csv(path, format_id=FORMAT)
    if fps != 25 or len(t)<2:
        raise ValueError('FEATURES_REQUIRE_25FPS_AND_TWO_FRAMES')
    p,r,skeleton = official_load(path)
    rep = MotionMamba2MotionRep(skeleton, fps=25)
    features = rep(torch.from_numpy(r).float()[None],torch.from_numpy(p).float()[None],to_normalize=False,to_canonicalize=False)
    transform = {'heading':0.,'planar_translation':[0.,0.], 'applied':False}
    if canonicalize:
        angle = torch.atan2(features[0,0,4],features[0,0,3])
        features = rep.rotate_to(features,torch.tensor(0.))
        first = features[:,0,[0,2]].clone()
        features = rep.translate_2d(features,-first)
        transform = {'heading':float(angle),'planar_translation':first[0].tolist(),'applied':True}
    if features.shape[-1] != 414:
        raise ValueError('NON_G1_REPRESENTATION')
    return features, rep, transform
