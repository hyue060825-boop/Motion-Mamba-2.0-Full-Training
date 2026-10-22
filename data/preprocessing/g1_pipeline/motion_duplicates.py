"""Canonical physical motion fingerprints; candidate matches require provenance review."""
import json
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from .common import ROOT,write_json,write_jsonl,sha,digest


def physical_descriptor(q):
    q=np.asarray(q,float)
    if q.ndim!=2 or q.shape[1]!=36 or len(q)<3 or not np.isfinite(q).all():raise ValueError('DUPLICATE_FINGERPRINT_QPOS')
    rotation=R.from_quat(q[:,3:7],scalar_first=True);forward=rotation[0].apply([1.,0,0]);yaw=np.arctan2(forward[1],forward[0]);turn=R.from_euler('z',-yaw)
    root=q[:,:3].copy();root[:,:2]-=root[0,:2];root=turn.apply(root)
    orientations=(turn*rotation).as_matrix()[...,:2].reshape(len(q),6)
    # Descriptor sampling supports discovery only; no production motion is
    # resampled, time-warped, or deleted based on this fingerprint.
    t=np.linspace(0,len(q)-1,32);original=np.arange(len(q));features=np.c_[root,orientations,q[:,7:]]
    sampled=np.stack([np.interp(t,original,features[:,i]) for i in range(features.shape[1])],axis=1)
    return sampled.ravel(),{'frames':len(q),'duration_s':(len(q)-1)/25,'canonicalization':'one initial planar origin/heading; absolute root height retained',
        'descriptor_sha256':digest(np.round(sampled,6).tolist())}


def compare(rows,out,*,candidate_mean_coordinate_distance=.04):
    """rows contain sample_id/dataset/family/project_split/qpos_path + source hashes."""
    desc=[];meta=[];exact=defaultdict(list);family=defaultdict(list);sourcehash=defaultdict(list)
    for row in rows:
        path=ROOT/row['qpos_path'];q=np.load(path);vector,info=physical_descriptor(q);desc.append(vector);meta.append(info)
        exact[sha(path)].append(row['sample_id']);family[row['source_family_id']].append(row['sample_id'])
        if row.get('source_sha256'):sourcehash[row['source_sha256']].append(row['sample_id'])
    vectors=np.asarray(desc);pairs=[]
    if len(vectors):
        # L2 sphere bounds coordinate RMS; duration match prevents long/short
        # variants becoming unquestioned duplicates after descriptor sampling.
        tree=cKDTree(vectors);radius=candidate_mean_coordinate_distance*np.sqrt(vectors.shape[1])
        for i,j in sorted(tree.query_pairs(radius)):
            a,b=rows[i],rows[j];da,db=meta[i]['duration_s'],meta[j]['duration_s']
            if abs(da-db)>max(.08,.02*max(da,db)):continue
            pairs.append({'sample_a':a['sample_id'],'sample_b':b['sample_id'],'dataset_a':a['dataset'],'dataset_b':b['dataset'],
                'split_a':a.get('project_split'),'split_b':b.get('project_split'),'same_family':a['source_family_id']==b['source_family_id'],
                'descriptor_coordinate_rms':float(np.sqrt(np.mean((vectors[i]-vectors[j])**2))),
                'duration_difference_s':abs(da-db),'status':'REVIEW_REQUIRED_NOT_AUTOMATIC_EXCLUSION'})
    report={'status':'PHYSICAL_DUPLICATE_CANDIDATES_REQUIRE_REVIEW','records':len(rows),'descriptor_info':dict(zip([r['sample_id'] for r in rows],meta)),
        'exact_qpos_file_groups':[{'sha256':k,'samples':v} for k,v in exact.items() if len(v)>1],
        'same_source_hash_groups':[{'sha256':k,'samples':v} for k,v in sourcehash.items() if len(v)>1],
        'same_family_groups':[{'family':k,'samples':v} for k,v in family.items() if len(v)>1],
        'near_duplicate_candidates':pairs,'cross_source_identity_proven':False,'automatic_deletion':False,
        'descriptor_limitations':'32-time-sample coordinate RMS over root/rotation columns/hinges is a search heuristic, not proof; no DTW used; mirror family is identified from lineage',
        'input_bindings':{str(ROOT/r['qpos_path']):sha(ROOT/r['qpos_path']) for r in rows},'code_sha256':sha(__file__)}
    write_json(out,report);return report
