"""Machine-readable training release validation; never promotes missing evidence."""
from __future__ import annotations
import json
import math
import numpy as np
from pathlib import Path
from .common import ROOT,write_json,read_jsonl,sha,atomic_bytes
from .gates import require_pass,gate,fingerprint

REQUIRED=['00_preflight','01_bones_calibration','02_hml100','03_hiphi100','04_fusion_qa','05_full_batch']

SOURCES={'bones','humanml3d','hiphi'}
MODEL_CONTRACT={'skeleton':'G1Skeleton34','nbjoints':34,'dof':29,'representation':414,'fps':25}


def runtime_evidence(root,expected_fingerprint):
    """Validate materialized training smoke evidence, never just a PASS string.

    The evidence producer must run the actual project loader/model and bind its
    dependencies. Missing runtime evidence is a blocker even when all upstream
    stages pass. This function does not itself run formal training.
    """
    path=root/'qa/aggregate/final_runtime_acceptance.json'
    evidence=json.loads(path.read_text())
    if evidence.get('schema_version')!=1:raise ValueError('UNKNOWN_RUNTIME_EVIDENCE_SCHEMA')
    if evidence.get('fingerprint')!=expected_fingerprint:raise ValueError('RUNTIME_EVIDENCE_STALE')
    if evidence.get('model_contract')!=MODEL_CONTRACT:raise ValueError('G1_MODEL_CONTRACT_MISMATCH')
    bindings=evidence.get('input_hashes',{})
    required={'manifests/accepted_train.jsonl','manifests/accepted_val.jsonl','manifests/accepted_test.jsonl',
              'contracts/motion_mamba2_414.json','contracts/g1_robot.json','contracts/g1_skeleton34.json',
              'contracts/bones_g1_csv.json','stats/physical_414.json','reports/leakage_audit.md'}
    if not required<=set(bindings):raise ValueError('RUNTIME_EVIDENCE_MISSING_INPUT_BINDINGS')
    for name,expected in bindings.items():
        p=root/name
        if not p.is_file() or sha(p)!=expected:raise ValueError('RUNTIME_INPUT_STALE:'+name)
    for key in ('batch_artifact','checkpoint_metadata','checkpoint_artifact','normalizer_artifact'):
        name=evidence.get(key)
        if not name or name not in bindings:raise ValueError('UNBOUND_RUNTIME_ARTIFACT:'+key)
    batch=np.load(root/evidence['batch_artifact'],allow_pickle=False)
    x=batch['features'];lengths=batch['lengths'];mask=batch['mask'];source=batch['sources']
    if x.ndim!=3 or x.shape[2]!=414 or x.dtype!=np.float32:raise ValueError('INVALID_TRAINING_BATCH')
    if lengths.shape!=(len(x),) or lengths.dtype.kind not in 'iu' or np.any(lengths<1) or np.any(lengths>x.shape[1]):raise ValueError('INVALID_BATCH_LENGTHS')
    expected=np.arange(x.shape[1])[None,:]<lengths[:,None]
    if mask.dtype!=np.bool_ or mask.shape!=expected.shape or not np.array_equal(mask,expected):raise ValueError('INVALID_BATCH_MASK')
    if source.shape!=(len(x),) or set(source.tolist())!=SOURCES:raise ValueError('BATCH_MISSING_SOURCE')
    if not np.isfinite(x[mask]).all():raise ValueError('NONFINITE_VALID_BATCH_FRAMES')
    steps=evidence.get('steps',[])
    if not 1<=len(steps)<=3:raise ValueError('EXPECTED_ONE_TO_THREE_TRAINING_STEPS')
    for step in steps:
        for name in ('loss','gradient_norm'):
            value=step.get(name)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):raise ValueError('INVALID_TRAINING_'+name.upper())
        if step['gradient_norm']<=0:raise ValueError('NO_TRAINING_GRADIENT')
    for key in ('checkpoint_reloaded','actual_project_dataloader','actual_project_model','tokenizer_source_reconstruction_pass','normalizer_checkpoint_compatible'):
        if evidence.get(key) is not True:raise ValueError('MISSING_RUNTIME_VALIDATION:'+key)
    checkpoint=json.loads((root/evidence['checkpoint_metadata']).read_text())
    # Metadata cannot hash itself or the checkpoint that embeds this metadata.
    compare={k:v for k,v in bindings.items() if k not in (evidence['checkpoint_metadata'],evidence['checkpoint_artifact'])}
    if checkpoint.get('model_contract')!=MODEL_CONTRACT or checkpoint.get('input_hashes')!=compare:raise ValueError('CHECKPOINT_METADATA_MISMATCH')
    error=evidence.get('checkpoint_reload_output_max_abs')
    if isinstance(error,bool) or not isinstance(error,(int,float)) or not math.isfinite(error) or not 0<=error<=1e-6:raise ValueError('CHECKPOINT_RELOAD_NUMERICAL_FAILURE')
    exports=evidence.get('per_source_export',{})
    if set(exports)!=SOURCES:raise ValueError('MISSING_SOURCE_EXPORT_ROUNDTRIP')
    limits={'translation_max_m':1e-4,'orientation_max_deg':.02,'hinge_max_deg':.02,'fk_position_max_m':.001,'fk_orientation_max_deg':.1}
    for source,metrics in exports.items():
        for key,limit in limits.items():
            value=metrics.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=limit:raise ValueError('EXPORT_ROUNDTRIP_FAILURE:'+source+':'+key)
    return {'status':'PASS','steps':len(steps),'sources':sorted(SOURCES),'valid_frames':int(mask.sum()),'evidence_path':str(path),'evidence_sha256':sha(path)}


def manifest_evidence(root):
    ids=set();families={};counts={source:0 for source in SOURCES};paths=[]
    for split in ('train','val','test'):
        path=root/f'manifests/accepted_{split}.jsonl';rows=read_jsonl(path);paths.append(path)
        if split=='train' and not rows:raise ValueError('EMPTY_ACCEPTED_TRAIN')
        for row in rows:
            sid=row['sample_id'];source=row['dataset'];family=row['source_family_id']
            if sid in ids:raise ValueError('DUPLICATE_ACCEPTED_ID:'+sid)
            ids.add(sid)
            if source not in SOURCES or row.get('project_split')!=split:raise ValueError('INVALID_ACCEPTED_SOURCE_OR_SPLIT')
            if families.setdefault(family,split)!=split:raise ValueError('SOURCE_FAMILY_SPLIT_LEAKAGE:'+family)
            if row.get('status')!='QA_PASS' or row.get('target_fps')!=25:raise ValueError('ACCEPTED_CLIP_NOT_QA_PASS_25FPS')
            if row.get('sim_qa_status') not in ('PASS','COVERED_BY_FROZEN_STRATIFIED_PANEL'):raise ValueError('DYNAMIC_QA_NOT_COVERED')
            if row.get('leakage_status')!='PASS':raise ValueError('UNRESOLVED_ACCEPTED_LEAKAGE')
            if source=='bones' and row.get('gmr_role')!='official_g1_main_pool':raise ValueError('BONES_MAIN_POOL_NOT_OFFICIAL')
            if source=='humanml3d' and not row.get('native_provenance_validated'):raise ValueError('HML_NATIVE_PROVENANCE_MISSING')
            if source=='hiphi' and row.get('body_only_validated') is not True:raise ValueError('HIPHI_BODY_ONLY_NOT_PROVEN')
            for path_key,hash_key in [('features_path','features_hash'),('qpos_path','qpos_hash'),('csv_path','csv_hash'),('sidecar_path','sidecar_hash')]:
                artifact=root/row[path_key]
                if not artifact.is_file() or sha(artifact)!=row.get(hash_key):raise ValueError('ACCEPTED_ARTIFACT_STALE:'+sid+':'+path_key)
            features=np.load(root/row['features_path'],mmap_mode='r',allow_pickle=False)
            qpos=np.load(root/row['qpos_path'],mmap_mode='r',allow_pickle=False)
            n=row['target_frames']
            if features.shape!=(n,414) or qpos.shape!=(n,36) or not np.isfinite(features).all() or not np.isfinite(qpos).all():raise ValueError('INVALID_ACCEPTED_ARRAY:'+sid)
            sidecar=json.loads((root/row['sidecar_path']).read_text())
            if sidecar.get('fps')!=25:raise ValueError('SIDECAR_FPS_MISMATCH:'+sid)
            if split=='train':counts[source]+=1
    if any(n==0 for n in counts.values()):raise ValueError('TRAIN_MANIFEST_MISSING_SOURCE')
    return {'status':'PASS','unique_clips':len(ids),'unique_families':len(families),'train_source_counts':counts}

def run():
    checks=[]
    for stage in REQUIRED:
        try: require_pass(stage);checks.append({'stage':stage,'status':'PASS'})
        except Exception as exc:checks.append({'stage':stage,'status':'BLOCKED','reason':str(exc)})
    for name,validate in [('accepted_data',lambda:manifest_evidence(ROOT)),('training_runtime',lambda:runtime_evidence(ROOT,fingerprint()))]:
        try:checks.append({'stage':name,**validate()})
        except Exception as exc:checks.append({'stage':name,'status':'BLOCKED','reason':str(exc)})
    status='PASS' if all(x['status']=='PASS' for x in checks) else 'BLOCKED'
    report={'stage':'06_training_data_ready','status':status,'checks':checks,'release_allowed':status=='PASS',
            'reason':None if status=='PASS' else 'Required upstream, accepted-data or actual training runtime evidence is invalid or missing.'}
    report_path=ROOT/'qa/aggregate/training_release_validation.json';write_json(report_path,report)
    inputs=[ROOT/f'gates/{s}.json' for s in REQUIRED]+[report_path,ROOT/'qa/aggregate/final_runtime_acceptance.json']
    if status=='PASS':
        runtime=json.loads((ROOT/'qa/aggregate/final_runtime_acceptance.json').read_text())
        inputs.extend(ROOT/name for name in runtime['input_hashes'])
    gate('06_training_data_ready',status,report['reason'],checks=checks,release_allowed=status=='PASS',
         upstream_dependencies=REQUIRED,actual_quantity=0 if status!='PASS' else checks[-2]['unique_clips'],
         input_paths=inputs)
    lines=['# Training acceptance\n',f'Status: {status}. TRAINING_DATA_READY is '+('verified.' if status=='PASS' else 'not issued.')+'\n']
    lines += [f"- {c['stage']}: {c['status']}"+(f" — {c['reason']}" if c.get('reason') else '') for c in checks]
    atomic_bytes(ROOT/'reports/final_acceptance.md',('\n'.join(lines)+'\n').encode())
    return report
