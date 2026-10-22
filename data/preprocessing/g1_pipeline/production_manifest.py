"""Flatten validated source output into the versioned G1 training data contract."""
import json
import subprocess
from .common import ROOT,sha,digest,now


def record(source,result,*,pipeline_version,sim_coverage,leakage):
    if result.get('contract_status')!='PASS' or result.get('status') not in ('QA_PASS','KINEMATIC_CANDIDATE'):raise ValueError('MANIFEST_REQUIRES_VALIDATED_OUTPUT')
    if sim_coverage.get('status') not in ('PASS','COVERED_BY_FROZEN_STRATIFIED_PANEL'):raise ValueError('MANIFEST_DYNAMIC_COVERAGE_REQUIRED')
    if leakage.get('status')!='PASS':raise ValueError('MANIFEST_LEAKAGE_REVIEW_REQUIRED')
    paths=result['output_paths'];contract=json.loads((ROOT/paths['contract_path']).read_text());timepath=ROOT/paths['sidecar_path'];time=json.loads(timepath.read_text())
    if contract['status']!='PASS' or contract['fps']!=25 or contract['feature_shape']!=[result['frames'],414]:raise ValueError('PRODUCTION_CONTRACT_NOT_G1')
    assets=json.loads((ROOT/'contracts/g1_robot.json').read_text());rep=json.loads((ROOT/'contracts/motion_mamba2_414.json').read_text())
    row=dict(source)
    row.update(id=source['sample_id'],feature_path=str(ROOT/paths['features_path']),frames=result['frames'],
        interval_convention='[start,end); output last sample <=last native timestamp; duration interval=N/25',
        native_index_mapping=source.get('native_interval'),target_fps=25,target_frames=result['frames'],
        duration_interval_sec=result['frames']/25,last_sample_time_sec=(result['frames']-1)/25,
        output_timestamps_hash=digest(time.get('timestamps',time.get('target_timestamps'))),
        skeleton='G1Skeleton34',nbjoints=34,dof=29,representation=414,features_shape=[result['frames'],414],
        status='QA_PASS',qa_flags=result.get('review_flags',[]),metrics_path=result.get('metrics_path'),
        human_adapter_name='official_g1_main_pool' if source['dataset']=='bones' else 'native_fk_to_gmr_smplx_semantics' if source['dataset']=='humanml3d' else 'hiphi_native_fk_source_specific_rest',
        adapter_version=pipeline_version,source_mapping_hash=result.get('mapping_sha256'),
        pipeline_commit=None,pipeline_commit_reason='isolated work directory has no Git repository; version and file hash bindings are authoritative',
        pipeline_semantic_version=pipeline_version,dependency_lock_hash=sha(ROOT/'audit/versions.lock.json'),
        csv_schema_version=sha(ROOT/'contracts/bones_g1_csv.json'),skeleton_contract_hash=sha(ROOT/'contracts/g1_skeleton34.json'),
        feature_layout_hash=sha(ROOT/'contracts/motion_mamba2_414.json'),target_mjcf_hash=sha(assets['xml']),
        qpos_joint_order_hash=digest(assets.get('joints')),normalizer_hash=digest({'mean':rep['mean_hash'],'std':rep['std_hash']}),
        canonical_transform=contract['canonical_transform'],filter_config=None,filter_config_reason='no unrecorded filter; all source temporal/ground/rate transforms are in metrics',
        split_group_id=source['source_family_id'],duplicate_group_id=leakage.get('duplicate_group_id'),leakage_status='PASS',leakage_evidence=leakage,
        sim_qa_status=sim_coverage['status'],sim_controller_hash=sim_coverage['controller_hash'],sim_coverage=sim_coverage,
        transform_chain=result.get('transform_chain'),transform_chain_reason=None if result.get('transform_chain') else 'see bound source time, solver, mapping and export provenance sidecars',
        source_hashes=result.get('source_hashes') or {'motion':result['source_sha256']},source_metadata_version=source['dataset_release'],
        attempt=result.get('attempt'),elapsed_sec=result.get('elapsed_sec'),created_at=now(),worker_id=result.get('worker_id'),
        native_provenance_validated=bool(result.get('native_provenance_validated')),
        body_only_validated=bool(result.get('body_only_validated')) if source['dataset']=='hiphi' else None)
    if source['dataset']=='bones':row['gmr_role']='official_g1_main_pool'
    else:
        row['gmr_commit']=subprocess.check_output(['git','-C',str(ROOT/'deps/GMR'),'rev-parse','HEAD'],text=True).strip()
        row['solver_config_hash']=result['mapping_sha256']
    row['resampler_version']=sha(ROOT/'g1_pipeline/resample.py')
    row['config_hash']=digest({'source_mapping':result.get('mapping_sha256'),'calibration_protocol':sha(ROOT/'configs/calibration_protocol.v25.json')})
    if result.get('mapping_config'):
        mapping=json.loads((ROOT/result['mapping_config']).read_text());row['scale_strategy']=mapping['scale_strategy']
    row['caption_ids']=[r['caption_id'] for r in source.get('caption_time_spans',[])] or None
    for field in ('session_id','take_id','source_body_model_type','source_skeleton_hash','source_pose_layout','height_source','body_shape_hash','gender_model','scale_strategy','caption_ids','caption_source','adapter_split','gmr_commit','solver_config_hash','resampler_version','config_hash'):
        if row.get(field) is None:row[field]=None;row[field+'_reason']='not available in this source record; no inferred identity or fabricated measurement'
    for key in ('features','qpos','csv','sidecar'):
        path=ROOT/paths[key+'_path'];bound=sha(path)
        if bound!=result['output_hashes'][str(path.relative_to(ROOT))]:raise ValueError('PRODUCTION_OUTPUT_CHANGED')
        row[key+'_path']=str(path.relative_to(ROOT));row[key+'_hash']=bound
    return row
