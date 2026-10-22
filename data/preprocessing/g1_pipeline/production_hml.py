"""Fixed HML100 runtime; cannot execute before valid BONES calibration."""
import argparse
import fcntl
import io
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from .common import ROOT,HML,read_jsonl,write_json,write_jsonl,sha,digest,atomic_bytes,now
from .gates import require_pass,gate as production_gate
from .pilot_scope import require_pilot,pilot_gate,AUTH
from .hml_native_runner import native_parameters
from .hml_native import native_pose_to_frames
from .native_rest_mapping import mirror_frames
from .source_action_semantics import classify
from .resample import time_grid
from .production_contract import export_and_check,save_array
from .source_motion_processing import process
from .spectral_qa import local_rotation_spectrum


def run(*,resume=False,dry_run=False,workers=1,max_clips=None,seed=20260915,pilot=False):
    if pilot:require_pilot('humanml3d')
    else:require_pass('01_bones_calibration')
    gate=pilot_gate if pilot else production_gate
    if workers!=1 or seed!=20260915:raise ValueError('FIXED_HML_SMOKE_RESOURCE_SEED')
    manifest=ROOT/'manifests/hml_smoke100.v2.jsonl';rows=read_jsonl(manifest)
    if len(rows)!=100 or len({r['source_family_id'] for r in rows})!=100:raise ValueError('FIXED_HML100_INDEPENDENCE')
    if any(r['official_split']!='train' or r['project_split']!='train' or r['leakage_status']!='FAMILY_SPLIT_CLEAR' for r in rows):raise ValueError('HML_SPLIT_PROVENANCE')
    staticpath=ROOT/'audit/hml_native_rest_mapping_v2.json';static=json.loads(staticpath.read_text())
    if static['measured']!=100 or static['manifest_sha256']!=sha(manifest) or static['module_sha256']!=sha(ROOT/'g1_pipeline/native_rest_mapping.py'):
        raise ValueError('STATIC_SOURCE_MAPPING_STALE_OR_INCOMPLETE')
    mappings={r['sample_id']:r for r in static['results']}
    modules=['production_hml','production_contract','source_motion_processing','source_action_semantics','native_rest_mapping','hml_native','hml_native_runner','resample','spectral_qa','source_smoke_evaluation','render_frozen_review']
    fingerprint={'manifest':sha(manifest),'static_mappings':sha(staticpath),'calibration_gate':sha(AUTH if pilot else ROOT/'gates/01_bones_calibration.json'),
        'code':{n:sha(ROOT/'g1_pipeline'/f'{n}.py') for n in modules}}
    if pilot:fingerprint.update(pilot_authorization=sha(AUTH),pilot_scope_code=sha(ROOT/'g1_pipeline/pilot_scope.py'))
    namespace=('pilot_' if pilot else '')+'hml100_native_v1_'+digest(fingerprint)[:12];out=ROOT/'work'/namespace
    design={'execution_scope':'isolated_pilot' if pilot else 'production','namespace':namespace,'expected':100,'fingerprint':fingerprint,'source_body_model':'native SMPL-H',
        'target':'locked project G1','resources':{'cpu_workers':1,'threads':1,'gpu':False,'estimated_output_gb':1},
        'geometry_scenes':{'flat':'locked horizontal floor, all compiled collision geometries',
          'ground_multisupport':'same plane and all body contacts; intentional native floor support is allowed, forbidden self-collision is not exempted',
          'external_or_unknown':'reject missing support scene; retain input denominator'},
        'fixed_quality_thresholds':'configs/gates.v2.yaml; numerical/ground/self/skating and frozen relative flags',
        'human_frame_interface':'native_fk_to_gmr_smplx_semantics','parameters_are_smplx':False}
    if dry_run:return {**design,'dry_run':True,'input_duration_sec':sum(r['native_end_sec']-r['native_start_sec'] for r in rows)}
    if max_clips is not None and not 1<=max_clips<=100:raise ValueError('HML_FIXED_PANEL_BUDGET')
    out.mkdir(exist_ok=True)
    with (out/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('HML_NAMESPACE_ALREADY_RUNNING')
        if (out/'run.json').exists():
            if json.loads((out/'run.json').read_text())!=design:raise ValueError('HML_NAMESPACE_CHANGED')
            if not resume:raise ValueError('HML_NAMESPACE_REQUIRES_RESUME')
        write_json(out/'run.json',design)
        gate('02_hml100','RUNNING',input_paths=[manifest,staticpath,ROOT/'gates/01_bones_calibration.json'],upstream_dependencies=['01_bones_calibration'])
        previous={r['sample_id']:r for r in read_jsonl(out/'results.jsonl')} if (out/'results.jsonl').exists() else {}
        for row in rows[:max_clips]:
            sid=row['sample_id'];mid=row['source_motion_id'];started=time.monotonic();mp=mappings[sid];path=HML/row['source_paths']['native']
            config=ROOT/mp['config_path'];adir=out/mid;old=previous.get(sid,{})
            if sha(path)!=mp['native_sha256'] or sha(config)!=mp['config_sha256']:raise ValueError('SEALED_HML_INPUT_CHANGED')
            mapping_audit=ROOT/mp['audit_path']
            if sha(mapping_audit)!=mp['audit_sha256']:raise ValueError('SEALED_HML_MAPPING_AUDIT_CHANGED')
            provenance=json.loads(mapping_audit.read_text())['source_provenance']
            native_model=Path(provenance['native_model']);dmpl_model=Path(str(native_model).replace('/smplh/','/dmpls/'))
            if sha(native_model)!=provenance['native_model_sha256'] or sha(dmpl_model)!=provenance['dmpl_model_sha256']:
                raise ValueError('SEALED_HML_BODY_MODEL_CHANGED')
            if resume and old.get('status') in ('KINEMATIC_CANDIDATE','REJECTED','REVIEW_REQUIRED') and old.get('output_hashes') and all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in old['output_hashes'].items()):
                print(sid,'verified resume',flush=True);continue
            entry={'sample_id':sid,'status':'ERROR','dataset':'humanml3d','source_family_id':row['source_family_id']}
            write_json(out/'heartbeat.json',{'time':now(),'pid':os.getpid(),'sample_id':sid,'phase':'native_fk_and_gmr'})
            try:
                interval=row['native_interval'];native=native_parameters(path,start_frame=interval['native_first_index'],end_frame=interval['native_end_index_exclusive'])
                source_times=(native['start_frame']+np.arange(len(native['full_pose'])))/native['source_fps'];t,tmap=time_grid(source_times,end=interval['end_sec'])
                frames=native_pose_to_frames(native['full_pose'],native['trans'],native['parents'],native['offsets'],native['source_fps'],t,start_frame=native['start_frame'])
                if row['mirror_flag']:frames=mirror_frames(frames)
                semantics=classify(row);q,original,metrics,residuals=process(frames,t,config,semantics,dataset='humanml3d',pilot=pilot)
                native_rotations=R.from_rotvec(native['full_pose'].reshape(-1,3)).as_matrix().reshape(native['full_pose'].shape[:2]+(3,3))
                metrics['native_spectrum']=local_rotation_spectrum(native_rotations[:,:22],native['source_fps'])
                if metrics['native_spectrum']['review_required']:metrics['review_flags'].append('NATIVE_HIGH_FREQUENCY_REVIEW_REQUIRED')
                adir.mkdir(exist_ok=True);save_array(adir/'unprocessed_qpos.npy',original)
                write_json(adir/'time.json',{'source_time_mapping':tmap,'source_interval':interval,'resample_count':1,'source_model':'SMPL-H52_156','mirror':row['mirror_flag']})
                write_json(adir/'solver.json',metrics);buf=io.BytesIO();np.savez_compressed(buf,timestamps=t,**residuals);atomic_bytes(adir/'human_targets.npz',buf.getvalue())
                contract,paths,hashes=export_and_check(adir,sid,q,t,{'sample_id':sid,'source':'humanml3d_native_smplh','native_path':str(path),'native_sha256':sha(path),
                    'native_interval':interval,'mapping_config_sha256':sha(config),'scale_strategy':json.loads(config.read_text())['scale_strategy']},row['captions'])
                rejected=list(metrics['rejections']);flags=list(metrics['review_flags'])
                if contract['status']!='PASS':rejected.append('NUMERICAL_CONTRACT_FAILURE')
                if any(v is not None and v>.20 for v in contract['skating_p95_m_s'].values()):rejected.append('FOOT_SKATING')
                for p in [adir/'unprocessed_qpos.npy',adir/'time.json',adir/'solver.json',adir/'human_targets.npz']:hashes[str(p.relative_to(ROOT))]=sha(p)
                entry.update(status='REJECTED' if rejected else 'REVIEW_REQUIRED' if flags else 'KINEMATIC_CANDIDATE',
                    frames=len(q),duration_interval_sec=len(q)/25,last_sample_time_sec=float(t[-1]-t[0]),
                    rejections=rejected,review_flags=flags,contract_status=contract['status'],source_semantics=semantics,
                    source_metadata=row,output_paths=paths,output_hashes=hashes,mapping_config=str(config.relative_to(ROOT)),mapping_sha256=sha(config),
                    metrics_path=str((adir/'solver.json').relative_to(ROOT)),source_sha256=sha(path),native_provenance_validated=True)
            except Exception as exc:entry.update(error=repr(exc),traceback=__import__('traceback').format_exc())
            entry.update(elapsed_sec=time.monotonic()-started,attempt=old.get('attempt',0)+1);previous[sid]=entry
            results=[previous[r['sample_id']] for r in rows if r['sample_id'] in previous]
            write_jsonl(out/'results.jsonl',results);print(sid,entry['status'],entry.get('rejections'),entry.get('error',''),flush=True)
        results=[previous[r['sample_id']] for r in rows if r['sample_id'] in previous]
        counts={state:sum(r['status']==state for r in results) for state in ('KINEMATIC_CANDIDATE','REVIEW_REQUIRED','REJECTED','ERROR')}
        report={'status':'SMOKE_MEASURED_NOT_RELEASE','namespace':namespace,'expected':100,'attempted':len(results),
            'pending':100-len(results),'counts':counts,'design':design,'results':results}
        write_json(ROOT/'qa/aggregate'/f'{namespace}.json',report);write_json(ROOT/('audit/current_pilot_hml100_run.json' if pilot else 'audit/current_hml100_run.json'),{'namespace':namespace,'report_sha256':sha(ROOT/'qa/aggregate'/f'{namespace}.json')})
        gate('02_hml100','BLOCKED','Fixed HML smoke requires complete numerical/scene/anomaly/video review before acceptance.',
            actual_quantity=len(results),metrics=counts,input_paths=[manifest,staticpath,ROOT/'qa/aggregate'/f'{namespace}.json'],upstream_dependencies=['01_bones_calibration'])
        if len(results)==100:
            from .source_smoke_evaluation import select_videos,evaluate
            from .render_frozen_review import run as render
            render(namespace,list(select_videos(namespace)),source='humanml3d')
            evaluate(namespace,'humanml3d',pilot=pilot)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--max-clips',type=int);p.add_argument('--pilot',action='store_true')
    a=p.parse_args();r=run(resume=a.resume,dry_run=a.dry_run,max_clips=a.max_clips,pilot=a.pilot);print({k:v for k,v in r.items() if k!='results'})
