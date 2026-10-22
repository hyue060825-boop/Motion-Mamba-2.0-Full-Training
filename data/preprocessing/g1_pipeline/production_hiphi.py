"""Gated fixed HiPHI100 execution with native BVH FK and source-specific rest."""
import argparse
import fcntl
import io
import json
import os
import time
import numpy as np
from .common import ROOT,read_jsonl,write_json,write_jsonl,sha,digest,atomic_bytes,now
from .gates import require_pass,gate as production_gate
from .pilot_scope import require_pilot,pilot_gate,AUTH
from .hiphi_native import frames_at_25hz,read_bvh
from .source_action_semantics import classify
from .source_motion_processing import process
from .production_contract import save_array,export_and_check
from .spectral_qa import local_rotation_spectrum


def run(*,resume=False,dry_run=False,workers=1,max_clips=None,seed=20260915,pilot=False):
    if pilot:require_pilot('hiphi')
    else:require_pass('02_hml100')
    gate=pilot_gate if pilot else production_gate
    if workers!=1 or seed!=20260915:raise ValueError('FIXED_HIPHI_RESOURCE_SEED')
    manifest=ROOT/'manifests/hiphi_smoke100.jsonl';rows=read_jsonl(manifest)
    staticpath=ROOT/'audit/hiphi_native_rest_mapping_v1.json';static=json.loads(staticpath.read_text())
    packagepath=ROOT/'audit/hiphi_package_evidence_v1.json';package=json.loads(packagepath.read_text())
    if len(rows)!=100 or len({r['source_family_id'] for r in rows})!=100:raise ValueError('FIXED_HIPHI100_INDEPENDENCE')
    if any(r['project_split']!='train' or r['is_hoi']!='false' or r['mirror_flag'] for r in rows):raise ValueError('HIPHI_PANEL_PROVENANCE')
    if static['measured']!=100 or static['manifest_sha256']!=sha(manifest) or static['module_sha256']!=sha(ROOT/'g1_pipeline/hiphi_rest_mapping.py') or static['shared_mapping_sha256']!=sha(ROOT/'g1_pipeline/native_rest_mapping.py'):
        raise ValueError('STATIC_HIPHI_MAPPING_STALE_OR_INCOMPLETE')
    if package['input_metadata_pass']!=100 or package['measured']!=100:raise ValueError('HIPHI_PACKAGE_PROVENANCE_INCOMPLETE')
    mappings={r['sample_id']:r for r in static['results']};packages={r['sample_id']:r for r in package['results']}
    if set(mappings)!=set(packages) or set(mappings)!={r['sample_id'] for r in rows}:raise ValueError('HIPHI_EVIDENCE_PANEL_MISMATCH')
    modules=['production_hiphi','production_contract','source_motion_processing','source_action_semantics','native_rest_mapping','hiphi_rest_mapping','hiphi_native','resample','spectral_qa','source_smoke_evaluation','render_frozen_review']
    fingerprint={'manifest':sha(manifest),'static_mappings':sha(staticpath),'package_evidence':sha(packagepath),
        'hml_gate':sha(ROOT/('gates/pilot/02_hml100.json' if pilot else 'gates/02_hml100.json')),'code':{name:sha(ROOT/'g1_pipeline'/f'{name}.py') for name in modules}}
    if pilot:fingerprint.update(pilot_authorization=sha(AUTH),pilot_scope_code=sha(ROOT/'g1_pipeline/pilot_scope.py'))
    namespace=('pilot_' if pilot else '')+'hiphi100_native_v1_'+digest(fingerprint)[:12];out=ROOT/'work'/namespace
    design={'execution_scope':'isolated_pilot' if pilot else 'production','namespace':namespace,'expected':100,'fingerprint':fingerprint,'source':'HiPHI Body-only native55joint BVH',
        'resources':{'cpu_workers':1,'threads':1,'gpu':False,'estimated_output_gb':2},
        'scale':'exported-rig pelvis height to locked G1 rest; metadata actor height audited without double scaling',
        'quality_thresholds':'frozen calibration gates.v2 plus explicit source scene review'}
    if dry_run:return {**design,'dry_run':True}
    if max_clips is not None and not 1<=max_clips<=100:raise ValueError('HIPHI_FIXED_PANEL_BUDGET')
    out.mkdir(exist_ok=True)
    with (out/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('HIPHI_NAMESPACE_ALREADY_RUNNING')
        if (out/'run.json').exists():
            if json.loads((out/'run.json').read_text())!=design:raise ValueError('HIPHI_NAMESPACE_CHANGED')
            if not resume:raise ValueError('HIPHI_NAMESPACE_REQUIRES_RESUME')
        write_json(out/'run.json',design)
        gate('03_hiphi100','RUNNING',input_paths=[manifest,staticpath,packagepath],upstream_dependencies=['02_hml100'])
        previous={r['sample_id']:r for r in read_jsonl(out/'results.jsonl')} if (out/'results.jsonl').exists() else {}
        for row in rows[:max_clips]:
            sid=row['sample_id'];mid=row['source_motion_id'];started=time.monotonic();mp=mappings[sid];pp=packages[sid];old=previous.get(sid,{})
            source=ROOT/mp['native_path'];config=ROOT/mp['config_path'];adir=out/mid
            if pp['status']!='INPUT_METADATA_PASS' or pp['failures'] or pp['metadata']['is_hoi'] or pp['metadata']['objects']:raise ValueError('BODY_ONLY_PACKAGE_REQUIRED')
            if sha(source)!=mp['native_sha256'] or sha(source)!=pp['bvh_member_sha256'] or sha(config)!=mp['config_sha256'] or sha(ROOT/mp['audit_path'])!=mp['audit_sha256']:
                raise ValueError('SEALED_HIPHI_INPUT_CHANGED')
            if resume and old.get('status') in ('KINEMATIC_CANDIDATE','REVIEW_REQUIRED','REJECTED') and old.get('output_hashes') and all((ROOT/p).exists() and sha(ROOT/p)==h for p,h in old['output_hashes'].items()):
                print(sid,'verified resume',flush=True);continue
            write_json(out/'heartbeat.json',{'time':now(),'pid':os.getpid(),'sample_id':sid,'phase':'native_fk_and_gmr'})
            entry={'sample_id':sid,'dataset':'hiphi','source_family_id':row['source_family_id'],'status':'ERROR'}
            try:
                frames,tinfo=frames_at_25hz(source);t=np.asarray(tinfo['time_mapping']['target_timestamps'])
                if tinfo['source_frames']!=row['original_frames']:raise ValueError('HIPHI_SOURCE_FRAME_METADATA_CONFLICT')
                semantics=classify(row);q,original,metrics,residuals=process(frames,t,config,semantics,dataset='hiphi',pilot=pilot)
                native=read_bvh(source);relevant={n for n in json.loads(config.read_text())['human_scale_table']}
                indices=[i for i,n in enumerate(native['nodes']) if n['name'] in relevant]
                metrics['native_spectrum']=local_rotation_spectrum(native['local_rotations'][:,indices],native['fps'])
                if metrics['native_spectrum']['review_required']:metrics['review_flags'].append('NATIVE_HIGH_FREQUENCY_REVIEW_REQUIRED')
                adir.mkdir(exist_ok=True);save_array(adir/'unprocessed_qpos.npy',original);write_json(adir/'time.json',tinfo);write_json(adir/'solver.json',metrics)
                buf=io.BytesIO();np.savez_compressed(buf,timestamps=t,**residuals);atomic_bytes(adir/'human_targets.npz',buf.getvalue())
                contract,paths,hashes=export_and_check(adir,sid,q,t,{'sample_id':sid,'source':'hiphi_body_only_bvh',
                    'native_path':str(source),'native_sha256':sha(source),'mapping_sha256':sha(config),
                    'actor_height_m':row['height_m'],'scale_strategy':json.loads(config.read_text())['scale_strategy'],
                    'package_evidence_sha256':sha(packagepath),'resample_count':1},row['captions'])
                rejected=list(metrics['rejections']);flags=list(metrics['review_flags'])
                if contract['status']!='PASS':rejected.append('NUMERICAL_CONTRACT_FAILURE')
                if any(v is not None and v>.2 for v in contract['skating_p95_m_s'].values()):rejected.append('FOOT_SKATING')
                if mp['height_discrepancy_above_10_percent']:flags.append('EXPORTED_RIG_VS_ACTOR_HEIGHT_REVIEW_REQUIRED')
                for path in [adir/'unprocessed_qpos.npy',adir/'time.json',adir/'solver.json',adir/'human_targets.npz']:hashes[str(path.relative_to(ROOT))]=sha(path)
                entry.update(status='REJECTED' if rejected else 'REVIEW_REQUIRED' if flags else 'KINEMATIC_CANDIDATE',
                    frames=len(q),duration_interval_sec=len(q)/25,last_sample_time_sec=float(t[-1]-t[0]),
                    rejections=rejected,review_flags=flags,contract_status=contract['status'],source_semantics=semantics,
                    source_metadata={**row,'original_fps':tinfo['source_fps']},output_paths=paths,output_hashes=hashes,
                    mapping_config=str(config.relative_to(ROOT)),mapping_sha256=sha(config),metrics_path=str((adir/'solver.json').relative_to(ROOT)),
                    source_sha256=sha(source),native_provenance_validated=True)
            except Exception as exc:entry.update(error=repr(exc),traceback=__import__('traceback').format_exc())
            entry.update(elapsed_sec=time.monotonic()-started,attempt=old.get('attempt',0)+1);previous[sid]=entry
            write_jsonl(out/'results.jsonl',[previous[r['sample_id']] for r in rows if r['sample_id'] in previous])
            print(sid,entry['status'],entry.get('rejections'),entry.get('error',''),flush=True)
        results=[previous[r['sample_id']] for r in rows if r['sample_id'] in previous]
        counts={state:sum(r['status']==state for r in results) for state in ('KINEMATIC_CANDIDATE','REVIEW_REQUIRED','REJECTED','ERROR')}
        report={'status':'SMOKE_MEASURED_NOT_RELEASE','namespace':namespace,'expected':100,'attempted':len(results),'pending':100-len(results),'counts':counts,'design':design,'results':results}
        reportpath=ROOT/'qa/aggregate'/f'{namespace}.json';write_json(reportpath,report);write_json(ROOT/('audit/current_pilot_hiphi100_run.json' if pilot else 'audit/current_hiphi100_run.json'),{'namespace':namespace,'report_sha256':sha(reportpath)})
        gate('03_hiphi100','BLOCKED','Complete source numeric/scene/anomaly/video acceptance is required.',actual_quantity=len(results),metrics=counts,
            input_paths=[manifest,staticpath,packagepath,reportpath],upstream_dependencies=['02_hml100'])
        if len(results)==100:
            from .source_smoke_evaluation import select_videos,evaluate
            from .render_frozen_review import run as render
            render(namespace,list(select_videos(namespace)),source='hiphi')
            evaluate(namespace,'hiphi',pilot=pilot)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--max-clips',type=int);p.add_argument('--pilot',action='store_true')
    a=p.parse_args();r=run(resume=a.resume,dry_run=a.dry_run,max_clips=a.max_clips,pilot=a.pilot);print({k:v for k,v in r.items() if k!='results'})
