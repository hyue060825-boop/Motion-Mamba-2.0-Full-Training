"""Fixed smoke/dev collision-refinement runner with sealed resume inputs."""
import argparse
import io
import json
import time
import numpy as np
from .common import ROOT, BONES, read_jsonl, write_json, write_jsonl, sha, atomic_bytes
from .gates import require_pass
from .locked_target import prepare, create_retargeter
from .gmr_soma import parse
from .gmr_collision_refinement import refine
from .bones_continuity_qa import metrics as continuity
from .robot import read_csv, FORMAT
from .resample import qpos as resample


def run(panel='smoke', resume=False):
    require_pass('00_preflight')
    if panel not in ('smoke', 'dev'):
        raise ValueError('HELD_OUT_REQUIRES_FROZEN_CONFIGURATION')
    namespace = f'bones_collision_{panel}_v19'; out = ROOT/'work'/namespace
    base = f'bones_temporal_arms_{panel}_v15'; base_root = ROOT/'work'/base
    manifest = ROOT/f'manifests/bones_pairs_{panel}.jsonl'; rows = read_jsonl(manifest)
    expected = 16 if panel=='smoke' else 200
    if len(rows)!=expected:
        raise ValueError('FIXED_PANEL_SIZE_MISMATCH')
    source_run = ROOT/'qa/aggregate'/f'{base}.json'
    source = {r['sample_id']: r for r in json.loads(source_run.read_text())['results']}
    if len(source)!=expected or any(r['status']!='DIAGNOSTIC_COMPLETE' for r in source.values()):
        raise ValueError('INCOMPLETE_SEALED_BASELINE')
    cfgpath = ROOT/'configs/bones_temporal_arms_v15.json'; cfg = json.loads(cfgpath.read_text())
    xml, asset = prepare()
    settings = {'penalty': 2000., 'posture_weight': 5., 'max_nfev': 80, 'correction_velocity_weight': .4}
    modules = ['bones_collision_development', 'gmr_collision_refinement', 'gmr_soma', 'locked_target',
               'bones_continuity_qa', 'bones_pair_qa', 'summarize_bones_qa', 'robot', 'resample']
    design = {'status': 'DEVELOPMENT_ONLY', 'panel': panel, 'namespace': namespace,
              'manifest_sha256': sha(manifest), 'base_report_sha256': sha(source_run), 'config_sha256': sha(cfgpath),
              'settings': settings, 'root_fixed': True, 'time_fixed': True, 'held_out_read': False,
              'expected': expected, 'reference_used_in_objective': False,
              'modules': {name: sha(ROOT/'g1_pipeline'/f'{name}.py') for name in modules},
              'asset_hash': asset['asset_hash'], 'resources': {'workers': 1, 'cpu_threads': 1, 'gpu': False},
              'estimated_disk_bytes': sum(r['frames'] for r in source.values())*2000,
              'estimated_cpu_hours': sum(r['frames'] for r in source.values())*.15/3600}
    if (out/'run.json').exists():
        if json.loads((out/'run.json').read_text())!=design:
            raise ValueError('NAMESPACE_FINGERPRINT_CHANGED')
        if not resume:
            raise ValueError('EXISTING_NAMESPACE_REQUIRES_RESUME')
    out.mkdir(exist_ok=True); write_json(out/'run.json', design)
    previous = {r['sample_id']: r for r in read_jsonl(out/'results.jsonl')} if (out/'results.jsonl').exists() else {}
    results = []
    for row in rows:
        started = time.monotonic(); sid = row['sample_id']; mid = row['source_motion_id']
        original = base_root/(mid+'.npy'); timepath = original.with_suffix('.time.json')
        bvh = ROOT/'work'/('gmr_soma_smoke' if panel=='smoke' else 'bones_pairs_dev')/(mid+'.bvh')
        reference_path = BONES/row['source_paths']['g1']
        op = out/(mid+'.npy'); tp = out/(mid+'.time.json'); sp = out/(mid+'.solver.json')
        record = {'sample_id': sid, 'status': 'ERROR', 'retargeted': False, 'reference_loaded': False}
        try:
            bound = source[sid]
            for path, key in [(original, 'output_sha256'), (timepath, 'time_sha256'),
                              (bvh, 'source_sha256'), (reference_path, 'reference_sha256')]:
                if sha(path)!=bound[key]:
                    raise ValueError('BASE_INPUT_HASH_CHANGED:'+key)
            cached = previous.get(sid, {})
            if resume and cached.get('status')=='DIAGNOSTIC_COMPLETE' and all(
                    p.exists() and sha(p)==cached.get(k) for p,k in
                    [(op,'output_sha256'),(tp,'time_sha256'),(sp,'solver_sha256')]):
                q = np.load(op)
                if q.shape==(cached['frames'],36) and np.isfinite(q).all():
                    results.append(cached); continue
            q = np.load(original); mapping = json.loads(timepath.read_text())
            t = np.asarray(mapping['source']['target_timestamps'])
            frames,_ = parse(bvh,{n:n for n in cfg['human_scale_table']},target_timestamps=t,rotation_basis='world_only')
            solver = create_retargeter(cfgpath,xml); corrected=[]; infos=[]; delta=None
            for i,(frame,seed) in enumerate(zip(frames,q)):
                pose,info = refine(solver,frame,seed,previous_correction=delta,**settings)
                delta=pose[7:]-seed[7:];corrected.append(pose);infos.append(info)
                if i%100==0:print(sid,i,'/',len(q),flush=True)
            corrected=np.asarray(corrected)
            if corrected.shape!=q.shape or not np.array_equal(corrected[:,:7],q[:,:7]):
                raise ValueError('ROOT_OR_LENGTH_CHANGED')
            reference,rt,_=read_csv(reference_path,format_id=FORMAT,fps=row['original_fps'])
            reference,_,_=resample(reference,rt,solver.model.jnt_range[1:],end=np.nextafter(t[-1],np.inf))
            if reference.shape!=q.shape:raise ValueError('REFERENCE_LENGTH_MISMATCH')
            error=np.rad2deg(abs(corrected[:,7:]-reference[:,7:]))
            buf=io.BytesIO();np.save(buf,corrected);atomic_bytes(op,buf.getvalue())
            atomic_bytes(tp,timepath.read_bytes());write_json(sp,{'frames':infos,'settings':settings})
            record={**bound,'hinge_mae_deg':float(error.mean()),'per_joint_mae_deg':error.mean(axis=0).tolist(),
                    'output_sha256':sha(op),'solver_sha256':sha(sp),'base_output_sha256':sha(original),
                    'candidate_continuity':continuity(solver.model,corrected,t),
                    'correction_mae_deg':float(np.rad2deg(abs(corrected[:,7:]-q[:,7:])).mean()),
                    'optimizer_budget_frames':sum(not r['optimizer_success'] for r in infos)}
        except Exception as exc:
            record['error']=repr(exc);record['traceback']=__import__('traceback').format_exc()
        record['elapsed_sec']=time.monotonic()-started;results.append(record)
        write_jsonl(out/'results.jsonl',results)
        print(sid,record['status'],record.get('error',''),flush=True)
    report={'status':'DEVELOPMENT_ONLY','fixed_panel_size':expected,'actual_attempted':len(results),
            'retargeted':sum(r['retargeted'] for r in results),'error_count':sum(r['status']=='ERROR' for r in results),
            'run_config':design,'results':results}
    write_json(ROOT/'qa/aggregate'/f'{namespace}.json',report)
    from .bones_pair_qa import run as pair_qa
    from .summarize_bones_qa import summarize
    pair_qa(namespace,panel);summarize(namespace)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=['smoke','dev'],default='smoke')
    p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.panel,a.resume)
