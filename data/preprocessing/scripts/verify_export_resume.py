"""Real exporter interruption, recovery and two-process equivalence on BONES dev."""
import json
import multiprocessing as mp
import os
import numpy as np
from g1_pipeline.common import ROOT,read_jsonl,write_json,sha
from g1_pipeline.batch_state import State
from g1_pipeline.gates import require_pass
from g1_pipeline.production_contract import export_and_check

MID='pels_air_punch_001__A493'
BASE=ROOT/'work/export_resume_validation_v1'
FINGERPRINT={'production_contract':sha(ROOT/'g1_pipeline/production_contract.py'),'batch_state':sha(ROOT/'g1_pipeline/batch_state.py')}


def worker(directory,crash=False):
    require_pass('00_preflight');row=next(r for r in read_jsonl(ROOT/'manifests/bones_pairs_dev.jsonl') if r['source_motion_id']==MID)
    state=State(directory,FINGERPRINT);state.register([row],'export-validation-v1')
    with state.own(row['sample_id'],resume=True) as owned:
        if owned is None:state.close();return
        source=ROOT/'work/bones_frozen_dev_v25_verification';q=np.load(source/(MID+'.npy'));t=np.array(json.loads((source/(MID+'.time.json')).read_text())['source']['target_timestamps'])
        if crash:
            from g1_pipeline import production_contract
            original=production_contract.export_csv
            def interrupted(*args,**kwargs):
                original(*args,**kwargs);state.heartbeat(row['sample_id'],'EXPORTED');os._exit(17)
            production_contract.export_csv=interrupted
        info,paths,hashes=export_and_check(directory/'artifacts',row['sample_id'],q,t,{'source':'fixed existing BONES dev; resume validation'},row['captions'])
        if info['status']!='PASS':raise ValueError('EXPORT_RESUME_NUMERICAL_FAILURE')
        payload={'artifact_hashes':{str(ROOT/p):h for p,h in hashes.items()},'frames':len(q),
            **{k:str(ROOT/paths[k]) for k in ('contract_path','qpos_path','features_path')}}
        state.commit(row['sample_id'],'QA_PASS',payload)
    state.close()


def run():
    require_pass('00_preflight')
    if BASE.exists():raise ValueError('EXPORT_RESUME_AUDIT_NAMESPACE_EXISTS')
    BASE.mkdir();ctx=mp.get_context('spawn');crashed=ctx.Process(target=worker,args=(BASE/'interrupted',True));crashed.start();crashed.join(60)
    if crashed.is_alive():crashed.terminate();crashed.join();raise RuntimeError('CONTROLLED_EXPORT_INTERRUPTION_TIMEOUT')
    if crashed.exitcode!=17:raise ValueError('CONTROLLED_EXPORT_DID_NOT_INTERRUPT')
    state=State(BASE/'interrupted',FINGERPRINT);before=state.rows();state.close()
    if len(before)!=1 or before[0]['status']!='EXPORTED' or before[0]['output_json'] is not None:raise ValueError('PARTIAL_EXPORT_COMMITTED_SUCCESS')
    worker(BASE/'interrupted');worker(BASE/'serial')
    children=[ctx.Process(target=worker,args=(BASE/f'parallel{i}',)) for i in range(2)]
    for child in children:child.start()
    for child in children:
        child.join(60)
        if child.is_alive():child.terminate();child.join();raise RuntimeError('PARALLEL_EXPORT_TIMEOUT')
        if child.exitcode!=0:raise ValueError('PARALLEL_EXPORT_FAILED')
    paths=[BASE/name/'artifacts' for name in ('interrupted','serial','parallel0','parallel1')];matches={}
    for name in ('motion.csv','features.npy','qpos.npy'):
        hashes=[sha(p/name) for p in paths];matches[name]={'sha256':hashes,'identical':len(set(hashes))==1}
    state=State(BASE/'interrupted',FINGERPRINT);after=state.rows();state.close();worker(BASE/'interrupted')
    state=State(BASE/'interrupted',FINGERPRINT);again=state.rows();state.close()
    if len(after)!=1 or after[0]['attempt']!=2 or after!=again:raise ValueError('EXPORT_RESUME_NOT_IDEMPOTENT')
    report={'status':'PASS' if all(v['identical'] for v in matches.values()) else 'FAIL','sample_id':'bones:'+MID,
        'real_exporter_runs':4,'parallel_workers':2,'controlled_exit_code':17,'before_recovery_status':before[0]['status'],
        'after_recovery_status':after[0]['status'],'recovery_attempts':after[0]['attempt'],'identical_artifacts':matches,
        'all_actual_official_and_project_loader_contracts_pass':True,'fingerprint':FINGERPRINT,
        'scope':'exporter/actual loaders/features and atomic commit; not parallel GMR equivalence or full-batch release',
        'source_retargeting_run':False}
    write_json(ROOT/'audit/export_resume_validation_v1.json',report);print(report['status'])


if __name__=='__main__':run()
