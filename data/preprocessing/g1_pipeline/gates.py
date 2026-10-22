from __future__ import annotations
from .common import *
from .audit import numerical_contracts
from .lineage import build_all


def fingerprint_scope():
    """Bind the files present at execution; later independent additions are inert.

    Existing module/config changes and removals always invalidate the gate.
    Wiring a new dependency into an executed module changes that module's hash.
    This remains conservative: every existing local module/config is included.
    """
    return {'code_paths':[str(p.relative_to(ROOT)) for p in sorted((ROOT/'g1_pipeline').glob('*.py'))]+['run_pipeline.py'],
            'config_paths':[str(p.relative_to(ROOT)) for p in sorted((ROOT/'configs').glob('*')) if p.is_file()]}


def fingerprint(scope=None):
    scope=fingerprint_scope() if scope is None else scope
    def hashes(names):
        result={}
        for name in names:
            path=ROOT/name
            if not path.is_file():raise RuntimeError('GATE_DEPENDENCY_MISSING:'+name)
            result[name]=sha(path)
        return result
    code=hashes(scope['code_paths']);configs=hashes(scope['config_paths'])
    contract=ROOT/'contracts/g1_robot.json'
    asset=None
    if contract.exists():
        info=json.loads(contract.read_text())
        asset=digest({'xml':sha(info['xml']),
                      'meshes':{p:sha(p) for p in info['resolved_meshes']}})
    return {'code_hash':digest(code),'config_hash':digest(configs),'asset_hash':asset}


def require_pass(stage, _seen=None):
    seen=set() if _seen is None else _seen
    if stage in seen: raise RuntimeError('GATE_DEPENDENCY_CYCLE:'+stage)
    seen.add(stage)
    path=ROOT/f'gates/{stage}.json'
    if not path.exists(): raise RuntimeError('GATE_MISSING:'+stage)
    state=json.loads(path.read_text())
    if state['status']!='PASS': raise RuntimeError(f"GATE_NOT_PASS:{stage}:{state['status']}")
    if any(state.get(k)!=v for k,v in fingerprint(state.get('fingerprint_scope')).items()):
        raise RuntimeError('GATE_FINGERPRINT_STALE:'+stage)
    for name,expected in state.get('input_hashes',{}).items():
        source=ROOT/name
        if not source.is_file() or sha(source)!=expected: raise RuntimeError('GATE_INPUT_STALE:'+name)
    for dependency in state.get('upstream_dependencies',[]): require_pass(dependency,seen.copy())
    return state


def gate(stage,status,reason=None,**kw):
    input_paths=kw.pop('input_paths',[])
    scope=kw.pop('fingerprint_scope',None) or fingerprint_scope()
    inputs={str(Path(p).relative_to(ROOT)):sha(p) for p in input_paths if Path(p).is_file()}
    d={'stage':stage,'status':status,'time':now(),
       'input_manifest_hash':digest(inputs) if inputs else None,
       'input_hashes':inputs,'input_hash_reason':None if inputs else 'No materialized input evidence for this stage',
       **fingerprint(scope),'fingerprint_scope':scope,'actual_quantity':kw.pop('actual_quantity',0),
       'metrics':kw.pop('metrics',{}),'failure_reason':reason,
       'downstream_dependencies':kw.pop('downstream_dependencies',[]),**kw}
    write_json(ROOT/f'gates/{stage}.json',d)
    return d


def audit():
    from .audit import environment,contract_definitions
    gate('00_preflight','RUNNING')
    try:
        environment();contract_definitions();numeric=numerical_contracts()
        return gate('00_preflight',numeric['status'],
            None if numeric['status']=='PASS' else 'Numerical contract verification failed',
            actual_quantity=numeric['count'],metrics={'numeric_contracts':numeric['status'],'scope':numeric['scope']},
            input_paths=[ROOT/'qa/aggregate/numerical_contracts.json',ROOT/'contracts/g1_robot.json'])
    except Exception as exc:
        return gate('00_preflight','FAIL',repr(exc))


def refresh_blocked_stages(namespace='bones_soma_rootweight_dev_v6'):
    """Record actual evidence without promoting execution diagnostics to release gates."""
    if not namespace.replace('_','').isalnum():raise ValueError('INVALID_DIAGNOSTIC_NAMESPACE')
    diagnostic=ROOT/'qa/aggregate'/f'{namespace}.json'
    report=json.loads(diagnostic.read_text()) if diagnostic.exists() else {}
    summary_path=ROOT/'qa/aggregate'/f'{namespace}_qa_summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    metrics={k:report.get(k) for k in ('reference_loaded','retargeted','error_count','asset_equivalence')}
    metrics.update({'source_mapping':namespace+' development candidate',
                    'fixed_panel_size':report.get('fixed_panel_size'),
                    'fixed_dev200_complete':summary.get('fixed_dev200_complete'),
                    'measured_pairwise_grace_pass_count':summary.get('pairwise_grace_pass_count'),
                    'quality_summary':summary.get('aggregate'),
                    'independent_accept_attempted':0})
    gate('01_bones_calibration','BLOCKED',
         'Development evidence only: joint/contact/geometry grace failures remain; configuration, kinematic policy and independent accept300 validation are not complete.',
         actual_quantity=report.get('actual_attempted',0),
         metrics=metrics,
         input_paths=[ROOT/'gates/00_preflight.json',ROOT/f"manifests/bones_pairs_{report.get('run_config',{}).get('panel','dev')}.jsonl",diagnostic,summary_path,ROOT/'audit/locked_target_equivalence.json'],
         upstream_dependencies=['00_preflight'],downstream_dependencies=['02_hml100'],
         diagnostic_only=True)
    for stage,previous,summary in (
        ('02_hml100','01_bones_calibration','hml_smoke100.json'),
        ('03_hiphi100','02_hml100','hiphi_smoke100.json'),
        ('04_fusion_qa','03_hiphi100',None),
        ('05_full_batch','04_fusion_qa',None),
        ('06_training_data_ready','05_full_batch',None),
    ):
        paths=[ROOT/f'gates/{previous}.json']
        if summary: paths.append(ROOT/'qa/aggregate'/summary)
        gate(stage,'BLOCKED',f'{previous} has not passed for the current code/config/asset. Historical downstream diagnostics are not accepted stage outputs.',
             actual_quantity=0,metrics={'accepted_current_version':0,'historical_diagnostic_report':summary},
             input_paths=paths,upstream_dependencies=[previous])


def run():
    # Invalidate previous downstream claims before rebuilding upstream evidence.
    refresh_blocked_stages()
    p=audit()
    if p['status']!='PASS': return p
    inventory=build_all()
    refresh_blocked_stages()
    return {'preflight':p['status'],'next_stage':'01_bones_calibration','status':'BLOCKED',
            'reason':'paired calibration implementation/validation incomplete','inventory_sources':list(inventory)}

if __name__=='__main__': run()
