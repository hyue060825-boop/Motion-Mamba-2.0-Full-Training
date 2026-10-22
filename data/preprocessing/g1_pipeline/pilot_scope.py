"""Explicit isolated-pilot authorization; never supplies a production PASS."""
import json
from .common import ROOT, sha
from .calibration_protocol import require_frozen, PATH
from .gates import gate

AUTH = ROOT / 'audit/pilot_v25_authorization.json'


def require_pilot(source):
    if source not in ('humanml3d', 'hiphi'):
        raise ValueError('INVALID_PILOT_SOURCE')
    require_frozen()
    auth = json.loads(AUTH.read_text())
    if auth['scope'] != 'isolated_pilot_only' or auth['policy_sha256'] != sha(PATH):
        raise ValueError('INVALID_PILOT_AUTHORIZATION')
    for name, expected in auth['protected_evidence'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('PILOT_PROTECTED_EVIDENCE_CHANGED:' + name)
    m = auth['fixed_manifests'][source]
    if sha(ROOT / m['path']) != m['sha256']:
        raise ValueError('PILOT_FIXED_MANIFEST_CHANGED')
    if source == 'hiphi':
        p = ROOT / 'gates/pilot/02_hml100.json'
        state = json.loads(p.read_text())
        if state['status'] not in ('PASS', 'FAIL') or state['actual_quantity'] != 100:
            raise RuntimeError('HML_PILOT_100_ATTEMPTS_AND_EVALUATION_REQUIRED')
        if state.get('execution_scope') != 'isolated_pilot':
            raise ValueError('HML_PILOT_SCOPE_REQUIRED')
        for name, expected in state['input_hashes'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('HML_PILOT_EVIDENCE_CHANGED:' + name)
    return auth


def pilot_gate(stage, status, reason=None, **kw):
    if stage not in ('02_hml100', '03_hiphi100'):
        raise ValueError('INVALID_PILOT_STAGE')
    kw['upstream_dependencies'] = []
    kw['execution_scope'] = 'isolated_pilot'
    kw['training_release'] = False
    kw['authorization_sha256'] = sha(AUTH)
    return gate('pilot/' + stage, status, reason, **kw)
