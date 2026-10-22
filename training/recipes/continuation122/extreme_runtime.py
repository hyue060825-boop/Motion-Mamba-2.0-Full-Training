"""Admit tested scheduling optimizations without modifying the learning recipe."""
from common import *
import contextlib
sys.path.insert(0,str(ROOT/'acceleration_v5'))
import layer_graph, geometry_fast, valid_scope

_mode='';_enabled=False

def install(c):
    global _mode
    path=ROOT/'acceleration_extreme/ADMISSION.json'
    if not path.exists():return
    record=read(path);assert record['passed']
    for filename,digest in record['source_hashes'].items():
        assert sha(ROOT/filename)==digest,('Unverified accelerated source',filename)
    gate=read(record['full_update_gate'])
    assert gate['checks'] and all(row['passed'] for row in gate['checks'])
    assert all(row['input_state_exact'] and row['weights_exact'] and row['Adam_exact']
               and row['rng_exact'] and row['clip_exact'] for row in gate['checks'])
    _mode=record['mode']
    if 'layer_graph' in _mode:layer_graph.install(c)
    if 'valid_scope' in _mode:valid_scope.install()

def enable(value):
    global _enabled
    _enabled=bool(value and _mode)
    layer_graph.enable(_enabled and 'layer_graph' in _mode)
    valid_scope.begin(_enabled and 'valid_scope' in _mode)

def positions_only(raw):
    return geometry_fast.positions_only(raw.motion_rep,_enabled and 'geometry' in _mode)

def stats():
    return dict(mode=_mode,enabled=_enabled,layer=layer_graph.stats(),validity=valid_scope.stats())

def end():
    global _enabled
    valid_scope.end();_enabled=False
