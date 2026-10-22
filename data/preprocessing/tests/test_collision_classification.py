import numpy as np
from g1_pipeline.bones_pair_qa import scene,replay
from g1_pipeline.robot import compile_target


def test_floor_is_not_duplicated_or_counted_as_self_collision():
    m=scene();reference,_=compile_target()
    assert m.ngeom==reference.ngeom
    q=m.qpos0.copy();q[2]-=.5
    _,_,metrics=replay(m,q[None])
    assert metrics['ground_penetration_max_m']>.2
    assert all('world' not in key for key in metrics['raw_self_contact_pair_counts'])
