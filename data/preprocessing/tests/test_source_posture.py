import numpy as np
from g1_pipeline.source_posture import support_evidence


def pose(hip,chest,knees,feet,n=25):
    return [np.tile(np.array(v,dtype=float),(n,)+(1,)*np.asarray(v).ndim) for v in (hip,chest,knees,feet)]


def test_human_standing_and_squat_remain_separate_from_supine_support():
    standing=pose([0,0,.95],[0,0,1.4],[[-.1,0,.5],[.1,0,.5]],[[-.1,0,.05],[.1,0,.05]])
    assert not support_evidence(*standing,1.8)['ground_multisupport_review']
    squat=pose([0,0,.5],[.1,0,.9],[[-.1,.1,.3],[.1,.1,.3]],[[-.1,0,.05],[.1,0,.05]])
    assert not support_evidence(*squat,1.8)['ground_multisupport_review']
    supine=pose([0,0,.1],[.4,0,.15],[[-.1,0,.5],[.1,0,.5]],[[-.1,0,.8],[.1,0,.8]])
    assert support_evidence(*supine,1.8)['ground_multisupport_review']


def test_posture_rule_ignores_world_translation():
    points=pose([0,0,.1],[.4,0,.15],[[-.1,0,.5],[.1,0,.5]],[[-.1,0,.8],[.1,0,.8]])
    shifted=[x+np.array([30.,-15.,10.]) for x in points]
    assert support_evidence(*points,1.8)['ground_multisupport_review']==support_evidence(*shifted,1.8)['ground_multisupport_review']
