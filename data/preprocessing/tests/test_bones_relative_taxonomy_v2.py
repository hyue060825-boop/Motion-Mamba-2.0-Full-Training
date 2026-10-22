from g1_pipeline.bones_relative_motion_v2 import labels


def test_semantic_categories_keep_evasion_separate_and_unknowns_unassigned():
    assert set(labels({'category':'walking, evading'})) == {'locomotion','balance_evasion'}
    assert labels({'category':'swimming'}) == []
    assert labels({'category':'action','source_motion_id':'unknown_take'}) == []
    assert labels({'category':'crouching idle'}) == ['crouching']


def test_description_disambiguates_without_motion_quality():
    assert labels({'category':'combat','source_description':'person punches themselves in the face'}) == ['upper_body']
    assert labels({'category':'combat','source_description':'unknown'}) == []
    assert labels({'category':'interacting','source_description':'stopping someone with right hand'}) == ['upper_body']
    assert labels({'category':'action','source_motion_id':'body_stretch_2'}) == ['dancing_exercising']
