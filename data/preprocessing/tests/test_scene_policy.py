from g1_pipeline.scene_policy import classify


def test_crouch_without_props_is_only_candidate():
    result=classify({'body_position':'crouching','props':'none','category':'crouching'})
    assert result['scene_class']=='flat_ground_core_candidate'
    assert result['core_eligible']
    assert 'kinematic QA still required' in result['reason']


def test_external_support_is_not_flat_ground():
    for take in ('climb_ladder_001','sit_chair_001','climb_stairs_001','swimming_001'):
        result=classify({'body_position':'standing','props':'none','take_id':take})
        assert result['scene_class']=='external_support_or_nonflat'
        assert not result['core_eligible']


def test_unknown_or_object_metadata_never_auto_passes_core():
    assert classify({})['scene_class']=='unclassified'
    assert not classify({})['core_eligible']
    result=classify({'body_position':'standing','props':'small item','category':'gesture'})
    assert result['scene_class']=='object_interaction_review'
    assert not result['core_eligible']


def test_dynamic_action_metadata_overrides_initial_standing_posture():
    for category in ('standing, lying down', 'falling', 'flipping'):
        result=classify({'body_position':'standing','props':'none','category':category,'take_id':'take_1'})
        assert result['scene_class']=='ground_multisupport'
        assert not result['core_eligible']
    assert classify({'body_position':'standing','take_id':'cartwheelin_001'})['scene_class']=='ground_multisupport'


def test_ordinary_jump_and_upper_body_do_not_require_invented_support():
    assert classify({'body_position':'standing','category':'jumping','props':'none'})['core_eligible']
    assert classify({'body_position':'standing','category':'gesture','props':'none'})['core_eligible']
    assert classify({'body_position':'standing','category':'climbing stairs','props':'none'})['scene_class']=='external_support_or_nonflat'


def test_technical_description_resolves_incomplete_standing_tag():
    for text in ('alternate raising legs while lying on the floor',
                 'person standing sits down on the floor with their legs crossed'):
        result=classify({'body_position':'standing','category':'exercising','source_description':text})
        assert result['scene_class']=='ground_multisupport'
        assert result['evidence_field']=='source_description'
    assert classify({'body_position':'standing','source_description':'stands on the floor and waves both hands'})['core_eligible']
    assert classify({'body_position':'standing, crouching','source_description':'a slavic squat with knees to the sides'})['core_eligible']
    assert classify({'body_position':'standing','source_description':'sits down on a chair'})['scene_class']=='external_support_or_nonflat'
    assert classify({'body_position':'standing','props':'bed','source_description':'lies down'})['scene_class']=='external_support_or_nonflat'
