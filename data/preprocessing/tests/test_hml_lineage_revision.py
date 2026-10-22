import pytest
from g1_pipeline import hml_lineage_v2 as module


@pytest.mark.parametrize('fps,frames,start,end,expected',[
    (120.,2000,18,218,(108,1308,1307/120)),
    (120.,511,0,86,(0,511,510/120)),
    (59.999988556,1000,1,10,(2,20,19/59.999988556)),
])
def test_native_crop_last_sample_is_inside_exclusive_boundary(monkeypatch,fps,frames,start,end,expected):
    monkeypatch.setattr(module,'caption_intervals',lambda path,interval:[])
    row={'native_interval':{},'provenance_evidence':{'index_row':{'source_path':'ACCAD/take','start_frame':str(start),'end_frame':str(end)}},
         'native_metadata':{'fps':fps,'frames':frames},'caption_source':'example.txt','sample_id':'fixed','project_split':'train'}
    fixed=module.repair(row);interval=fixed['native_interval']
    assert (interval['native_first_index'],interval['native_end_index_exclusive'])==expected[:2]
    assert interval['last_sample_sec']==pytest.approx(expected[2])
    assert interval['last_sample_sec']<interval['end_sec']
    assert fixed['sample_id']==row['sample_id'] and fixed['project_split']==row['project_split']
    assert row['native_interval']=={}
