import multiprocessing as mp
import os
import json
import pytest
from g1_pipeline.batch_state import State
from g1_pipeline.common import sha

ROW={'sample_id':'fixture:1','dataset_release':'v1','source_motion_id':'one','mirror_flag':False,'native_start_sec':0,'native_end_sec':1}


def crash(directory):
    state=State(directory,{'test':'fixed'})
    with state.own('fixture:1') as row:
        state.heartbeat(row['sample_id'],'EXPORTED')
        # Simulate interruption after temporary artifact write, before manifest
        # commit. OS exit releases the per-item file lock.
        (state.directory/'half.tmp').write_text('partial')
        os._exit(7)


def test_interrupted_process_releases_item_and_resume_commits_once(tmp_path):
    state=State(tmp_path,{'test':'fixed'});state.register([ROW],'semantic-v1');state.close()
    process=mp.get_context('fork').Process(target=crash,args=(tmp_path,));process.start();process.join(10)
    assert process.exitcode==7
    state=State(tmp_path,{'test':'fixed'})
    with state.own('fixture:1',resume=True) as row:
        assert row==ROW
        evidence=tmp_path/'rejection.json';evidence.write_text(json.dumps({'reason':'fixture unsupported'}))
        state.commit(row['sample_id'],'UNSUPPORTED',{'artifact_hashes':{str(evidence):sha(evidence)}})
    with state.own('fixture:1',resume=True) as row:assert row is None
    assert len(state.rows())==1 and state.rows()[0]['attempt']==2
    assert state.rows()[0]['status']=='UNSUPPORTED'
    evidence.write_text('changed')
    with state.own('fixture:1',resume=True) as row:assert row==ROW
    state.close()


def test_changed_batch_namespace_never_reuses_previous_work(tmp_path):
    State(tmp_path,{'code':'one'}).close()
    with pytest.raises(ValueError,match='FINGERPRINT_CHANGED'):State(tmp_path,{'code':'two'})
