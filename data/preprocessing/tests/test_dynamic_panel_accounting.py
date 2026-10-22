from g1_pipeline.production_fusion import select_panel,aggregate_dynamic


def fixtures():
    panel=[];tracks=[]
    for source in ('bones','humanml3d','hiphi'):
        for i in range(30):
            sid=f'{source}:{i}'
            panel.append({'sample_id':sid,'dataset':source,'source_family_id':sid,'matching_class':'locomotion','action_labels':['locomotion']})
            for seed in (0,1,2):tracks.append({'sample_id':sid,'seed':seed,'status':'PASS','joint_rmse_rad':.1})
    return panel,tracks


def test_dynamic_seed_runs_cannot_hide_failed_independent_clips():
    panel,tracks=fixtures()
    for row in tracks:
        if row['sample_id'] in {f'hiphi:{i}' for i in range(4)} and row['seed']==0:row['status']='FAIL'
    report=aggregate_dynamic(panel,tracks)
    assert report['status']=='FAIL'
    assert report['source_summary']['hiphi']['success_fraction_all3seeds']==26/30
    assert report['source_summary']['hiphi']['mean_seed_success_fraction']>.95
    assert 'SOURCE_DYNAMIC_SUCCESS_BELOW_90_PERCENT:hiphi' in report['failure_reasons']


def test_missing_seed_blocks_dynamic_acceptance():
    panel,tracks=fixtures();report=aggregate_dynamic(panel,tracks[:-1])
    assert report['status']=='FAIL' and report['incomplete_ids']==['hiphi:29']


def test_panel_mirrors_do_not_count_as_independent_families():
    panel,_=fixtures();duplicates=[dict(r,sample_id=r['sample_id']+'_mirror') for r in panel]
    selected,coverage=select_panel(panel+duplicates)
    assert len(selected)==90
    assert all(len({r['source_family_id'] for r in selected if r['dataset']==source})==30 for source in coverage)
