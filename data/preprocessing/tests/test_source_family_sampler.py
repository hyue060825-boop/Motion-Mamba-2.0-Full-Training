from collections import Counter
from g1_pipeline.training_data import SourceFamilySampler


def test_source_sampling_not_dominated_by_large_source_or_mirror_family():
    records=[]
    for source,count in [('bones',100),('humanml3d',10),('hiphi',2)]:
        for family in range(count):
            for mirror in range(10 if family==0 else 1):
                records.append({'dataset':source,'source_family_id':f'{source}:{family}'})
    sampler=SourceFamilySampler(records,samples=12000)
    draws=list(sampler);counts=Counter(records[i]['dataset'] for i,_ in draws)
    assert all(3700<counts[s]<4300 for s in counts)
    hiphi=Counter(records[i]['source_family_id'] for i,_ in draws if records[i]['dataset']=='hiphi')
    assert .45<hiphi['hiphi:0']/sum(hiphi.values())<.55
    assert draws==list(sampler)
    sampler.set_epoch(1)
    assert draws!=list(sampler)
