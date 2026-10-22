import numpy as np
from g1_pipeline.fusion_statistics import Moments,Reservoir


def test_stable_streaming_variance_is_frame_weighted_and_merge_invariant():
    rng=np.random.default_rng(9);x=1e8+rng.normal(size=(101,414));a=Moments();b=Moments()
    a.add(x[:3]);a.add(x[3:57]);b.add(x[57:]);a.merge(b)
    np.testing.assert_allclose(a.mean,x.mean(axis=0),rtol=0,atol=1e-7)
    np.testing.assert_allclose(a.m2/a.count,x.var(axis=0),rtol=1e-7)
    assert a.count==101


def test_quantile_reservoir_bound_and_seed_reproducibility():
    x=np.random.default_rng(2).normal(size=(100,414)).astype(np.float32);a=Reservoir(limit=13);b=Reservoir(limit=13)
    for r in (a,b):r.add(x[:20]);r.add(x[20:])
    assert a.seen==100 and a.values.shape==(13,414)
    np.testing.assert_array_equal(a.values,b.values)
