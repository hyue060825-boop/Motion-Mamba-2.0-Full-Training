import itertools
import numpy as np
from scipy.spatial.transform import Rotation as R
from g1_pipeline.common import ROOT
from g1_pipeline.locked_target import create_retargeter
from g1_pipeline.gmr_arm_trajectory_seed import ArmChain
from g1_pipeline.gmr_arm_branch_path import shoulder_branches,choose_path


def test_analytic_shoulder_branches_reconstruct_compiled_chain():
    solver=create_retargeter(ROOT/'configs/bones_soma_rootweight_v6.json',ROOT/'assets/project_g1_v1/g1.xml')
    rng=np.random.default_rng(43)
    for side in ('left','right'):
        chain=ArmChain(solver.model,side)
        for _ in range(40):
            q=rng.uniform(chain.bounds[:,0],chain.bounds[:,1]);parent=R.random(random_state=rng).as_matrix()
            _,rot=chain.fk(q[None],np.zeros((1,3)),parent[None])
            branches=shoulder_branches(chain,parent,rot[0,0]);assert branches
            assert any(np.max(abs(branch-q[:3]))<1e-8 for branch in branches)
            for branch in branches:
                candidate=q.copy();candidate[:3]=branch
                _,actual=chain.fk(candidate[None],np.zeros((1,3)),parent[None])
                np.testing.assert_allclose(actual[0,0],rot[0,0],atol=1e-10)


def test_global_branch_path_matches_exhaustive_search():
    candidates=[np.array([[0.],[3.]]),np.array([[2.9],[.1]]),np.array([[3.1],[.2]])]
    costs=[np.array([0.,1.]),np.array([0.,.2]),np.array([0.,.2])]
    _,path,total=choose_path(candidates,costs,step_weight=2.)
    all_paths=[]
    for indices in itertools.product(range(2),repeat=3):
        objective=sum(costs[i][j] for i,j in enumerate(indices))
        objective+=4*sum(float(np.sum((candidates[i][indices[i]]-candidates[i-1][indices[i-1]])**2)) for i in range(1,3))
        all_paths.append((objective,list(indices)))
    optimum=min(all_paths)
    np.testing.assert_allclose(total,optimum[0]);assert path==optimum[1]
