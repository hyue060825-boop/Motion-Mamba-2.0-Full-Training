"""Development GMR convergence wrapper with the actual weighted task objective.

Uses GMR's targets, Mink QP, damping, limits and two stages unchanged. Unlike
upstream's unweighted SE(3) stopping norm, this checks the objective being solved.
No official robot reference is available to this solver.
"""
import numpy as np


def cost(configuration, tasks):
    return float(sum(np.sum((task.cost*task.compute_error(configuration))**2) for task in tasks))


def converge(solver, tasks, *, max_iterations=50, relative_tolerance=1e-6):
    import mink
    c=solver.configuration;dt=float(solver.model.opt.timestep)
    initial=previous=cost(c,tasks);history=[initial];backtracks=0;reason='iteration_budget'
    for _ in range(max_iterations):
        q=c.q.copy()
        velocity=mink.solve_ik(c,tasks,dt,solver.solver,solver.damping,solver.ik_limits)
        accepted=False
        for factor in (1.,.5,.25,.125,.0625,.03125):
            c.update(q);c.integrate_inplace(velocity*factor,dt)
            value=cost(c,tasks)
            if np.isfinite(value) and value<=previous:
                accepted=True;break
            backtracks+=1
        if not accepted:
            c.update(q);reason='no_descent';break
        history.append(value)
        improvement=previous-value;previous=value
        if improvement<=relative_tolerance*max(1.,previous):
            reason='relative_objective_convergence';break
    return {'initial_cost':initial,'final_cost':history[-1],'accepted_iterations':len(history)-1,
            'backtracks':backtracks,'reason':reason,'cost_history':history,
            'solver_dt_s':dt,'physical_frame_dt_s':.04}


def retarget(solver, human_frame, settings):
    solver.update_targets(human_frame,offset_to_ground=False);reports=[]
    for stage in (1,2):
        if getattr(solver,f'use_ik_match_table{stage}'):
            report=converge(solver,getattr(solver,f'tasks{stage}'),**settings)
            report['stage']=stage;reports.append(report)
    return solver.configuration.q.copy(),reports
