"""Human-target GMR task refinement with explicit self-penetration penalties.

Development experiment. Root is fixed; bounded physical hinge angles move.
No official robot pose enters the objective. This is not a release solver.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares


def collision_pairs(model):
    from mink import CollisionAvoidanceLimit
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    pairs = CollisionAvoidanceLimit(model, [(geoms, geoms)]).geom_id_pairs
    excluded = set(map(int, model.exclude_signature))
    return sorted((a, b) for a, b in pairs
                  if (min(int(model.geom_bodyid[a]), int(model.geom_bodyid[b])) << 16)
                  + max(int(model.geom_bodyid[a]), int(model.geom_bodyid[b])) not in excluded)


def collision_terms(model, data, pairs, minimum=-.002):
    residual = np.zeros(len(pairs)); jacobian = np.zeros((len(pairs), model.nv))
    fromto = np.empty(6); ja = np.empty((3, model.nv)); jb = np.empty_like(ja)
    for index, (a, b) in enumerate(pairs):
        distance = mujoco.mj_geomDistance(model, data, a, b, .05, fromto)
        if distance >= minimum:
            continue
        normal = fromto[3:]-fromto[:3]
        length = np.linalg.norm(normal)
        if length < 1e-12:
            raise ValueError('UNDEFINED_COLLISION_NORMAL')
        normal /= length
        mujoco.mj_jac(model, data, ja, None, fromto[:3], model.geom_bodyid[a])
        mujoco.mj_jac(model, data, jb, None, fromto[3:], model.geom_bodyid[b])
        # The returned segment reverses for penetrating convex geometries.
        gradient = (1 if distance >= 0 else -1)*normal@(jb-ja)
        residual[index] = minimum-distance
        jacobian[index] = -gradient
    return residual, jacobian


def refine(solver, frame, seed, *, penalty=2000., posture_weight=5., max_nfev=150,
           previous_correction=None, correction_velocity_weight=0.):
    solver.update_targets(frame, offset_to_ground=False)
    model = solver.model; configuration = solver.configuration
    if model.nq != 36 or model.nv != 35:
        raise ValueError('UNEXPECTED_G1_CONTRACT')
    pairs = collision_pairs(model); q = seed.copy(); cached = None; output = None
    if previous_correction is not None:
        previous_correction = np.asarray(previous_correction)
        if previous_correction.shape != (29,) or not np.isfinite(previous_correction).all():
            raise ValueError('INVALID_PREVIOUS_CORRECTION')
    def evaluate(x):
        nonlocal cached, output
        if cached is not None and np.array_equal(x, cached):
            return output
        q[7:] = x; configuration.update(q)
        errors = []; jacobians = []
        for task in solver.tasks2:
            errors.append(task.cost*task.compute_error(configuration))
            jacobians.append(task.cost[:, None]*task.compute_jacobian(configuration)[:, 6:])
        collision, jac = collision_terms(model, configuration.data, pairs)
        errors.extend([penalty*collision, posture_weight*(x-seed[7:])])
        jacobians.extend([penalty*jac[:, 6:], posture_weight*np.eye(29)])
        if previous_correction is not None:
            weight = correction_velocity_weight/.04
            errors.append(weight*(x-seed[7:]-previous_correction))
            jacobians.append(weight*np.eye(29))
        cached = x.copy(); output = np.concatenate(errors), np.concatenate(jacobians)
        return output
    configuration.update(seed)
    before = collision_terms(model, configuration.data, pairs)[0]
    # GMR QP tolerances can leave a seed a few ulps outside a hinge range.
    # SciPy requires a strictly feasible initial point even for those values.
    excess = max(0., float(np.max(model.jnt_range[1:, 0]-seed[7:])),
                 float(np.max(seed[7:]-model.jnt_range[1:, 1])))
    if excess > 1e-4:
        raise ValueError('SEED_EXCEEDS_NUMERIC_HINGE_LIMIT_TOLERANCE')
    proposed = seed[7:] if previous_correction is None else seed[7:]+previous_correction
    initial = np.clip(proposed, model.jnt_range[1:, 0]+1e-12, model.jnt_range[1:, 1]-1e-12)
    fit = least_squares(lambda x: evaluate(x)[0], initial, jac=lambda x: evaluate(x)[1],
                        bounds=(model.jnt_range[1:, 0], model.jnt_range[1:, 1]),
                        max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    q[7:] = fit.x; configuration.update(q)
    after = collision_terms(model, configuration.data, pairs)[0]
    return q.copy(), {'objective': float(2*fit.cost), 'nfev': fit.nfev,
                      'optimizer_success': bool(fit.success), 'message': fit.message,
                      'maximum_penetration_excess_before_m': float(before.max(initial=0)),
                      'maximum_penetration_excess_after_m': float(after.max(initial=0)),
                      'hinge_correction_max_deg': float(np.rad2deg(abs(q[7:]-seed[7:])).max()),
                      'hinge_correction_mae_deg': float(np.rad2deg(abs(q[7:]-seed[7:])).mean()),
                      'collision_pairs': len(pairs), 'reference_input': False,
                      'seed_limit_excess_rad': excess,
                      'initial_bound_projection_max_rad': float(np.max(abs(initial-proposed))),
                      'root_preserved': bool(np.array_equal(q[:7], seed[:7]))}
