import copy
from typing import Tuple

import numpy as np
import scipy.linalg as spla

from odyssey.distribution import Distribution

from iliad.integrators.info import LagrangianLeapfrogInfo
from iliad.integrators.states import LagrangianLeapfrogState
from iliad.linalg import solve_psd


def christoffel(inv_metric: np.ndarray, jac_metric: np.ndarray) -> np.ndarray:
    """Computes the Christoffel symbols given the inverse metric and the matrix of
    partial derivatives of the metric.

    Args:
        inv_metric: The inverse of the Riemannian metric.
        jac_metric: The partial derivatives of the Riemannian metric.

    Returns:
        C: The Christoffel symbols corresponding to the metric.

    """
    a = np.einsum('im,lmk->ikl', inv_metric, jac_metric)
    b = np.einsum('im,kml->ikl', inv_metric, jac_metric)
    c = np.einsum('im,mkl->ikl', inv_metric, jac_metric)
    C = 0.5*(a + b - c)
    return C

def velocity_step_alt(state: LagrangianLeapfrogState, step_size: float) -> Tuple[np.ndarray, float]:
    """Alternative implementation of the velocity step in the Lagrangian integrator
    that makes more explicit use of the Chistoffel symbols.

    Args:
        state: The state of the Lagrangian dynamics.
        step_size: The integration step-size.

    Returns:
        new_vel: The updated velocity variable.
        logdet: The log-determinant of the transformation of the state variable.

    """
    vel = state.velocity
    Id = np.eye(len(vel))
    grad = -state.grad_log_posterior + 0.5*state.grad_logdet_metric
    nat_grad = state.inv_metric.dot(grad)
    a = vel - step_size*nat_grad
    C = christoffel(state.inv_metric, state.jac_metric.swapaxes(0, -1))
    r = C@vel
    b = Id + step_size*r
    lu, piv = spla.lu_factor(b, check_finite=False)
    ldb = np.sum(np.log(np.diag(np.abs(lu))))
    new_vel = spla.lu_solve((lu, piv), a, check_finite=False, overwrite_b=True)
    s = C@new_vel
    logdet = np.linalg.slogdet(Id - step_size*s)[1] - ldb
    return new_vel, logdet

def velocity_step(state: LagrangianLeapfrogState, step_size: float) -> Tuple[np.ndarray, float]:
    """Function that computes the new velocity variable given a step-size and a
    state object containing the gradient of the log-posterior and the gradient
    of the log-determinant of the metric. We also compute the log-determinant
    of the transformation of the velocity variable.

    Args:
        state: The state of the Lagrangian dynamics.
        step_size: The integration step-size.

    Returns:
        new_vel: The updated velocity variable.
        logdet: The log-determinant of the transformation of the state variable.

    """
    vel = state.velocity
    Id = np.eye(len(vel))
    grad = -state.grad_log_posterior + 0.5*state.grad_logdet_metric
    J = state.jac_metric
    C = 0.5*(J + np.transpose(J, [1, 2, 0]) - np.transpose(J, [2, 1, 0]))
    r = C@vel
    a = state.metric@vel - step_size*grad
    b = state.metric + step_size*r
    lu, piv = spla.lu_factor(b, check_finite=False)
    ldb = np.sum(np.log(np.diag(np.abs(lu))))
    new_vel = spla.lu_solve((lu, piv), a, check_finite=False, overwrite_b=True)
    s = C@new_vel
    logdet = np.linalg.slogdet(state.metric - step_size*s)[1] - ldb
    return new_vel, logdet

def single_step(
        distr: Distribution,
        state: LagrangianLeapfrogState,
        info: LagrangianLeapfrogInfo,
        step_size: float,
        inverted: bool
) -> Tuple[LagrangianLeapfrogState, LagrangianLeapfrogInfo]:
    """Implements a single step of the Lagrangian leapfrog integrator. A flag is
    included to swap the order of integration of velocity and position in order
    to reduce the number of Jacobian determinant computations from four to
    two.

    Args:
        distr: The distribution that guides the time evolution of the Euclidean
            Lagrangian trajectory.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        info: An object that keeps track of the number of fixed point iterations
            and whether or not integration has been successful. For the Lagrange
            integrator, also computes the log-determinant of the transformation.
        step_size: Integration step-size.
        inverted: Whether or not to invert the order of integration.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An augmented information object with the updated number of fixed
            point iterations and boolean indicator for successful integration.

    """
    half_step = 0.5*step_size
    if not inverted:
        vb, logdet = velocity_step(state, half_step)
        state.velocity = vb
        state.position = state.position + step_size*state.velocity
        state.update(distr)
        state.velocity, new_logdet = velocity_step(state, half_step)
        info.logdet += logdet + new_logdet
    else:
        state.position = state.position + half_step*state.velocity
        state.update(distr)
        state.velocity, logdet = velocity_step(state, step_size)
        state.position = state.position + half_step*state.velocity
        info.logdet += logdet
    return state, info

def lagrangian_leapfrog(
        state: LagrangianLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        inverted: bool
) -> Tuple[LagrangianLeapfrogState, LagrangianLeapfrogInfo]:
    """Implements the numerical integrator for Lagrangian Monte Carlo, which averts
    the need for implicit updates but at the cost of requiring four Jacobian
    determinant calculations. By inverting the order of integration, this can
    be reduced to two Jacobian determinant computations.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        distr: The distribution that guides the time evolution of the Euclidean
            Lagrangian trajectory.
        inverted: Whether or not to invert the order of integration.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    state.velocity = state.inv_metric.dot(state.momentum)
    state = copy.copy(state)
    info = LagrangianLeapfrogInfo()
    info.logdet -= state.logdet_metric
    for i in range(num_steps):
        state, info = single_step(distr, state, info, step_size, inverted)

    if inverted:
        state.log_posterior, state.metric = distr.lagrangian_quantities(state.position)
        state.inv_metric, state.sqrtm_metric = solve_psd(state.metric)

    state.momentum = state.metric.dot(state.velocity)
    state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
    info.logdet += state.logdet_metric
    return state, info
