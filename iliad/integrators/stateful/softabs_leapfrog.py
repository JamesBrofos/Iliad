import copy
from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import SoftAbsLeapfrogInfo
from iliad.integrators.states import SoftAbsLeapfrogState
from iliad.integrators.terminal import cond
from iliad.integrators.fields import riemannian, softabs


def momentum_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        half_step: float,
        state: SoftAbsLeapfrogState,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Computes the update to the momentum variable using the equations of motion
    determined by the SoftAbs metric.

    Args:
        val: A tuple containing the current guess for the fixed point of the
            momentum, the difference between the momentum at this fixed point
            iteration and the last, and the number of fixed point iterations
            considered so far.
        half_step: One-half the integration step-size.
        state: The current state of the SoftAbs metric system.

    Returns:
        pm: The updated momentum variable.
        delta: The difference between the updated momentum variable and the
            guess.
        num_iters: The number of fixed point iterations attempted so far.

    """
    pmcand, _, num_iters = val
    f = softabs.force(pmcand,
                      state.grad_log_posterior,
                      state.jac_hessian,
                      state.hessian_eigenvals,
                      state.softabs_eigenvals,
                      state.softabs_inv_eigenvals,
                      state.hessian_eigenvecs,
                      state.alpha)
    pm = state.momentum + half_step * f
    delta = pm - pmcand
    num_iters += 1
    return pm, delta, num_iters

def position_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        half_step: float,
        distr: Distribution,
        state: SoftAbsLeapfrogState,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Computes the update to the position variable using the equations of motion
    determined by the SoftAbs metric.

    Args:
        val: A tuple containing the current guess for the fixed point of the
            position, the difference between the position at this fixed point
            iteration and the last, and the number of fixed point iterations
            considered so far.
        half_step: One-half the integration step-size.
        distr: The distribution that guides the time evolution of the Euclidean
            Hamiltonian trajectory.
        state: The current state of the SoftAbs metric system.

    Returns:
        qn: The updated momentum variable.
        delta; The difference between the updated position variable and the
            guess.
        num_iters: The number of fixed point iterations attempted so far.

    """
    qncand, _, num_iters = val
    H = distr.hessian(qncand)
    l, U, lt, inv_lt, metric, inv_metric = softabs.decomposition(H, state.alpha)
    newvel = inv_metric@state.momentum
    midvel = state.velocity + newvel
    qn = state.position + half_step * midvel
    delta = qn - qncand
    num_iters += 1
    return qn, delta, num_iters

def single_step(
        distr: Distribution,
        state: SoftAbsLeapfrogState,
        info: SoftAbsLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int
) -> Tuple[SoftAbsLeapfrogState, SoftAbsLeapfrogInfo]:
    """Implements a single step of the SoftAbs leapfrog integrator which is
    designed to provide a general metric for Riemannian manifold Hamiltonian
    Monte Carlo.

    Args:
        distr: The distribution that guides the time evolution of the Euclidean
            Hamiltonian trajectory.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        info: An object that keeps track of the number of fixed point iterations
            and whether or not integration has been successful.
        step_size: Integration step-size.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An augmented information object with the updated number of fixed
            point iterations and boolean indicator for successful integration.

    """
    # Precompute the half step-size and the number of dimensions of the
    # position variable. Extract the position and momentum variables.
    half_step = 0.5 * step_size
    qo, po = state.position, state.momentum
    num_dims = len(qo)

    # Precompute the initial difference vector, which is set to be an array of
    # infinite values.
    delta = np.inf*np.ones(num_dims)
    val = (po + half_step*state.force, delta, 0)
    while cond(val, thresh, max_iters):
        val = momentum_step(val, half_step, state)

    pm, delta_mom, num_iters_mom = val
    success_mom = np.max(np.abs(delta_mom)) < thresh
    state.velocity = state.inv_metric@pm
    state.momentum = pm

    val = (qo + step_size*state.velocity, delta, 0)
    while cond(val, thresh, max_iters):
        val = position_step(val, half_step, distr, state)

    qn, delta_pos, num_iters_pos = val
    success_pos = np.max(np.abs(delta_pos)) < thresh
    state.position = qn

    # Final explicit update to the momentum variable.
    state.update(distr)
    state.momentum += half_step*state.force

    # Determine if integration was successful (to the desired precision).
    success = np.logical_and(success_mom, success_pos)
    # Update the information on the current state by incrementing the total
    # number of iterations and whether or not the most recent step had
    # successful integration.
    info.num_iters_pos += num_iters_pos
    info.num_iters_mom += num_iters_mom
    info.success &= success
    return state, info

def softabs_leapfrog(
        state: SoftAbsLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        thresh: float,
        max_iters: int
) -> Tuple[SoftAbsLeapfrogState, SoftAbsLeapfrogInfo]:
    """Implements the SoftAbs leapfrog integrator.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        hessian: Function to compute the Hessian of the log-posterior.
        auxiliaries: Function to compute the log-posterior, the gradient of the
            log-posterior, the Hessian, and the Jacobian of the Hessian.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    state = copy.copy(state)
    info = SoftAbsLeapfrogInfo()
    for i in range(num_steps):
        state, info = single_step(
            distr,
            state,
            info,
            step_size,
            thresh,
            max_iters
        )

    L = np.linalg.cholesky(state.metric)
    state.velocity = state.inv_metric.dot(state.momentum)
    state.sqrtm_metric = L
    state.logdet_metric = 2.0*np.sum(np.log(np.diag(L)))
    return state, info
