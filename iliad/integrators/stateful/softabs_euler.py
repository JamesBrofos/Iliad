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
        step_size: float,
        state: SoftAbsLeapfrogState,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Computes the update to the momentum variable using the equations of motion
    determined by the SoftAbs metric.

    Args:
        val: A tuple containing the current guess for the fixed point of the
            momentum, the difference between the momentum at this fixed point
            iteration and the last, and the number of fixed point iterations
            considered so far.
        step_size: The integration step-size.
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
    pm = state.momentum + step_size * f
    delta = pm - pmcand
    num_iters += 1
    return pm, delta, num_iters

def position_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        step_size: float,
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
        step_size: The integration step-size.
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
    qn = state.position + step_size * newvel
    delta = qn - qncand
    num_iters += 1
    return qn, delta, num_iters

def euler_a_single_step(
        distr: Distribution,
        state: SoftAbsLeapfrogState,
        info: SoftAbsLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int,
) -> Tuple[SoftAbsLeapfrogState, SoftAbsLeapfrogInfo]:
    """The Euler-A integrator is a symplectic map that integrates Hamilton's
    equations of motion for a general non-separable
    Hamiltonian. It updates the position implicitly and then computes an
    explicit update to the momentum variable.

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
    # Unpack the position and momentum.
    qo, po = state.position, state.momentum
    num_dims = len(qo)
    # Precompute the initial difference vector, which is set to be an array of
    # infinite values.
    delta = np.inf*np.ones(num_dims)
    # Fixed point iteration to solve the implicit update to the position.
    val = (qo + step_size*state.velocity, delta, 0)
    while cond(val, thresh, max_iters):
        val = position_step(val, step_size, distr, state)

    qn, delta, num_iters = val
    success = np.max(np.abs(delta)) < thresh
    # Update the state with the new position and compute the updated momentum.
    state.position = qn
    state.update(distr)
    state.momentum += step_size*state.force
    info.num_iters_pos += num_iters
    info.success &= success
    return state, info

def euler_b_single_step(
        distr: Distribution,
        state: SoftAbsLeapfrogState,
        info: SoftAbsLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int,
) -> Tuple[SoftAbsLeapfrogState, SoftAbsLeapfrogInfo]:
    """The Euler-B integrator is a symplectic map that integrates Hamilton's
    equations of motion for a general non-separable Hamiltonian. It updates the
    momentum implicitly and then computes an explicit update to the position
    variable.

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
    # Unpack the position and momentum.
    qo, po = state.position, state.momentum
    num_dims = len(qo)
    # Precompute the initial difference vector, which is set to be an array of
    # infinite values.
    delta = np.inf*np.ones(num_dims)
    # Fixed point iteration to solve the implicit update to the momentum.
    val = (po + step_size*state.force, delta, 0)
    while cond(val, thresh, max_iters):
        val = momentum_step(val, step_size, state)

    pn, delta, num_iters = val
    vn = state.inv_metric@pn
    success = np.max(np.abs(delta)) < thresh
    # Update the state's new position.
    state.momentum = pn
    state.velocity = vn
    state.position += step_size*vn
    state.update(distr)
    info.num_iters_mom += num_iters
    info.success &= success
    return state, info

def softabs_euler_a(
        state: SoftAbsLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        thresh: float,
        max_iters: int,
) -> Tuple[SoftAbsLeapfrogState, SoftAbsLeapfrogInfo]:
    state = copy.copy(state)
    info = SoftAbsLeapfrogInfo()
    for i in range(num_steps):
        state, info = euler_a_single_step(
            distr,
            state,
            info,
            step_size,
            thresh,
            max_iters,
        )

    L = np.linalg.cholesky(state.metric)
    state.velocity = state.inv_metric.dot(state.momentum)
    state.sqrtm_metric = L
    state.logdet_metric = 2.0*np.sum(np.log(np.diag(L)))
    return state, info

def softabs_euler_b(
        state: SoftAbsLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        thresh: float,
        max_iters: int,
) -> Tuple[SoftAbsLeapfrogState, SoftAbsLeapfrogInfo]:
    state = copy.copy(state)
    info = SoftAbsLeapfrogInfo()
    for i in range(num_steps):
        state, info = euler_b_single_step(
            distr,
            state,
            info,
            step_size,
            thresh,
            max_iters,
        )

    L = np.linalg.cholesky(state.metric)
    state.velocity = state.inv_metric.dot(state.momentum)
    state.sqrtm_metric = L
    state.logdet_metric = 2.0*np.sum(np.log(np.diag(L)))
    return state, info
