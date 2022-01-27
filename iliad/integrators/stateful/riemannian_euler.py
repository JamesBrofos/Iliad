import copy
from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import RiemannianLeapfrogInfo
from iliad.integrators.states import RiemannianLeapfrogState
from iliad.integrators.terminal import cond
from iliad.integrators.fields import riemannian
from iliad.linalg import solve_psd


def position_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        step_size: float,
        distr: Distribution,
        state: RiemannianLeapfrogState
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the position variable.

    Args:
        val: Tuple containing the fixed point position, the
            iteration-over-iteration change, and the number of fixed point
            iterations computed so far.
        step_size: The integration step-size.
        distr: The distribution that guides the time evolution of the Euclidean
            Hamiltonian trajectory.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.

    Returns:
        qn: The updated position variable.
        delta: The iteration-over-iteration difference in the position.
        num_iters: The number of fixed point iterations incremented by one.

    """
    qncand, _, num_iters = val
    G = distr.riemannian_metric(qncand)
    newvel, L = solve_psd(G, state.momentum)
    qn = state.position + step_size * newvel
    delta = qn - qncand
    num_iters += 1
    return qn, delta, num_iters

def momentum_step(
        val: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
        step_size: float,
        state: RiemannianLeapfrogState
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the momentum variable.

    Args:
        val: Tuple containing the fixed point momentum and velocity, the
            iteration-over-iteration change, and the number of fixed point
            iterations computed so far.
        step_size: The integration step-size.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.

    Returns:
         pm: The updated momentum.
         vm: The updated velocity.
         delta: The iteration-over-iteration difference in the momentum.
         num_iters: The number of fixed point iterations incremented by one.

    """
    pmcand, _, _, num_iters = val
    # Compute the gradient of the Hamiltonian with respect to position.
    vm = riemannian.velocity(state.inv_metric, pmcand)
    f = riemannian.force(vm, state.grad_log_posterior, state.jac_metric, state.grad_logdet_metric)
    pm = state.momentum + step_size * f
    delta = pm - pmcand
    num_iters += 1
    return pm, vm, delta, num_iters

def euler_a_single_step(
        distr: Distribution,
        state: RiemannianLeapfrogState,
        info: RiemannianLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int,
) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
    """The Euler-A integrator is a symplectic map that integrates Hamilton's
    equations of motion for a general non-separable Hamiltonian. It updates the
    position implicitly and then computes an explicit update to the momentum
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
        state: RiemannianLeapfrogState,
        info: RiemannianLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int,
) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
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
    val = (po + step_size*state.force, delta, delta, 0)
    while cond(val, thresh, max_iters):
        val = momentum_step(val, step_size, state)

    pn, vn, delta, num_iters = val
    success = np.max(np.abs(delta)) < thresh
    # Update the state's new position.
    state.momentum = pn
    state.velocity = vn
    state.position += step_size*vn
    state.update(distr)
    info.num_iters_mom += num_iters
    info.success &= success
    return state, info

def riemannian_euler_a(
        state: RiemannianLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        thresh: float,
        max_iters: int,
) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
    state = copy.copy(state)
    info = RiemannianLeapfrogInfo()
    for i in range(num_steps):
        state, info = euler_a_single_step(
            distr,
            state,
            info,
            step_size,
            thresh,
            max_iters,
        )

    state.velocity = state.inv_metric.dot(state.momentum)
    state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
    return state, info

def riemannian_euler_b(
        state: RiemannianLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        thresh: float,
        max_iters: int,
) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
    state = copy.copy(state)
    info = RiemannianLeapfrogInfo()
    for i in range(num_steps):
        state, info = euler_b_single_step(
            distr,
            state,
            info,
            step_size,
            thresh,
            max_iters,
        )

    state.velocity = state.inv_metric.dot(state.momentum)
    state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
    return state, info

