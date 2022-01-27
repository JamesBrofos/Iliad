import copy
from typing import Callable, Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import RiemannianLeapfrogInfo
from iliad.integrators.states import RiemannianLeapfrogState
from iliad.integrators.terminal import cond
from iliad.linalg import solve_psd
from iliad.integrators.fields import riemannian, softabs
from iliad.integrators.vectors.vector_fields import riemannian_velocity_and_force


def momentum_step(val: Tuple, qo: np.ndarray, po: np.ndarray, half_step: float, force_vector: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the momentum variable."""
    pmcand, _, num_iters = val
    dp = force_vector(qo, pmcand)
    pm = po + half_step*dp
    delta = pm - pmcand
    num_iters += 1
    return pm, delta, num_iters

def position_step(val: Tuple, qo: np.ndarray, po: np.ndarray, half_step: float, velocity_vector: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the position variable."""
    qncand, _, num_iters = val
    dq = velocity_vector(qo, po) + velocity_vector(qncand, po)
    qn = qo + half_step*dq
    delta = qn - qncand
    num_iters += 1
    return qn, delta, num_iters

def single_step(
        velocity_vector: Callable,
        force_vector: Callable,
        state: RiemannianLeapfrogState,
        info: RiemannianLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int
) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
    """Implements a single step of the generalized leapfrog integrator to compute a
    trajectory of a non-separable Hamiltonian. This implementation of the
    generalized leapfrog integrator does not incorporate caching.

    Args:
        velocity_vector: Vector field for the rate of change of the position
            variable.
        force_vector: Vector field for the rate of change of the momentum
            variable.
        state: An object containing the position and momentum variables of the
            state in phase space.
        info: An object that keeps track of the number of fixed point iterations
            and whether or not integration has been successful.
        step_size: Integration step_size.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    # Fixed point iteration and half step-size.
    qo, po = state.position, state.momentum
    delta = np.ones_like(state.position) * np.inf
    half_step = 0.5*step_size
    # Fixed point iteration to determine the first half-step in the momentum.
    val = (po, delta, 0)
    while cond(val, thresh, max_iters):
        val = momentum_step(val, qo, po, half_step, force_vector)
    pm, delta_mom, num_iters_mom = val
    success_mom = np.max(np.abs(delta_mom)) < thresh

    # Fixed point iteration to determine the next position.
    val = (qo, delta, 0)
    while cond(val, thresh, max_iters):
        val = position_step(val, qo, pm, half_step, velocity_vector)
    qn, delta_pos, num_iters_pos = val
    success_pos = np.max(np.abs(delta_pos)) < thresh

    # Final update to the momentum variable.
    pn = pm + half_step*force_vector(qn, pm)
    state.position, state.momentum = qn, pn
    info.num_iters_pos += num_iters_pos
    info.num_iters_mom += num_iters_mom
    info.success &= np.logical_and(success_pos, success_mom)
    return state, info

def riemannian_leapfrog(
        state: RiemannianLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        velocity_vector: Callable,
        force_vector: Callable,
        thresh: float,
        max_iters: int
) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
    """Implements the multiple-step generalized leapfrog integrator (without
    caching) for computing proposals for use in Hamiltonian Monte Carlo.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space.
        step_size: Integration step_size.
        num_steps: Number of integration steps.
        distr: The distribution that guides the time evolution of the Euclidean
            Hamiltonian trajectory.
        velocity_vector: Vector field for the rate of change of the position
            variable.
        force_vector: Vector field for the rate of change of the momentum
            variable.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    state = copy.copy(state)
    info = RiemannianLeapfrogInfo()
    for i in range(num_steps):
        state, info = single_step(velocity_vector, force_vector, state, info, step_size, thresh, max_iters)

    state.update(distr)
    state.logdet_metric = 2*np.sum(np.log(np.diag(state.sqrtm_metric)))
    return state, info
