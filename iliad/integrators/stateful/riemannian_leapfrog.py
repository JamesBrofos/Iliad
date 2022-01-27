import copy
from typing import Tuple

import numpy as np
import scipy.linalg as spla

from odyssey.distribution import Distribution

from iliad.integrators.info import RiemannianLeapfrogInfo
from iliad.integrators.states import RiemannianLeapfrogState
from iliad.integrators.terminal import cond
from iliad.integrators.fields import riemannian
from iliad.linalg import solve_psd


def momentum_step(
        val: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
        half_step: float,
        state: RiemannianLeapfrogState
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the momentum variable.

    Args:
        val: Tuple containing the fixed point momentum and velocity, the
            iteration-over-iteration change, and the number of fixed point
            iterations computed so far.
        half_step: One-half the integration step-size.
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
    pm = state.momentum + half_step * f
    delta = pm - pmcand
    num_iters += 1
    return pm, vm, delta, num_iters

def newton_momentum_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        o: np.ndarray,
        E: np.ndarray,
        Id: np.ndarray,
        half_step: float,
        state: RiemannianLeapfrogState,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Updates the momentum variable by finding the solution to a root-finding
    problem via Newton's method. Newton's method has a faster order of
    convergence than fixed point iteration but has a greater computational
    burden, since it involves inverting the Jacobian of the function whose root
    is sought.

    Args:
        val: Tuple containing the fixed point momentum and velocity, the
            iteration-over-iteration change, and the number of fixed point
            iterations computed so far.
        o: Precomputed component of the gradient of the Hamiltonian with respect
            to the position.
        E: Precomputed tensor containing the product of the inverse metric with
            the product of the Jacobian of the metric and the inverse metric.
        half_step: One-half the integration step-size.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.

    Returns:
         pm: The updated momentum.
         delta: The iteration-over-iteration difference in the momentum.
         num_iters: The number of fixed point iterations incremented by one.

    """
    pmcand, _, num_iters = val
    po = state.momentum
    f = o + 0.5*pmcand@E@pmcand
    g = pmcand - (po + half_step*f)
    J = Id - half_step*E@pmcand
    x, resid, rank, svals = spla.lstsq(J, g, overwrite_a=True, overwrite_b=True, check_finite=False, lapack_driver='gelsy')
    pm = pmcand - x
    delta = pm - pmcand
    num_iters += 1
    return pm, delta, num_iters

def position_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        half_step: float,
        distr: Distribution,
        state: RiemannianLeapfrogState
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Function to find the fixed point of the position variable.

    Args:
        val: Tuple containing the fixed point position, the
            iteration-over-iteration change, and the number of fixed point
            iterations computed so far.
        half_step: One-half the integration step-size.
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
    sumvel = state.velocity + newvel
    qn = state.position + half_step * sumvel
    delta = qn - qncand
    num_iters += 1
    return qn, delta, num_iters

def newton_position_step(
        val: Tuple[np.ndarray, np.ndarray, int],
        half_step: float,
        distr: Distribution,
        Id: np.ndarray,
        state: RiemannianLeapfrogState,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Computes the update to the position variable using Newton's method instead
    of fixed point iteration. Newton's method has a higher order of convergence
    so that fewer iterations are required. On the other hand, each iteration of
    Newton requires a greater computational effort, so there is a trade-off
    between convergence speed and computational expediency.

    Args:
        val: Tuple containing the fixed point position, the
            iteration-over-iteration change, and the number of fixed point
            iterations computed so far.
        half_step: One-half the integration step-size.
        distr: The distribution that guides the time evolution of the Euclidean
            Hamiltonian trajectory.
        Id: The identity matrix, which is useful to precompute for computational
            purposes.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.

    Returns:
        qn: The updated position variable.
        delta: The iteration-over-iteration difference in the position.
        num_iters: The number of fixed point iterations incremented by one.

    """
    qncand, _, num_iters = val
    G, dG = distr.riemannian_metric_and_jacobian(qncand)
    iG, _ = solve_psd(G, Id)
    E = np.einsum('ij,jkl->ikl', iG, np.einsum('ijk,jl->ilk', dG, iG))
    Ep = np.einsum('ijk,j->ik', E, state.momentum)
    J = Id + half_step*Ep
    g = qncand - state.position - half_step*(iG@state.momentum + state.velocity)
    x, resid, rank, svals = spla.lstsq(J, g, overwrite_a=True, overwrite_b=True, check_finite=False, lapack_driver='gelsy')
    qn = qncand - x
    delta = qn - qncand
    num_iters += 1
    return qn, delta, num_iters

def single_step(
        distr: Distribution,
        state: RiemannianLeapfrogState,
        info: RiemannianLeapfrogInfo,
        step_size: float,
        thresh: float,
        max_iters: int,
        newton_momentum: bool,
        newton_position: bool
) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
    """Implements a single step of the generalized leapfrog integrator, which
    involves an implicitly-defined update of the momentum, an implicit update
    to position, and a second explicit update to momentum.

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
        newton_momentum: Whether or not to enable Newton iterations for the
            momentum fixed point equation.
        newton_position: Whether or not to enable Newton iterations for the
            position fixed point equation.

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

    # The first step of the integrator is to find a fixed point of the momentum
    # variable.
    if newton_momentum:
        val = (po + half_step*state.force, delta, 0)
        Id = np.eye(num_dims)
        A = np.einsum('ijk,jl->ilk', state.jac_metric, state.inv_metric)
        E = np.einsum('il,ljk->kij', state.inv_metric, A)
        o = state.grad_log_posterior - state.grad_logdet_metric
        while cond(val, thresh, max_iters):
            val = newton_momentum_step(val, o, E, Id, half_step, state)
        pm, delta_mom, num_iters_mom = val
        vm = riemannian.velocity(state.inv_metric, pm)
    else:
        val = (po + half_step*state.force, delta, delta, 0)
        while cond(val, thresh, max_iters):
            val = momentum_step(val, half_step, state)
        pm, vm, delta_mom, num_iters_mom = val

    success_mom = np.max(np.abs(delta_mom)) < thresh
    state.velocity = vm
    state.momentum = pm

    # The second step of the integrator is to find a fixed point of the
    # position variable. The first momentum gradient could be conceivably
    # cached and saved.
    val = (qo + step_size*state.velocity, delta, 0)
    if not newton_position:
        while cond(val, thresh, max_iters):
            val = position_step(val, half_step, distr, state)
    else:
        Id = np.eye(num_dims)
        while cond(val, thresh, max_iters):
            val = newton_position_step(val, half_step, distr, Id, state)

    qn, delta_pos, num_iters_pos = val
    success_pos = np.max(np.abs(delta_pos)) < thresh
    state.position = qn

    # Last step is to do an explicit half-step of the momentum variable.
    state.update(distr)
    state.momentum += half_step*state.force
    info.kl += 0.5*delta_pos@state.metric@delta_pos

    # Determine if integration was successful (to the desired precision).
    success = np.logical_and(success_mom, success_pos)
    # Update the information on the current state by incrementing the total
    # number of iterations and whether or not the most recent step had
    # successful integration.
    info.num_iters_pos += num_iters_pos
    info.num_iters_mom += num_iters_mom
    info.success &= success
    return state, info

def riemannian_leapfrog(
        state: RiemannianLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        thresh: float,
        max_iters: int,
        newton_momentum: bool,
        newton_position: bool
) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
    """Implements the generalized leapfrog integrator which avoids recomputing
    redundant quantities at each iteration.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        step_size: Integration step-size.
        num_steps: Number of integration steps.
        distr: The distribution that guides the time evolution of the Euclidean
            Hamiltonian trajectory.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.
        newton_momentum: Whether or not to enable Newton iterations for the
            momentum fixed point equation.
        newton_position: Whether or not to enable Newton iterations for the
            position fixed point equation.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    state = copy.copy(state)
    info = RiemannianLeapfrogInfo()
    for i in range(num_steps):
        state, info = single_step(
            distr,
            state,
            info,
            step_size,
            thresh,
            max_iters,
            newton_momentum,
            newton_position
        )

    state.velocity = state.inv_metric.dot(state.momentum)
    state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
    return state, info
