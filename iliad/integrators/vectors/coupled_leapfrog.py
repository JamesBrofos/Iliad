import copy
from typing import Callable, Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.fields import softabs
from iliad.integrators.info import CoupledInfo
from iliad.integrators.terminal import cond
from iliad.integrators.states.coupled_state import CoupledState
from iliad.linalg import solve_psd, sqrtm


def phi_a(qn: np.ndarray, xn: np.ndarray, pn: np.ndarray, yn: np.ndarray, step_size: float, vector_field: Callable) -> Tuple[np.ndarray]:
    vel, force = vector_field(qn, yn)
    dxn = step_size * vel
    dpn = step_size * force
    return qn, xn + dxn, pn + dpn, yn

def phi_b(qn: np.ndarray, xn: np.ndarray, pn: np.ndarray, yn: np.ndarray, step_size: float, vector_field: Callable) -> Tuple[np.ndarray]:
    vel, force = vector_field(xn, pn)
    dqn = step_size * vel
    dyn = step_size * force
    return qn + dqn, xn, pn, yn + dyn

def phi_c(qn: np.ndarray, xn: np.ndarray, pn: np.ndarray, yn: np.ndarray, step_size: float, omega: float) -> Tuple[np.ndarray]:
    cos = np.cos(2*omega*step_size)
    sin = np.sin(2*omega*step_size)
    add = np.vstack([qn + xn, pn + yn])
    qnmxn, pnmyn = qn - xn, pn - yn
    Rsub = np.vstack((
        np.hstack((cos*qnmxn + sin*pnmyn)),
        np.hstack((-sin*qnmxn + cos*pnmyn))))
    qnpn = 0.5*(add + Rsub).ravel()
    xnyn = 0.5*(add - Rsub).ravel()
    (qn, pn), (xn, yn) = np.split(qnpn, 2), np.split(xnyn, 2)
    return qn, xn, pn, yn

def coupled_integrate(
        vector_field: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        omega: float
) -> Tuple[np.ndarray]:
    """Implements the explicit integrator for non-separable Hamiltonian dynamics.
    The coupled explicit integrator is composed of three component integration
    steps.

    Args:
        vector_field: A function returning the time derivatives of position and
            momentum.
        force_vector: Function computing the time derivative of momentum.
        zo: Tuple containing the position and momentum variables in the expanded
            phase space.
        step_size: Integration step_size.
        omega: Binding strength between the two approximate solutions.

    Returns:
        qn: Terminal state of the original position variable.
        xn: Terminal state of the expanded position variable.
        pn: Terminal state of the original momentum variable.
        yn: Terminal state of the expanded momentum variable.

    """
    # Compute prerequisite quantities for the explicit integrator.
    half_step = step_size / 2.0

    # Apply the explicit integrator to the input.
    qo, xo, po, yo = zo
    qn, xn, pn, yn = phi_a(qo, xo, po, yo, half_step, vector_field)
    if omega > 0:
        qn, xn, pn, yn = phi_b(qn, xn, pn, yn, half_step, vector_field)
        qn, xn, pn, yn = phi_c(qn, xn, pn, yn, step_size, omega)
        qn, xn, pn, yn = phi_b(qn, xn, pn, yn, half_step, vector_field)
    else:
        qn, xn, pn, yn = phi_b(qn, xn, pn, yn, step_size, vector_field)
    qn, xn, pn, yn = phi_a(qn, xn, pn, yn, half_step, vector_field)
    return qn, xn, pn, yn

def constraint(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The holonomic constraint function with which to equip the Lagrange
    multiplier augmented explicit integrator. The constraint states that the
    position variables in the expanded phase space must be equal.

    Args:
        q: Original position variable.
        x: Expanded position variable.

    Returns:
        out: The element-wise difference between the original and expanded
            position variables.

    """
    return q - x

def loss(
        vector_field: Callable,
        zo: Tuple[np.ndarray],
        step_size: float,
        omega: float,
        mu: np.ndarray
) -> np.ndarray:
    """A loss function representing violation of the constraint function with
    respect to the inputs. In practice, one will want to identify the Lagrange
    multipliers that cause the constraint to be satisfied.

    Args:
        vector_field: A function returning the time derivatives of position and
            momentum.
        zo: Tuple containing the position and momentum variables in the expanded
            phase space.
        step_size: Integration step_size.
        omega: Binding strength between the two approximate solutions.

    Returns:
        c: The element-wise difference between the original and expanded
            position variables.
        zn: The output of the explicit integrator.

    """
    qo, xo, po, yo = zo
    zn = coupled_integrate(
        vector_field,
        (qo, xo, po + mu, yo - mu),
        step_size,
        omega
    )
    c = constraint(zn[0], zn[1])
    return c, zn

def step(val, vector_field, zo, step_size, omega):
    """Single step of a Newton iteration to identify constraint-preserving Lagrange
    multipliers.

    """
    # Broyden's method.
    mup, _, J, Jinv, cp, num_iters = val
    Dx = -Jinv@cp
    mun = mup + Dx
    cn, aux = loss(vector_field, zo, step_size, omega, mun)
    Df = cn - cp
    # Update inverse using Sherman–Morrison formula.
    u = (Df - J@Dx) / (Dx@Dx)
    v = Dx
    J += np.outer(u, v)
    div = 1. + v@Jinv@u
    if np.abs(div) > 1e-10:
        Jinv -= (Jinv@np.outer(u, v)@Jinv) / div
    else:
        num_mu = len(mun)
        J = np.eye(num_mu)
        Jinv = np.eye(num_mu)
    num_iters += 1
    return mun, aux, J, Jinv, cn, num_iters

def single_step(
        vector_field: Callable,
        state: CoupledState,
        info: CoupledInfo,
        step_size: float,
        omega: float,
        thresh: float,
        max_iters: int
) -> Tuple:
    """Use the explicit integrator in combination with Lagrange multipliers in
    order to satisfy the constraints that the position and momentum variables
    in the expanded phase space are equal along trajectories.

    Args:
        vector_field: A function returning the time derivatives of position and
            momentum.
        state: An object containing the position and momentum variables of the
            state in phase space.
        info: An object that keeps track of the number of fixed point iterations
            and whether or not integration has been successful.
        step_size: Integration step_size.
        omega: Binding strength between the two approximate solutions.
        thresh: Convergence tolerance for Newton's method to find Lagrange
            multipliers.
        max_iters: Maximum number of iterations.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    qo, po = state.position, state.momentum
    zo = (qo, qo, po, po)
    mu = np.zeros_like(qo)
    # Decide whether or not to initialize the estimate of the Jacobian with the
    # identity matrix or with a finite-difference approximation of the
    # Jacobian.
    num_mu = len(mu)
    J = np.eye(num_mu)
    Jinv = np.eye(num_mu)
    # I think the correct course is to provide the auxiliary data. If the code
    # doesn't complete a single iteration, then the auxiliary data will
    # remain a vector of zeros, which is clearly incorrect.
    cn, aux = loss(vector_field, zo, step_size, omega, mu)
    val = (mu, aux, J, Jinv, cn, 1)
    while cond(val, thresh, max_iters):
        val = step(val, vector_field, zo, step_size, omega)
    mu, (qn, xn, pn, yn), J, Jinv, cn, num_iters = val

    # Compute whether or not integration was successful.
    success = np.max(np.abs(cn)) < thresh
    # Averaging the momentum variables is the projection to the cotangent
    # bundle of the manifold. The averaging of the position variables is not
    # necessary; they are equal under the constraint. However, averaging has a
    # nicer aesthetic when only approximate constraint satisfaction is
    # required.
    qm = 0.5*(qn + xn)
    pm = 0.5*(pn + yn)
    state.position, state.momentum = qm, pm
    info.num_iters += num_iters
    info.success &= success
    return state, info

def coupled_leapfrog(
        state: CoupledState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        vector_field: Callable,
        omega: float,
        thresh: float,
        max_iters: int
) -> Tuple[CoupledState, CoupledInfo]:
    """Implements the coupled explicit integrator where Lagrange multipliers are
    used to satisfy reversibility and volume preservation.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space.
        step_size: Integration step_size.
        num_steps: Number of integration steps.
        log_posterior: The log-density of the posterior from which to sample.
        metric_handler: Function to compute the Riemannian metric, its inverse,
            its matrix square root, and its log-determinant.
        vector_field: A function returning the time derivatives of position and
            momentum.
        omega: Binding strength between the two approximate solutions.
        thresh: Convergence tolerance for Newton's method to find Lagrange
            multipliers.
        max_iters: Maximum number of iterations.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.

    """
    state = copy.copy(state)
    info = CoupledInfo()
    for i in range(num_steps):
        state, info = single_step(
            vector_field,
            state,
            info,
            step_size,
            omega,
            thresh,
            max_iters
        )

    state.update(distr)
    return state, info
