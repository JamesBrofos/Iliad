import copy
from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import EuclideanLeapfrogInfo
from iliad.integrators.states import EuclideanLeapfrogState


def single_step(
        distr: Distribution,
        state: EuclideanLeapfrogState,
        info: EuclideanLeapfrogInfo,
        step_size: float
) -> Tuple[EuclideanLeapfrogState, EuclideanLeapfrogInfo]:
    """Implements a single step of the leapfrog integrator, which is symmetric,
    symplectic, and second-order accurate for separable Hamiltonian systems.

    Args:
        distr: The distribution that guides the time evolution of the Euclidean
            Hamiltonian trajectory.
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior
            and gradients.
        info: An object that keeps track of the number of fixed point iterations
            and whether or not integration has been successful.
        step_size: Integration step_size.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and its gradient.
        info: An information object with the indicator of successful integration.

    """
    half_step = 0.5*step_size
    state.momentum += half_step * state.force
    state.velocity = state.inv_metric.dot(state.momentum)
    state.position += step_size * state.velocity
    state.update(distr)
    state.momentum += half_step * state.force
    return state, info

def euclidean_leapfrog(
        state: EuclideanLeapfrogState,
        step_size: float,
        num_steps: int,
        distr: Distribution
) -> Tuple[EuclideanLeapfrogState, EuclideanLeapfrogInfo]:
    """Implements the leapfrog integrator for a separable Hamiltonian.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior
            and gradients.
        step_size: Integration step_size.
        num_steps: Number of integration steps.
        distr: The distribution that guides the time evolution of the Euclidean
            Hamiltonian trajectory.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and its gradient.
        info: An information object with the indicator of successful integration.

    """
    state = copy.copy(state)
    info = EuclideanLeapfrogInfo()
    for i in range(num_steps):
        state, info = single_step(distr, state, info, step_size)
    state.velocity = state.inv_metric.dot(state.momentum)
    return state, info
