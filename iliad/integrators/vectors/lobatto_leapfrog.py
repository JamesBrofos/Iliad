import copy
from typing import Callable, Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import LobattoInfo
from iliad.integrators.states import LobattoState
from iliad.integrators.terminal import cond


def lobatto_step(
        val: Tuple,
        zo: Tuple[np.ndarray],
        step_size: float,
        vector_field: Callable
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    tq = np.array([[     0.0,     0.0,       0.0],
                   [5.0 / 24, 1.0 / 3, -1.0 / 24],
                   [ 1.0 / 6, 2.0 / 3,   1.0 / 6]])
    tp = np.array([[1.0 / 6, -1.0 / 6, 0.0],
                   [1.0 / 6,  1.0 / 3, 0.0],
                   [1.0 / 6,  5.0 / 6, 0.0]])
    qo, po = zo
    qncand, pncand, _, _, _, num_iters = val
    aq, ap = vector_field(qncand[0], pncand[0])
    bq, bp = vector_field(qncand[1], pncand[1])
    cq, cp = vector_field(qncand[2], pncand[2])
    f = np.array([aq, bq, cq])
    g = np.array([ap, bp, cp])
    qn = qo + step_size * np.dot(tq, f)
    pn = po + step_size * np.dot(tp, g)
    delta = np.concatenate([np.ravel(qn - qncand), np.ravel(pn - pncand)])
    num_iters += 1
    return qn, pn, f, g, delta, num_iters

def single_step(
        vector_field: Callable,
        state: LobattoState,
        info: LobattoInfo,
        step_size: float,
        thresh: float,
        max_iters: int
) -> Tuple[LobattoState, LobattoInfo]:
    # Resolve the fixed point iteration involving all of the intermediate
    # positions, momenta, velocities, and forces.
    qo, po = state.position, state.momentum
    m = len(po)
    delta = np.ones(2*3*m) * np.inf
    f = np.zeros([3, m])
    g = np.zeros([3, m])
    # Predictor step using the current velocity and force followed by an
    # iterative corrector step.
    vel, force = vector_field(state.position, state.momentum)
    qa = qo + step_size/6 * vel
    qb = qa + 2*step_size/3 * vel
    qc = qb + step_size/6 * vel
    pa = po + step_size/6 * force
    pb = pa + 2*step_size/3 * force
    pc = pb + step_size/6 * force
    val = (
        np.array([qa, qb, qc]),
        np.array([pa, pb, pc]),
        f,
        g,
        delta,
        0
    )
    while cond(val, thresh, max_iters):
        val = lobatto_step(val, (qo, po), step_size, vector_field)

    _, _, f, g, delta, num_iters = val
    tq = np.array([1.0 / 6, 2.0 / 3, 1.0 / 6])
    tp = np.array([1.0 / 6, 2.0 / 3, 1.0 / 6])
    qn = qo + step_size*np.dot(tq, f)
    pn = po + step_size*np.dot(tp, g)
    success = np.max(np.abs(delta)) < thresh
    state.position = qn
    state.momentum = pn
    info.num_iters += num_iters
    info.success &= success
    return state, info

def lobatto_leapfrog(
        state: LobattoState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        vector_field: Callable,
        thresh: float,
        max_iters: int
):
    state = copy.copy(state)
    info = LobattoInfo()
    for i in range(num_steps):
        state, info = single_step(vector_field, state, info, step_size, thresh, max_iters)

    state.update(distr)
    return state, info
