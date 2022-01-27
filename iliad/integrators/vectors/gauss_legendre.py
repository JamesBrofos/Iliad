import copy
from typing import Callable, Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import GaussLegendreInfo
from iliad.integrators.states import GaussLegendreState
from iliad.integrators.terminal import cond


def gauss_legendre_step(
        val: Tuple,
        zo: Tuple[np.ndarray],
        a: np.ndarray,
        step_size: float,
        vector_field: Callable
):
    qo, po = zo
    qncand, pncand, _, _, _, num_iters = val
    m = len(qo)
    num_stages = len(a)
    dq = np.zeros([num_stages, m])
    dp = np.zeros([num_stages, m])
    for i in range(num_stages):
        dq[i], dp[i] = vector_field(qncand[i], pncand[i])
    qn = qo + step_size * np.dot(a, dq)
    pn = po + step_size * np.dot(a, dp)
    delta = np.concatenate([np.ravel(qn - qncand), np.ravel(pn - pncand)])
    num_iters += 1
    return qn, pn, dq, dp, delta, num_iters

def single_step(
        vector_field: Callable,
        state: GaussLegendreState,
        info: GaussLegendreInfo,
        step_size: float,
        a: np.ndarray,
        b: np.ndarray,
        thresh: float,
        max_iters: int
) -> Tuple[GaussLegendreState, GaussLegendreInfo]:
    qo, po = state.position, state.momentum
    m = len(po)
    num_stages = len(b)
    delta = np.ones(2*num_stages*m) * np.inf
    qnc = np.array([qo for _ in range(num_stages)])
    pnc = np.array([po for _ in range(num_stages)])
    dq = np.zeros_like(qnc)
    dp = np.zeros_like(pnc)
    val = (
        qnc,
        pnc,
        dq,
        dp,
        delta,
        0
    )
    while cond(val, thresh, max_iters):
        val = gauss_legendre_step(val, (qo, po), a, step_size, vector_field)
    qnc, pnc, dq, dp, delta, num_iters = val
    qn = qo + step_size*np.dot(b, dq)
    pn = po + step_size*np.dot(b, dp)
    success = np.max(np.abs(delta)) < thresh
    state.position = qn
    state.momentum = pn
    info.num_iters += num_iters
    info.success &= success
    return state, info

def gauss_legendre(
        state: GaussLegendreState,
        step_size: float,
        num_steps: int,
        distr: Distribution,
        vector_field: Callable,
        a: np.ndarray,
        b: np.ndarray,
        thresh: float,
        max_iters: int
):
    state = copy.copy(state)
    info = GaussLegendreInfo()
    for i in range(num_steps):
        state, info = single_step(vector_field, state, info, step_size, a, b, thresh, max_iters)

    state.update(distr)
    return state, info
