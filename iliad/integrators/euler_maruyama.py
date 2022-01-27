from typing import Tuple

import numpy as np
import scipy.linalg as spla
import scipy.stats as spst

from iliad.statistics import normal
from iliad.integrators.states import State, LagrangianLeapfrogState, RiemannianLeapfrogState, SoftAbsLeapfrogState

from odyssey.distribution import Distribution


def mean_and_invcov_and_logdet(
        state: RiemannianLeapfrogState,
        eps: float,
        langevin_type: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes the mean, inverse covariance, and log-determinant of the covariance
    matrix for the proposal distribution of the Riemannian Metropolis-adjusted
    Langevin algorithm.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        eps: The discretization step-size.
        langevin_type: Indicates what kind of Langevin stochastic differential
            equation to integrator.

    Returns:
        mean: Mean of the MMALA proposal.
        invcov: Inverse covariance of the MMALA proposal.
        logdet: The log-determinant of the MMALA covariance matrix.
        gamma: Riemannian component of the drift function.

    """
    if langevin_type == 'mala':
        q = state.position
        k = len(q)
        glp = state.grad_log_posterior
        invcov = np.eye(k) / eps**2
        logdet = 2*k*np.log(eps)
        mean = q + 0.5*eps**2*glp
        gamma = 0.0
        return mean, invcov, logdet, gamma
    elif langevin_type == 'mmala':
        q = state.position
        k = len(q)
        glp = state.grad_log_posterior
        G = state.metric
        iG = state.inv_metric
        L = state.sqrtm_metric
        dG = state.jac_metric
        n = 0.5*iG.dot(glp)
        gamma = -0.5*np.sum(np.einsum('ik,jk->ij', iG, np.einsum('mkj,mj->jk', dG, iG)), axis=-1)
        mean = q + eps**2*(n + gamma)
        invcov = G / eps**2
        logdet = -2*np.sum(np.log(np.diag(L))) + 2*k*np.log(eps)
        return mean, invcov, logdet, gamma
    elif langevin_type == 'smala':
        q = state.position
        k = len(q)
        glp = state.grad_log_posterior
        G = state.metric
        iG = state.inv_metric
        L = state.sqrtm_metric
        n = 0.5*iG.dot(glp)
        gamma = 0.0
        mean = q + eps**2*(n + gamma)
        invcov = G / eps**2
        logdet = -2*np.sum(np.log(np.diag(L))) + 2*k*np.log(eps)
        return mean, invcov, logdet, gamma

def euler_maruyama(
        state: State,
        distr: Distribution,
        eps: float,
        langevin_type: str
) -> Tuple[RiemannianLeapfrogState, float]:
    """Applies the Euler-Maruyama integrator to the stochastic differential
    equation on the manifold in order to compute a proposal for the manifold
    MALA Markov chain. Also computes the Metropolis-Hastings correction.

    Args:
        state: An object containing the position and momentum variables of the
            state in phase space, and possibly previously computed log-posterior,
            metrics, and gradients.
        distr: The distribution from which to sample using the manifold MALA.
        eps: The discretization step-size.

    Returns:
        new_state: The proposal state computed by discretizing the Langevin
            manifold stochastic differential equation.
        mh: The Metropolis-Hastings accept-reject decision.

    """
    if langevin_type == 'mala':
        z = eps * np.random.normal(size=len(state.position))
    elif langevin_type in ('mmala', 'smala'):
        z = eps * spla.solve_triangular(
            state.sqrtm_metric.T,
            np.random.normal(size=state.position.shape),
            lower=False,
            overwrite_b=True,
            check_finite=False
        )
    else:
        raise NotImplementedError("Unrecognized Langevin type '{}'.".format(langevin_type))
    mean, invcov, logdet, _ = mean_and_invcov_and_logdet(state, eps, langevin_type)
    qn = mean + z
    ld = normal.logpdf(z, logdet, invcov.dot(z))

    # Handle the different kinds of states that can be employed with the
    # Metropolis-adjusted Langevin algorithm.
    if isinstance(state, RiemannianLeapfrogState):
        new_state = RiemannianLeapfrogState(qn, np.zeros_like(qn))
        new_state.update(distr)
    elif isinstance(state, LagrangianLeapfrogState):
        new_state = LagrangianLeapfrogState(qn, np.zeros_like(qn))
        new_state.update(distr)
    elif isinstance(state, SoftAbsLeapfrogState):
        new_state = SoftAbsLeapfrogState(qn, np.zeros_like(qn), state.alpha)
        new_state.update(distr)
        new_state.sqrtm_metric = np.linalg.cholesky(new_state.metric)
    else:
        raise ValueError()
    new_state.logdet_metric = 2.0*np.sum(np.log(np.diag(new_state.sqrtm_metric)))
    rev_mean, rev_invcov, rev_logdet, _ = mean_and_invcov_and_logdet(new_state, eps, langevin_type)
    delta = state.position - rev_mean
    rev_ld = normal.logpdf(delta, rev_logdet, rev_invcov.dot(delta))
    mh = new_state.log_posterior + rev_ld - state.log_posterior - ld
    return new_state, mh
