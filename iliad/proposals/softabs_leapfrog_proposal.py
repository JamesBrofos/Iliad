from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import SoftAbsLeapfrogInfo
from iliad.integrators.stateful import softabs_leapfrog
from iliad.integrators.states import SoftAbsLeapfrogState
from .proposal import Proposal, error_intercept, momentum_negation
from .riemannian_leapfrog_proposal import RiemannianLeapfrogProposalInfo

class SoftAbsLeapfrogProposalInfo(RiemannianLeapfrogProposalInfo):
    pass

class SoftAbsLeapfrogProposal(Proposal):
    """The Riemannian Hamiltonian Monte Carlo algorithm takes advantage of
    second-order geometry in order to adapt proposals to directions of greatest
    variation in the posterior. The SoftAbs metric is a smooth transformation
    of the Hessian so that it becomes positive definite, giving a generic
    metric compatible with Riemannian manifold HMC.

    Parameters:
        alpha: The SoftAbs sharpness parameter.
        distr: The distribution that the transition kernel will sample.
        thresh: Convergence tolerance for fixed point iterations.
        max_iters: Maximum number of fixed point iterations.

    """
    def __init__(self,
                 alpha: float,
                 distr: Distribution,
                 thresh: float,
                 max_iters: int):
        super().__init__(distr, SoftAbsLeapfrogProposalInfo(), "smala")
        self.thresh = thresh
        self.max_iters = max_iters
        self.alpha = alpha

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: SoftAbsLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[SoftAbsLeapfrogState, SoftAbsLeapfrogInfo]:
        state, info = softabs_leapfrog(
            state,
            step_size,
            num_steps,
            self.distr,
            self.thresh,
            self.max_iters
        )
        self.info.num_iters_pos.update(info.num_iters_pos / num_steps)
        self.info.num_iters_mom.update(info.num_iters_mom / num_steps)
        return state, info

    def first_state(self, qt: np.ndarray) -> SoftAbsLeapfrogState:
        p = np.zeros_like(qt)
        state = SoftAbsLeapfrogState(qt, p, self.alpha)
        state.update(self.distr)
        state.sqrtm_metric = np.linalg.cholesky(state.metric)
        state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
        return state
