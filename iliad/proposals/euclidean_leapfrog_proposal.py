from typing import Tuple

import numpy as np
import scipy.sparse as spsr

from odyssey.distribution import Distribution

from iliad.integrators.info import EuclideanLeapfrogInfo
from iliad.integrators.stateful import euclidean_leapfrog
from iliad.integrators.states import EuclideanLeapfrogState
from .proposal import Proposal, ProposalInfo, error_intercept, momentum_negation


class EuclideanLeapfrogProposal(Proposal):
    """The leapfrog integrator assumes a fixed metric, which is supplied to the
    constructor of the leapfrog proposal. The required quantities of the metric
    are then computed and cached.

    Parameters:
        distr: The distribution that the transition kernel will sample.

    """
    def __init__(self, distr: Distribution):
        super().__init__(distr, ProposalInfo())
        self.prepare_metric()

    def prepare_metric(self):
        self.metric, self.sqrtm_metric, self.inv_metric = self.distr.euclidean_metric()
        if spsr.issparse(self.sqrtm_metric):
            self.logdet_metric = 2.0*np.sum(np.log(self.sqrtm_metric.diagonal()))
        else:
            self.logdet_metric = 2.0*np.sum(np.log(np.diag(self.sqrtm_metric)))

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: EuclideanLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[EuclideanLeapfrogState, EuclideanLeapfrogInfo]:
        state, info = euclidean_leapfrog(state, step_size, num_steps, self.distr)
        return state, info

    def first_state(self, qt: np.ndarray) -> EuclideanLeapfrogState:
        p = np.zeros_like(qt)
        state = EuclideanLeapfrogState(qt, p)
        state.metric = self.metric
        state.inv_metric = self.inv_metric
        state.sqrtm_metric = self.sqrtm_metric
        state.logdet_metric = self.logdet_metric
        state.update(self.distr)
        return state
