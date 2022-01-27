from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import SoftAbsLeapfrogInfo
from iliad.integrators.stateful import softabs_euler_a, softabs_euler_b
from iliad.integrators.states import SoftAbsLeapfrogState
from .diagnostics import Diagnostics
from .proposal import Proposal, ProposalInfo, error_intercept, momentum_negation


class SoftAbsEulerProposalInfo(ProposalInfo):
    def __init__(self):
        super().__init__()
        self.num_iters_pos = Diagnostics()
        self.num_iters_mom = Diagnostics()

    def asdict(self):
        d = super().asdict()
        d['num. pos.'] = self.num_iters_pos.avg
        d['num. mom.'] = self.num_iters_mom.avg
        return d

class SoftAbsEulerProposal(Proposal):
    def __init__(self,
                 alpha: float,
                 distr: Distribution,
                 thresh: float,
                 max_iters: int
    ):
        super().__init__(distr, SoftAbsEulerProposalInfo(), "smala")
        self.thresh = thresh
        self.max_iters = max_iters
        self.alpha = alpha

    @error_intercept
    def propose(
            self,
            state: SoftAbsLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[SoftAbsLeapfrogState, SoftAbsLeapfrogInfo]:
        if np.random.uniform() < 0.5:
            integrator = softabs_euler_a
            ss = step_size
        else:
            integrator = softabs_euler_b
            ss = -step_size
        state, info = integrator(
            state,
            ss,
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
