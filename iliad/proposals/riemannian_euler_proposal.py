from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import RiemannianLeapfrogInfo
from iliad.integrators.stateful import riemannian_euler_a, riemannian_euler_b
from iliad.integrators.states import RiemannianLeapfrogState
from .diagnostics import Diagnostics
from .proposal import Proposal, ProposalInfo, error_intercept, momentum_negation


class RiemannianEulerProposalInfo(ProposalInfo):
    def __init__(self):
        super().__init__()
        self.num_iters_pos = Diagnostics()
        self.num_iters_mom = Diagnostics()

    def asdict(self):
        d = super().asdict()
        d['num. pos.'] = self.num_iters_pos.avg
        d['num. mom.'] = self.num_iters_mom.avg
        return d

class RiemannianEulerProposal(Proposal):
    def __init__(self,
                 distr: Distribution,
                 thresh: float,
                 max_iters: int
    ):
        super().__init__(distr, RiemannianEulerProposalInfo(), "mmala")
        self.thresh = thresh
        self.max_iters = max_iters

    @error_intercept
    def propose(
            self,
            state: RiemannianLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
        if np.random.uniform() < 0.5:
            integrator = riemannian_euler_a
            ss = step_size
        else:
            integrator = riemannian_euler_b
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

    def first_state(self, qt: np.ndarray) -> RiemannianLeapfrogState:
        p = np.zeros_like(qt)
        state = RiemannianLeapfrogState(qt, p)
        state.update(self.distr)
        state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
        return state
