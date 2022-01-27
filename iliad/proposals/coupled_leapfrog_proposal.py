from typing import Callable, Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import CoupledInfo
from iliad.integrators.vectors import coupled_leapfrog
from iliad.integrators.vectors.vector_fields import riemannian_vector_field, softabs_vector_field
from iliad.integrators.states import CoupledState
from .diagnostics import Diagnostics
from .proposal import Proposal, ProposalInfo, error_intercept, momentum_negation


class CoupledProposalInfo(ProposalInfo):
    def __init__(self):
        super().__init__()
        self.num_iters = Diagnostics()

    def asdict(self):
        d = super().asdict()
        d['num. iters.'] = self.num_iters.avg
        return d


class CoupledLeapfrogProposal(Proposal):
    def __init__(
            self,
            alpha: float,
            distr: Distribution,
            omega: float,
            thresh: float,
            max_iters: int
    ):
        super().__init__(distr, CoupledProposalInfo())
        self.alpha = alpha
        self.omega = omega
        self.thresh = thresh
        self.max_iters = max_iters
        if self.alpha > 0.0:
            self.vector_field = softabs_vector_field(distr, alpha)
        else:
            self.vector_field = riemannian_vector_field(distr)

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: CoupledState,
            step_size: float,
            num_steps: int
    ) -> Tuple[CoupledState, CoupledProposalInfo]:
        state, info = coupled_leapfrog(
            state,
            step_size,
            num_steps,
            self.distr,
            self.vector_field,
            self.omega,
            self.thresh,
            self.max_iters
        )
        self.info.num_iters.update(info.num_iters / num_steps)
        return state, info

    def first_state(self, qt: np.ndarray) -> CoupledState:
        p = np.zeros_like(qt)
        state = CoupledState(qt, p, self.alpha)
        state.update(self.distr)
        return state
