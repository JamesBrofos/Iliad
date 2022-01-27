from typing import Callable, Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import GaussLegendreInfo
from iliad.integrators.vectors import gauss_legendre
from iliad.integrators.vectors.vector_fields import riemannian_vector_field, softabs_vector_field
from iliad.integrators.states import GaussLegendreState
from .diagnostics import Diagnostics
from .proposal import Proposal, ProposalInfo, error_intercept, momentum_negation


class GaussLegendreProposalInfo(ProposalInfo):
    def __init__(self):
        super().__init__()
        self.num_iters = Diagnostics()

    def asdict(self):
        d = super().asdict()
        d['num. iters.'] = self.num_iters.avg
        return d

class GaussLegendreProposal(Proposal):
    def __init__(
            self,
            alpha: float,
            distr: Distribution,
            order: int,
            thresh: float,
            max_iters: int
    ):
        super().__init__(distr, GaussLegendreProposalInfo())
        self.alpha = alpha
        if order == 2:
            self.a = np.array([[1/2]])
            self.b = np.array([1.0])
        elif order == 4:
            self.a = np.array([[1/4, 1/4 - np.sqrt(3) / 6],
                               [1/4 + np.sqrt(3) / 6, 1/4]])
            self.b = np.array([1/2, 1/2])
        elif order == 6:
            self.a = np.array([
                [5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],
                [5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],
                [5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]
            ])
            self.b = np.array([5/18, 4/9, 5/18])
        else:
            raise ValueError('Unrecognized `order` argument.')

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
            state: GaussLegendreState,
            step_size: float,
            num_steps: int
    ) -> Tuple[GaussLegendreState, GaussLegendreInfo]:
        state, info = gauss_legendre(
            state,
            step_size,
            num_steps,
            self.distr,
            self.vector_field,
            self.a,
            self.b,
            self.thresh,
            self.max_iters
        )
        self.info.num_iters.update(info.num_iters / num_steps)
        return state, info

    def first_state(self, qt: np.ndarray) -> GaussLegendreState:
        p = np.zeros_like(qt)
        state = GaussLegendreState(qt, p, self.alpha)
        state.update(self.distr)
        return state
