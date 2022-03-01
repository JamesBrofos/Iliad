from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import RiemannianLeapfrogInfo
from iliad.integrators.stateful import riemannian_leapfrog
from iliad.integrators.states import RiemannianLeapfrogState
from .diagnostics import Diagnostics
from .proposal import Proposal, ProposalInfo, error_intercept, momentum_negation


class RiemannianLeapfrogProposalInfo(ProposalInfo):
    def __init__(self):
        super().__init__()
        self.num_iters_pos = Diagnostics()
        self.num_iters_mom = Diagnostics()
        self.pos_kl = Diagnostics()

    def asdict(self):
        d = super().asdict()
        d['num. pos.'] = self.num_iters_pos.avg
        d['num. mom.'] = self.num_iters_mom.avg
        d['pos. kl'] = self.pos_kl.avg
        return d

class RiemannianLeapfrogProposal(Proposal):
    """The Riemannian Hamiltonian Monte Carlo algorithm takes advantage of local
    geometric information in order to adapt proposals to directions in which
    the posterior exhibits the greatest variation locally.

    Parameters:
        distr: The distribution that the transition kernel will sample.
        thresh: The convergence threshold for fixed point iterations.
        max_iters: The maximum number of fixed point iterations to attempt.
        newton_momentum: Whether or not to enable Newton iterations for the
            momentum fixed point equation.
        newton_position: Whether or not to enable Newton iterations for the
            position fixed point equation.

    """
    def __init__(self,
                 distr: Distribution,
                 thresh: float,
                 max_iters: int,
                 newton_momentum: bool=False,
                 newton_position: bool=False,
                 newton_stability: bool=False
    ):
        super().__init__(distr, RiemannianLeapfrogProposalInfo(), "mmala")
        self.thresh = thresh
        self.max_iters = max_iters
        self.newton_momentum = newton_momentum
        self.newton_position = newton_position
        self.newton_stability = newton_stability

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: RiemannianLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[RiemannianLeapfrogState, RiemannianLeapfrogInfo]:
        state, info = riemannian_leapfrog(
            state,
            step_size,
            num_steps,
            self.distr,
            self.thresh,
            self.max_iters,
            self.newton_momentum,
            self.newton_position,
            self.newton_stability
        )
        self.info.num_iters_pos.update(info.num_iters_pos / num_steps)
        self.info.num_iters_mom.update(info.num_iters_mom / num_steps)
        self.info.pos_kl.update(info.kl / num_steps)
        return state, info

    def first_state(self, qt: np.ndarray) -> RiemannianLeapfrogState:
        p = np.zeros_like(qt)
        state = RiemannianLeapfrogState(qt, p)
        state.update(self.distr)
        state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
        return state
