from typing import Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.info import LagrangianLeapfrogInfo
from iliad.integrators.stateful import lagrangian_leapfrog
from iliad.integrators.states import LagrangianLeapfrogState
from .proposal import Proposal, ProposalInfo, error_intercept, momentum_negation


class LagrangianLeapfrogProposal(Proposal):
    """By converting from Hamiltonian mechanics to Lagrangian mechanics, one can
    deduce a fully explicit numerical integrator that can nonetheless take
    advantage of geometric concepts.

    Parameters:
        distr: The distribution that the transition kernel will sample.

    """
    def __init__(self, distr: Distribution, inverted: bool=False):
        super().__init__(distr, ProposalInfo(), "mmala")
        self.inverted = inverted

    @error_intercept
    @momentum_negation
    def propose(
            self,
            state: LagrangianLeapfrogState,
            step_size: float,
            num_steps: int
    ) -> Tuple[LagrangianLeapfrogState, LagrangianLeapfrogInfo]:
        state, info = lagrangian_leapfrog(
            state,
            step_size,
            num_steps,
            self.distr,
            self.inverted
        )
        return state, info

    def first_state(self, qt: np.ndarray) -> LagrangianLeapfrogState:
        p = np.zeros_like(qt)
        state = LagrangianLeapfrogState(qt, p)
        state.update(self.distr)
        state.logdet_metric = 2.0*np.sum(np.log(np.diag(state.sqrtm_metric)))
        return state
