from typing import Callable

import numpy as np

from iliad.integrators.states.lagrangian_leapfrog_state import LagrangianLeapfrogState
from iliad.integrators.fields import riemannian
from iliad.linalg import solve_psd

from odyssey.distribution import Distribution


class RiemannianLeapfrogState(LagrangianLeapfrogState):
    """The Riemannian leapfrog state uses the Fisher information matrix to provide
    a position-dependent Riemannian metric. As such, computing the gradients of
    the Hamiltonian requires higher derivatives of the metric, which vanish in
    the Euclidean case.

    """
    def __copy__(self):
        state = RiemannianLeapfrogState(self.position.copy(), self.momentum.copy())
        state.log_posterior = self.log_posterior.copy()
        state.grad_log_posterior = self.grad_log_posterior.copy()
        state.velocity = self.velocity.copy()
        state.metric = self.metric.copy()
        state.inv_metric = self.inv_metric.copy()
        state.sqrtm_metric = self.sqrtm_metric.copy()
        state.logdet_metric = self.logdet_metric.copy()
        state.jac_metric = self.jac_metric.copy()
        state.grad_logdet_metric = self.grad_logdet_metric.copy()
        state.force = self.force.copy()
        return state

    def update(self, distr: Distribution):
        super().update(distr)
        self.velocity = riemannian.velocity(self.inv_metric, self.momentum)
        self.force = riemannian.force(self.velocity, self.grad_log_posterior, self.jac_metric, self.grad_logdet_metric)
