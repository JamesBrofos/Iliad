from typing import Callable

import numpy as np

from iliad.integrators.states.leapfrog_state import LeapfrogState
from iliad.integrators.fields import riemannian
from iliad.linalg import solve_psd

from odyssey.distribution import Distribution


class LagrangianLeapfrogState(LeapfrogState):
    """The Riemannian leapfrog state uses the Fisher information matrix to provide
    a position-dependent Riemannian metric. As such, computing the gradients of
    the Hamiltonian requires higher derivatives of the metric, which vanish in
    the Euclidean case.

    """
    def __init__(self,
                 position: np.ndarray,
                 momentum: np.ndarray):
        super().__init__(position, momentum)
        self._jac_metric: np.ndarray
        self._grad_logdet_metric: np.ndarray

    def __copy__(self):
        state = LagrangianLeapfrogState(self.position.copy(), self.momentum.copy())
        state.log_posterior = self.log_posterior.copy()
        state.grad_log_posterior = self.grad_log_posterior.copy()
        state.velocity = self.velocity.copy()
        state.metric = self.metric.copy()
        state.inv_metric = self.inv_metric.copy()
        state.sqrtm_metric = self.sqrtm_metric.copy()
        state.logdet_metric = self.logdet_metric.copy()
        state.jac_metric = self.jac_metric.copy()
        state.grad_logdet_metric = self.grad_logdet_metric.copy()
        return state

    @property
    def jac_metric(self):
        return self._jac_metric

    @jac_metric.setter
    def jac_metric(self, value):
        self._jac_metric = value

    @jac_metric.deleter
    def jac_metric(self):
        del self._jac_metric

    @property
    def grad_logdet_metric(self):
        return self._grad_logdet_metric

    @grad_logdet_metric.setter
    def grad_logdet_metric(self, value):
        self._grad_logdet_metric = value

    @grad_logdet_metric.deleter
    def grad_logdet_metric(self):
        del self._grad_logdet_metric

    def update(self, distr: Distribution):
        num_dims = len(self.position)
        log_posterior, grad_log_posterior, metric, jac_metric = distr.riemannian_quantities(self.position)
        inv_metric, sqrtm_metric = solve_psd(metric)
        grad_logdet_metric = riemannian.grad_logdet(inv_metric, jac_metric, num_dims)
        self.log_posterior = log_posterior
        self.grad_log_posterior = grad_log_posterior
        self.metric = metric
        self.sqrtm_metric = sqrtm_metric
        self.inv_metric = inv_metric
        self.jac_metric = jac_metric
        self.grad_logdet_metric = grad_logdet_metric
