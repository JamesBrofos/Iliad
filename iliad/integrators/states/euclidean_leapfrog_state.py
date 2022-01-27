from typing import Callable

import numpy as np

from odyssey.distribution import Distribution

from .leapfrog_state import LeapfrogState


class EuclideanLeapfrogState(LeapfrogState):
    """The Euclidean leapfrog state implements the state object for Hamiltonian
    Monte Carlo with a constant metric. The Euclidean state needs to be updated
    if either the log-posterior or the gradient of the log-posterior are not
    available.

    """
    def __copy__(self):
        state = EuclideanLeapfrogState(self.position.copy(), self.momentum.copy())
        state.log_posterior = self.log_posterior.copy()
        state.grad_log_posterior = self.grad_log_posterior.copy()
        state.velocity = self.velocity.copy()
        state.force = self.force.copy()
        state.metric = self.metric.copy()
        state.inv_metric = self.inv_metric.copy()
        state.sqrtm_metric = self.sqrtm_metric.copy()
        state.logdet_metric = self.logdet_metric.copy()
        return state

    def update(self, distr: Distribution):
        log_posterior, grad_log_posterior = distr.euclidean_quantities(self.position)
        self.log_posterior = log_posterior
        self.grad_log_posterior = grad_log_posterior
        self.velocity = self.inv_metric.dot(self.momentum)
        self.force = self.grad_log_posterior

