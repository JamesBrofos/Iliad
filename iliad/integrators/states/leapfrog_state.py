import abc
from typing import Callable, Optional

import numpy as np

from .state import State


class LeapfrogState(State):
    """The leapfrog state caches the gradient of the log-posterior between steps so
    that it need not be recomputed. The leapfrog integrator also introduces
    variables representing the velocity and force of the system, which are the
    time derivatives of position and momentum, respectively.

    Args:
        position: The position variable.
        momentum: The momentum variable.

    """
    def __init__(self,
                 position: np.ndarray,
                 momentum: np.ndarray):
        super().__init__(position, momentum)
        self._grad_log_posterior: np.ndarray
        self._velocity: np.ndarray
        self._force: np.ndarray

    @abc.abstractmethod
    def update(self, auxiliaries: Callable):
        raise NotImplementedError()

    @property
    def grad_log_posterior(self):
        return self._grad_log_posterior

    @grad_log_posterior.setter
    def grad_log_posterior(self, value):
        self._grad_log_posterior = value

    @grad_log_posterior.deleter
    def grad_log_posterior(self):
        del self._grad_log_posterior

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = value

    @velocity.deleter
    def velocity(self):
        del self._velocity

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, value):
        self._force = value

    @force.deleter
    def force(self):
        del self._force
