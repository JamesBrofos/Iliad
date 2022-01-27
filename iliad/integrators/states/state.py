import abc
from typing import Optional

import numpy as np


class State(abc.ABC):
    """In Hamiltonian Monte Carlo a state consists, at minimum, of a position and
    momentum, which is a location in the phase space of the dynamical system.
    The position variable also corresponds to a particular value of the
    log-posterior which is evaluated at the given position. In addition, when
    computing the Hamiltonian, we will require the log-determinant and the
    inverse of the covariance matrix of the momentum. When sampling the
    momentum, we will require a square root of the covariance matrix. These
    quantities may depend on the position, or they may not, so they are not
    cleared in the abstract base class implementation.

    Args:
        position: The position variable.
        momentum: The momentum variable.

    """
    def __init__(self,
                 position: np.ndarray,
                 momentum: np.ndarray):
        self.position: np.ndarray = position
        self.momentum: np.ndarray = momentum
        self.velocity: np.ndarray

        self._log_posterior: float
        self._metric: np.ndarray
        self._inv_metric: np.ndarray
        self._sqrtm_metric: np.ndarray
        self._logdet_metric: float

    @property
    def log_posterior(self):
        return self._log_posterior

    @log_posterior.setter
    def log_posterior(self, value):
        self._log_posterior = value

    @log_posterior.deleter
    def log_posterior(self):
        del self._log_posterior

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value

    @metric.deleter
    def metric(self):
        del self._metric

    @property
    def inv_metric(self):
        return self._inv_metric

    @inv_metric.setter
    def inv_metric(self, value):
        self._inv_metric = value

    @inv_metric.deleter
    def inv_metric(self):
        del self._inv_metric

    @property
    def sqrtm_metric(self):
        return self._sqrtm_metric

    @sqrtm_metric.setter
    def sqrtm_metric(self, value):
        self._sqrtm_metric = value

    @sqrtm_metric.deleter
    def sqrtm_metric(self):
        del self._sqrtm_metric

    @property
    def logdet_metric(self):
        return self._logdet_metric

    @logdet_metric.setter
    def logdet_metric(self, value):
        self._logdet_metric = value

    @logdet_metric.deleter
    def logdet_metric(self):
        del self._logdet_metric
