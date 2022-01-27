import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.fields import softabs
from iliad.linalg import solve_psd, sqrtm

from .state import State


class CoupledState(State):
    def __init__(self, position: np.ndarray, momentum: np.ndarray, alpha: float):
        super().__init__(position, momentum)
        self.alpha = alpha

    def __copy__(self):
        state = CoupledState(self.position.copy(), self.momentum.copy(), self.alpha)
        return state

    def update(self, distr: Distribution):
        if self.alpha > 0.0:
            H = distr.hessian(self.position)
            _, U, lt, _, metric, inv_metric = softabs.decomposition(H, self.alpha)
            sqrtm_metric = sqrtm(lt, U)
            logdet_metric = np.sum(np.log(lt))
        else:
            metric = distr.riemannian_metric(self.position)
            inv_metric, sqrtm_metric = solve_psd(metric)
            logdet_metric = 2.0*np.sum(np.log(np.diag(sqrtm_metric)))
        self.log_posterior = distr.log_density(self.position)
        self.metric = metric
        self.inv_metric = inv_metric
        self.sqrtm_metric = sqrtm_metric
        self.logdet_metric = logdet_metric
        self.velocity = self.inv_metric@self.momentum
