import numpy as np


class RuppertAveraging:
    """Ruppert averaging is a method of stochastic approximation wherein we seek to
    find the solution of a univariate equation. However, the equation is
    measured with noise, so we seek a value that solves the equation in
    expectation. Ruppert averaging is asymptotically optimal under certain
    regularity conditions.

    Parameters:
        x0: Initial guess for the solution of the equation.
        omega: Learning rate decay parameter.
        maxval: Maximum value of the solution.
        minval: Minimum value of the solution.

    """
    def __init__(
            self,
            x0: float,
            omega: float,
            maxval: float=np.inf,
            minval: float=-np.inf
    ):
        self.x = x0
        self.xb = x0
        self.omega = omega
        self.t = 0
        self.maxval, self.minval = maxval, minval

    @property
    def eta(self):
        return self.t**-self.omega

    def update(self, new_term: float):
        self.t += 1
        self.x -= self.eta*new_term
        self.x = np.clip(self.x, self.minval, self.maxval)
        self.xb = self.t / (self.t + 1) * self.xb + self.x / (self.t + 1)
