import unittest

import numpy as np

from iliad.extras import newton_raphson
from odyssey.logistic import LogisticRegression, sigmoid


class TestNewtonRaphson(unittest.TestCase):
    def test_newton_raphson(self):
        num_obs, num_dims = 100, 5
        x = np.random.normal(size=(num_obs, num_dims))
        b = np.ones((x.shape[-1], ))
        p = sigmoid(x@b)
        y = np.random.binomial(1, p)
        alpha = 0.5
        distr = LogisticRegression(x, y, alpha)
        q = np.zeros_like(b)
        q = newton_raphson(q, distr, 1e-13)
        _, g, _, _ = distr.riemannian_quantities(q)
        self.assertTrue(np.allclose(g, 0.0))
