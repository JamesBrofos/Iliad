import unittest

import numpy as np

from iliad.integrators.states import RiemannianLeapfrogState
from iliad.integrators.euler_maruyama import mean_and_invcov_and_logdet
from iliad.linalg import solve_psd
from odyssey.banana import Banana, generate_data


class TestEulerMaruyama(unittest.TestCase):
    def test_euler_maruyama(self):
        t = 0.5
        sigma_theta = 2.
        sigma_y = 2.
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)
        distr = Banana(y, sigma_y, sigma_theta)

        eps = 1e-1
        q = np.random.normal(size=(2, ))
        p = np.zeros_like(q)
        state = RiemannianLeapfrogState(q, p)
        state.update(distr)

        def f(q):
            lp, glp, G, dG = distr.riemannian_quantities(q)
            iG = solve_psd(G)[0]
            return iG

        def g(q):
            delta = 1e-4
            Id = np.eye(len(q))
            x = np.zeros_like(q)
            for j in range(len(q)):
                x += (f(q + 0.5*delta*Id[j])[:, j] - f(q - 0.5*delta*Id[j])[:, j]) / delta
            return 0.5*x

        _, _, logdet, gamma = mean_and_invcov_and_logdet(state, eps, 'mmala')
        self.assertTrue(np.allclose(gamma, g(q)))

        lp, glp, G, dG = distr.riemannian_quantities(q)
        iG = solve_psd(G)[0]
        self.assertTrue(np.allclose(np.linalg.slogdet(eps**2*iG)[1], logdet))
