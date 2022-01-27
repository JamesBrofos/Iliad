import unittest

import numpy as np

from iliad.hamiltonian import hamiltonian
from iliad.integrators.fields import riemannian, softabs
from iliad.integrators.states import SoftAbsLeapfrogState
from iliad.linalg import solve_psd

from odyssey.banana import Banana, generate_data
from odyssey.logistic import LogisticRegression, sigmoid
from odyssey.neal_funnel import NealFunnel


class TestHamiltonian(unittest.TestCase):
    def test_riemannian_hamiltonian(self):
        if np.random.uniform() < 0.5:
            num_obs, num_dims = 100, 5
            x = np.random.normal(size=(num_obs, num_dims))
            b = np.ones((x.shape[-1], ))
            p = sigmoid(x@b)
            y = np.random.binomial(1, p)
            alpha = 0.5
            distr = LogisticRegression(x, y, alpha)
        else:
            num_dims = 2
            t = 0.5
            sigma_theta = 2.0
            sigma_y = 2.0
            theta, y = generate_data(t, sigma_y, sigma_theta, 100)
            distr = Banana(y, sigma_y, sigma_theta)

        def H(q, p):
            lp, glp, G, dG = distr.riemannian_quantities(q)
            iG, L = solve_psd(G)
            logdet = 2*np.sum(np.log(np.diag(L)))
            v = hamiltonian(p, lp, logdet, iG@p)
            return v

        q = np.random.normal(size=num_dims)
        p = np.random.normal(size=num_dims)
        u = np.random.normal(size=num_dims)
        delta = 1e-5
        lp, glp, G, dG = distr.riemannian_quantities(q)
        iG, L = solve_psd(G)
        gld = riemannian.grad_logdet(iG, dG, num_dims)
        dd = riemannian.velocity(iG, p)@u
        fd = (H(q, p + 0.5*delta*u) - H(q, p - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, dd))

        dd = -riemannian.force(iG@p, glp, dG, gld)@u
        fd = (H(q + 0.5*delta*u, p) - H(q - 0.5*delta*u, p)) / delta
        self.assertTrue(np.allclose(fd, dd))

    def test_softabs_hamiltonian(self):
        num_dims = int(np.ceil(10*np.random.uniform()))
        distr = NealFunnel(num_dims)
        x, v = distr.sample()
        q = np.hstack((x, v))
        alpha = 1e1
        state = SoftAbsLeapfrogState(q, np.zeros_like(q), alpha)
        state.update(distr)
        p = np.random.normal(size=q.shape)

        def H(q, p):
            ld, _, H, _ = distr.softabs_quantities(q)
            l, U, lt, inv_lt, metric, inv_metric= softabs.decomposition(H, alpha)
            logdet = np.sum(np.log(lt))
            ham = hamiltonian(p, ld, logdet, inv_metric@p)
            return ham

        delta = 1e-5
        u = np.random.normal(size=q.shape)
        fd = (H(q, p + 0.5*delta*u) - H(q, p - 0.5*delta*u)) / delta
        self.assertTrue(np.allclose(fd, riemannian.velocity(state.inv_metric, p)@u))
        fd = (H(q + 0.5*delta*u, p) - H(q - 0.5*delta*u, p)) / delta
        dHdq = -softabs.force(p,
                              state.grad_log_posterior,
                              state.jac_hessian,
                              state.hessian_eigenvals,
                              state.softabs_eigenvals,
                              state.softabs_inv_eigenvals,
                              state.hessian_eigenvecs,
                              alpha)@u
        self.assertTrue(np.allclose(fd, dHdq))
