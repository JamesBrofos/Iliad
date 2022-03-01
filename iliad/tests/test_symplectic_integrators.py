import unittest

import numpy as np

from odyssey.banana import Banana, generate_data
from odyssey.neal_funnel import NealFunnel

from iliad.integrators.states import LobattoState, RiemannianLeapfrogState, SoftAbsLeapfrogState, GaussLegendreState
from iliad.integrators.stateful import riemannian_euler_a, riemannian_euler_b, riemannian_leapfrog, softabs_euler_a, softabs_euler_b, softabs_leapfrog
from iliad.integrators.vectors import lobatto_leapfrog, gauss_legendre
from iliad.integrators.vectors.vector_fields import riemannian_vector_field


class TestSymplecticIntegrators(unittest.TestCase):
    def test_riemannian_euler(self):
        num_dims = 2
        t = 0.5
        sigma_theta = 2.0
        sigma_y = 2.0
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)
        distr = Banana(y, sigma_y, sigma_theta)

        q = np.array([t, np.sqrt(1-t**2)]) + 0.1*np.random.normal(size=(2, ))
        G = distr.riemannian_metric(q)
        L = np.linalg.cholesky(G)
        p = L@np.random.normal(size=q.shape)
        state = RiemannianLeapfrogState(q, p)
        state.update(distr)
        state.logdet_metric = 2*np.sum(np.log(np.diag(L)))

        step_size = 0.02
        num_steps = 5
        thresh = 1e-13
        max_iters = 10000

        state_a, _ = riemannian_euler_a(state, step_size, num_steps, distr, thresh, max_iters)
        state_b, _ = riemannian_euler_b(state_a, -step_size, num_steps, distr, thresh, max_iters)
        self.assertTrue(np.allclose(state_b.position, state.position))
        self.assertTrue(np.allclose(state_b.momentum, state.momentum))

        num_steps = 1
        state_a, _ = riemannian_euler_b(state, 0.5*step_size, num_steps, distr, thresh, max_iters)
        state_c, _ = riemannian_euler_a(state_a, 0.5*step_size, num_steps, distr, thresh, max_iters)
        state_l, _ = riemannian_leapfrog(state, step_size, num_steps, distr, thresh, max_iters, False, False, False)
        self.assertTrue(np.allclose(state_c.position, state_l.position))
        self.assertTrue(np.allclose(state_c.momentum, state_l.momentum))

        for step_size in np.logspace(-10, -1, 10):
            state_a, _ = riemannian_euler_a(state, step_size, num_steps, distr, thresh, max_iters)
            state_l, _ = riemannian_leapfrog(state, step_size, num_steps, distr, thresh, max_iters, False, False, False)
            d = np.linalg.norm(state_a.position - state_l.position)
            print('{:.3e}'.format(d))

    def test_softabs_euler(self):
        num_dims = int(np.ceil(10*np.random.uniform()))
        num_dims = 2
        distr = NealFunnel(num_dims)
        q = np.hstack(distr.sample())
        p = np.random.normal(size=q.shape)
        state = SoftAbsLeapfrogState(q, p, 1e4)
        state.update(distr)
        L = np.linalg.cholesky(state.metric)
        state.sqrtm_metric = L
        state.logdet_metric = 2.0*np.sum(np.log(np.diag(L)))
        state.momentum = L@np.random.normal(size=q.shape)
        state.velocity = state.inv_metric.dot(state.momentum)

        step_size = 0.1
        num_steps = 1
        thresh = 1e-13
        max_iters = 10000

        state_a, _ = softabs_euler_a(state, step_size, num_steps, distr, thresh, max_iters)
        state_b, _ = softabs_euler_b(state_a, -step_size, num_steps, distr, thresh, max_iters)
        self.assertTrue(np.allclose(state_b.position, state.position))
        self.assertTrue(np.allclose(state_b.momentum, state.momentum))

        num_steps = 1
        state_a, _ = softabs_euler_b(state, 0.5*step_size, num_steps, distr, thresh, max_iters)
        state_c, _ = softabs_euler_a(state_a, 0.5*step_size, num_steps, distr, thresh, max_iters)
        state_l, _ = softabs_leapfrog(state, step_size, num_steps, distr, thresh, max_iters)
        self.assertTrue(np.allclose(state_c.position, state_l.position))
        self.assertTrue(np.allclose(state_c.momentum, state_l.momentum))

        for step_size in np.logspace(-10, -1, 10):
            state_a, _ = softabs_euler_a(state, step_size, num_steps, distr, thresh, max_iters)
            state_l, _ = softabs_leapfrog(state, step_size, num_steps, distr, thresh, max_iters)
            d = np.linalg.norm(state_a.position - state_l.position)
            print('{:.3e}'.format(d))

    def test_lobatto_integrator(self):
        num_dims = 2
        t = 0.5
        sigma_theta = 2.0
        sigma_y = 2.0
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)
        distr = Banana(y, sigma_y, sigma_theta)

        q = np.array([t, np.sqrt(1-t**2)]) + 0.1*np.random.normal(size=(2, ))
        G = distr.riemannian_metric(q)
        L = np.linalg.cholesky(G)
        p = L@np.random.normal(size=q.shape)
        state_lf = RiemannianLeapfrogState(q, p)
        state_lf.update(distr)
        state_lf.logdet_metric = 2*np.sum(np.log(np.diag(L)))
        state_lo = LobattoState(q, p, 0.0)
        state_lo.update(distr)

        num_steps = 1
        step_size = 0.02
        thresh = 1e-13
        max_iters = 10000

        vector_field = riemannian_vector_field(distr)
        for step_size in np.logspace(-10, -1, 10):
            state_o, _ = lobatto_leapfrog(state_lo, step_size, num_steps, distr, vector_field, thresh, max_iters)
            state_l, _ = riemannian_leapfrog(state_lf, step_size, num_steps, distr, thresh, max_iters, False, False, False)
            d = np.linalg.norm(state_o.position - state_l.position)
            print('{:.3e}'.format(d))

    def test_gauss_legendre_integrator(self):
        num_dims = 2
        t = 0.5
        sigma_theta = 2.0
        sigma_y = 2.0
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)
        distr = Banana(y, sigma_y, sigma_theta)

        q = np.array([t, np.sqrt(1-t**2)]) + 0.1*np.random.normal(size=(2, ))
        G = distr.riemannian_metric(q)
        L = np.linalg.cholesky(G)
        p = L@np.random.normal(size=q.shape)
        state_lf = GaussLegendreState(q, p, 0.0)
        state_lf.update(distr)
        state_lf.logdet_metric = 2*np.sum(np.log(np.diag(L)))
        state_lo = LobattoState(q, p, 0.0)
        state_lo.update(distr)
        state_lo.logdet_metric = 2*np.sum(np.log(np.diag(L)))

        num_steps = 1
        thresh = 1e-13
        max_iters = 10000

        if np.random.uniform() < 0.5:
            a = np.array([[1/4, 1/4 - np.sqrt(3) / 6],
                          [1/4 + np.sqrt(3) / 6, 1/4]])
            b = np.array([1/2, 1/2])
        else:
            a = np.array([
                [5/36, 2/9 - np.sqrt(15)/15, 5/36 - np.sqrt(15)/30],
                [5/36 + np.sqrt(15)/24, 2/9, 5/36 - np.sqrt(15)/24],
                [5/36 + np.sqrt(15)/30, 2/9 + np.sqrt(15)/15, 5/36]
            ])
            b = np.array([5/18, 4/9, 5/18])

        vector_field = riemannian_vector_field(distr)
        for step_size in np.logspace(-10, -1, 10):
            state_o, _ = lobatto_leapfrog(state_lo, step_size, num_steps, distr, vector_field, thresh, max_iters)
            state_l, _ = gauss_legendre(state_lf, step_size, num_steps, distr, vector_field, a, b, thresh, max_iters)
            d = np.linalg.norm(state_o.position - state_l.position)
            print('{:.3e}'.format(d))

        a = np.array([[1/2]])
        b = np.array([1.0])
        step_size = 0.1
        state_a, _ = gauss_legendre(state_lf, step_size, num_steps, distr, vector_field, a, b, thresh, max_iters)
        state_b, _ = gauss_legendre(state_a, -step_size, num_steps, distr, vector_field, a, b, thresh, max_iters)
        self.assertTrue(np.allclose(state_b.position, state_lf.position))
        self.assertTrue(np.allclose(state_b.momentum, state_lf.momentum))
