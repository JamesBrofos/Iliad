import unittest

import numpy as np

from odyssey.banana import Banana, generate_data

from iliad.integrators.vectors.riemannian_leapfrog import riemannian_velocity_and_force
from iliad.integrators.states import RiemannianLeapfrogState
from iliad.integrators.stateful import riemannian_leapfrog
from iliad.integrators.vectors import riemannian_leapfrog as vector_field_riemannian_leapfrog

class TestRiemannianLeapfrog(unittest.TestCase):
    def test_integrator(self):
        num_dims = 2
        t = 0.5
        sigma_theta = 2.0
        sigma_y = 2.0
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)
        distr = Banana(y, sigma_y, sigma_theta)

        q = np.array([t, np.sqrt(1-t**2)])
        G = distr.riemannian_metric(q)
        L = np.linalg.cholesky(G)
        p = L@np.random.normal(size=q.shape)
        state = RiemannianLeapfrogState(q, p)
        state.update(distr)
        state.logdet_metric = 2*np.sum(np.log(np.diag(L)))

        step_size = 0.02
        num_steps = 1
        # This check may fail for low precision because the cached version of
        # the generalized leapfrog integrator uses a predictor step.
        thresh = 1e-13
        max_iters = 10000

        glf_state_a, _ = riemannian_leapfrog(state, step_size, num_steps, distr, thresh, max_iters, False, False, False)
        velocity_vector, force_vector = riemannian_velocity_and_force(distr)
        glf_state_b, _ = vector_field_riemannian_leapfrog(state, step_size, num_steps, distr, velocity_vector, force_vector, thresh, max_iters)
        self.assertTrue(np.allclose(glf_state_a.position, glf_state_b.position))
        self.assertTrue(np.allclose(glf_state_a.momentum, glf_state_b.momentum))
        self.assertTrue(np.allclose(glf_state_a.log_posterior, glf_state_b.log_posterior))

        glf_state_c, _ = riemannian_leapfrog(state, step_size, num_steps, distr, thresh, max_iters, True, True, True)
        self.assertTrue(np.allclose(glf_state_a.position, glf_state_c.position))
        self.assertTrue(np.allclose(glf_state_a.momentum, glf_state_c.momentum))
