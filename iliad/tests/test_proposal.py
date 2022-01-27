import unittest

import numpy as np

from iliad.integrators.info import EuclideanLeapfrogInfo
from iliad.integrators.stateful.lagrangian_leapfrog import velocity_step, velocity_step_alt
from iliad.proposals import CoupledLeapfrogProposal, EuclideanLeapfrogProposal, LagrangianLeapfrogProposal, LobattoProposal, RiemannianLeapfrogProposal, SoftAbsLeapfrogProposal, GaussLegendreProposal
from odyssey.banana import Banana, generate_data
from odyssey.logistic import LogisticRegression, sigmoid
from odyssey.neal_funnel import NealFunnel


class TestProposal(unittest.TestCase):
    def test_euclidean_detailed_balance(self):
        t = 0.5
        sigma_theta = 2.0
        sigma_y = 2.0
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)
        distr = Banana(y, sigma_y, sigma_theta)
        proposal = EuclideanLeapfrogProposal(distr)
        qt = np.random.uniform(size=(2, ))
        state = proposal.first_state(qt)
        state.momentum = np.random.normal(size=qt.shape)
        step_size = np.random.uniform(0.0, 0.1)
        num_steps = int(np.ceil(10*np.random.uniform()))
        rev = proposal.reverse(state, step_size, num_steps)
        self.assertTrue(rev[0] < -14.0)
        logdet = proposal.jacobian_determinant(state, step_size, num_steps, 0.0, 1e-5)
        self.assertTrue(logdet < -9.0)

    def test_riemannian_detailed_balance(self):
        t = 0.5
        sigma_theta = 2.0
        sigma_y = 2.0
        theta, y = generate_data(t, sigma_y, sigma_theta, 100)
        distr = Banana(y, sigma_y, sigma_theta)
        proposal = RiemannianLeapfrogProposal(distr, 1e-12, 1000)
        qt = np.random.uniform(size=(2, ))
        state = proposal.first_state(qt)
        state.momentum = np.random.normal(size=qt.shape)
        step_size = np.random.uniform(0.0, 0.01)
        num_steps = int(np.ceil(25*np.random.uniform()))
        rev = proposal.reverse(state, step_size, num_steps)
        self.assertTrue(rev[0] < -9.0)
        logdet = proposal.jacobian_determinant(state, step_size, num_steps, 0.0, 1e-5)
        self.assertTrue(logdet < -9.0)

    def test_softabs_detailed_balance(self):
        num_dims = int(np.ceil(10*np.random.uniform()))
        distr = NealFunnel(num_dims)
        proposal = SoftAbsLeapfrogProposal(1e4, distr, 1e-12, 1000)
        qt = np.hstack(distr.sample())
        state = proposal.first_state(qt)
        state.momentum = np.random.normal(size=qt.shape)
        step_size = np.random.uniform(0.0, 0.01)
        num_steps = int(np.ceil(25*np.random.uniform()))
        rev = proposal.reverse(state, step_size, num_steps)
        self.assertTrue(rev[0] < -9.0)
        logdet = proposal.jacobian_determinant(state, step_size, num_steps, 0.0, 1e-5)
        self.assertTrue(logdet < -6.0)

    def test_coupled_detailed_balance(self):
        num_dims = int(np.ceil(10*np.random.uniform()))
        distr = NealFunnel(num_dims)
        proposal = CoupledLeapfrogProposal(1e4, distr, 0.0, 1e-12, 1000)
        qt = np.hstack(distr.sample())
        state = proposal.first_state(qt)
        state.momentum = np.random.normal(size=qt.shape)
        step_size = np.random.uniform(0.0, 0.1)
        num_steps = int(np.ceil(25*np.random.uniform()))
        rev = proposal.reverse(state, step_size, num_steps)
        self.assertTrue(rev[0] < -9.0)
        logdet = proposal.jacobian_determinant(state, step_size, num_steps, 0.0, 1e-5)
        self.assertTrue(logdet < -6.0)

    def test_lobatto_detailed_balance(self):
        num_dims = int(np.ceil(10*np.random.uniform()))
        distr = NealFunnel(num_dims)
        proposal = LobattoProposal(1e4, distr, 1e-12, 1000)
        qt = np.hstack(distr.sample())
        state = proposal.first_state(qt)
        state.momentum = np.random.normal(size=qt.shape)
        step_size = np.random.uniform(0.0, 0.1)
        num_steps = int(np.ceil(25*np.random.uniform()))
        rev = proposal.reverse(state, step_size, num_steps)
        self.assertTrue(rev[0] < -9.0)
        logdet = proposal.jacobian_determinant(state, step_size, num_steps, 0.0, 1e-5)
        self.assertTrue(logdet < -6.0)

    def test_gauss_legendre_detailed_balance(self):
        order = np.random.choice([2, 4, 6])
        if np.random.uniform() < 0.5:
            num_dims = int(np.ceil(10*np.random.uniform()))
            distr = NealFunnel(num_dims)
            proposal = GaussLegendreProposal(1e4, distr, order, 1e-12, 1000)
            qt = np.hstack(distr.sample())
        else:
            t = 0.5
            sigma_theta = 2.0
            sigma_y = 2.0
            theta, y = generate_data(t, sigma_y, sigma_theta, 100)
            distr = Banana(y, sigma_y, sigma_theta)
            proposal = GaussLegendreProposal(0.0, distr, order, 1e-12, 1000)
            qt = np.random.uniform(size=(2, ))

        state = proposal.first_state(qt)
        state.momentum = np.random.normal(size=qt.shape)
        step_size = np.random.uniform(0.0, 0.1)
        num_steps = int(np.ceil(25*np.random.uniform()))
        num_steps = 1
        rev = proposal.reverse(state, step_size, num_steps)
        self.assertTrue(rev[0] < -9.0)
        logdet = proposal.jacobian_determinant(state, step_size, num_steps, 0.0, 1e-5)
        self.assertTrue(logdet < -6.0)

    def test_lagrangian_detailed_balance(self):
        if np.random.uniform() < 0.5:
            t = 0.5
            sigma_theta = 2.0
            sigma_y = 2.0
            theta, y = generate_data(t, sigma_y, sigma_theta, 100)
            distr = Banana(y, sigma_y, sigma_theta)
            qt = np.random.uniform(size=(2, ))
        else:
            num_obs, num_dims = 100, 5
            x = np.random.normal(size=(num_obs, num_dims))
            b = np.ones((x.shape[-1], ))
            p = sigmoid(x@b)
            y = np.random.binomial(1, p)
            alpha = 0.5
            distr = LogisticRegression(x, y, alpha)
            qt = np.random.uniform(size=(num_dims, ))
        inverted = np.random.uniform() < 0.5
        proposal = LagrangianLeapfrogProposal(distr, inverted)
        state = proposal.first_state(qt)
        state.momentum = state.sqrtm_metric@np.random.normal(size=qt.shape)
        step_size = np.random.uniform(0.0, 0.1)
        num_steps = int(np.ceil(100*np.random.uniform()))
        new_state, info = proposal.propose(state, step_size, num_steps)
        rev = proposal.reverse(state, step_size, num_steps)
        self.assertTrue(rev[0] < -9.0)
        logdet = proposal.jacobian_determinant(state, step_size, num_steps, info.logdet, 1e-5)
        check = logdet < -7.0
        if not check:
            print(logdet)
        self.assertTrue(check)

        nv, ld = velocity_step(state, step_size)
        nvp, ldp = velocity_step_alt(state, step_size)
        self.assertTrue(np.allclose(ld, ldp))
        self.assertTrue(np.allclose(nv, nvp))
