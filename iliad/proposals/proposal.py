import abc
import copy
import functools
from typing import Callable, Optional, Tuple

import numpy as np

from odyssey.distribution import Distribution

from .diagnostics import Diagnostics
from ..integrators.info import Info
from ..integrators.states import State


class ProposalInfo:
    """Diagnostic information from running the Hamiltonian Monte Carlo sampling
    procedure.

    Parameters:
        accept: The acceptance probability of the Markov chain.
        num_iters: The number of internal iterations computed by the numerical
            integrator.
        absrev: The absolute error in the reversibility of the integrator.
        relrev: The relative error in the reversibility of the integrator.
        jacdet: The error in the volume preservation of the integrator as
            measured by its difference from a unit Jacobian.
        invalid: Counter of how many invalid proposals have been generated.

    """
    def __init__(self):
        self.absrev = Diagnostics()
        self.relrev = Diagnostics()
        self.jacdet = {
            t: Diagnostics() for t in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        }
        self.invalid = Diagnostics()
        self.accprob = Diagnostics()

    def asdict(self):
        d = {
            'acc. prob.': self.accprob.avg,
            'rev. err.': self.absrev.avg,
            'jac. det. err.': self.jacdet[1e-5].avg,
            'invalid': self.invalid.summ
        }
        return d

def error_intercept(proposal):
    """Intercepts proposals to check for error states and to intercept them before
    they propagate any further.

    Args:
        proposal: Function to generate a proposal given a current state, a number
            of integration steps, and an integration step-size.

    Returns:
        wrap: The wrapped proposal that intercepts errors.

    """
    @functools.wraps(proposal)
    def wrap(*args, **kwargs):
        try:
            value = proposal(*args, **kwargs)
        except (ValueError, np.linalg.LinAlgError):
            state = args[1]
            info = Info()
            info.success = False
            info.invalid = True
            value = (state, info)
        return value
    return wrap

def momentum_negation(proposal):
    """Intercepts the proposal to negate the momentum variable, thereby producing a
    reversible transition kernel.

    Args:
        proposal: Function to generate a proposal given a current state, a number
            of integration steps, and an integration step-size.

    Returns:
        wrap: The wrapped proposal that negates the momentum.

    """
    @functools.wraps(proposal)
    def wrap(*args, **kwargs):
        value = proposal(*args, **kwargs)
        state, info = value
        state.momentum *= -1.0
        state.velocity *= -1.0
        return state, info
    return wrap

class Proposal(abc.ABC):
    """The proposal object provides a generic interface to two methods. The first
    computes a proposal for use in Hamiltonian Monte Carlo by integrating
    Hamilton's equations of motion with a given step-size and a prescribed
    number of steps. The second is a method to generate an initial state from a
    given initial position in phase space.

    Parameters:
        distr: The distribution from which to sample.
        info: Diagnostic information about the proposal.
        langevin_type: Indicates compatibility with Langevin proposals.

    """
    def __init__(self, distr: Distribution, info: ProposalInfo, langevin_type: Optional[str]=None):
        self.distr = distr
        self.info = info
        self.langevin_type = langevin_type

    @abc.abstractmethod
    def propose(
            self,
            state: State,
            step_size: float,
            num_steps: int
    ) -> Tuple[State, Info]:
        raise NotImplementedError()

    @abc.abstractmethod
    def first_state(self, qt: np.ndarray) -> State:
        raise NotImplementedError()

    def update_detailed_balance(self, state: State, step_size: float, num_steps: int, info: Info):
        for k in self.info.jacdet.keys():
            det = self.jacobian_determinant(state, step_size, num_steps, info.logdet, k)
            self.info.jacdet[k].update(det)
        absrev, relrev = self.reverse(state, step_size, num_steps)
        self.info.absrev.update(absrev)
        self.info.relrev.update(relrev)

    def update_acceptance_rate(self, metropolis: float):
        ap = 1.0 if metropolis > 0 else np.exp(metropolis)
        self.info.accprob.update(ap)

    def reverse(self, state: State, step_size: float, num_steps: int) -> Tuple[float, float]:
        """Compute the reversibility of the proposal operator by first integrating
        forward, then flipping the sign of the momentum, integrating again, and
        flipping the sign of the momentum a final time in order to compute the
        distance between the original position and the terminal position. If
        the operator is symmetric (reversible) then this distance should be
        very small.

        Args:
            state: The state of the Markov chain.
            step_size: Integration step-size.
            num_steps: Number of integration steps.

        Returns:
            log_abserr: The absolute error of the original position in phase space
                and the terminal position of the proposal operator.
            log_relerr: The relative error of the original position in phase space
                and the terminal position of the proposal operator.

        """
        q, p = state.position, state.momentum
        sp, info_p = self.propose(state, step_size, num_steps)
        sr, info_r = self.propose(sp, step_size, num_steps)
        qr, pr = sr.position, sr.momentum
        rev = np.sqrt(np.square(np.linalg.norm(q - qr)) + np.square(np.linalg.norm(p - pr)))
        abserr = np.maximum(rev, 1e-16)
        relerr = abserr / np.sqrt(np.square(np.linalg.norm(q)) + np.square(np.linalg.norm(p)))
        success = info_p.success and info_r.success
        log_abserr = np.log10(abserr) if success else np.nan
        log_relerr = np.log10(relerr) if success else np.nan
        return log_abserr, log_relerr

    def jacobian_determinant(self, state: State, step_size: float, num_steps: int, logdetact, delta: float) -> float:
        """Compute the Jacobian of the transformation for a single sample consisting of
        a position and momentum.

        Args:
            state: The state of the Markov chain.
            step_size: Integration step-size.
            num_steps: Number of integration steps.
            logdetact: The actual (theoretical) log-determinant of the Jacobian.
            delta: Perturbation size for numerical Jacobian.

        Returns:
            err: The logarithm of the error between the Jacobian determinant of the
                transformation computed using finite differences and unity.

        """
        # Redefine the proposal operator as a map purely to phase-space to
        # phase-space with no additional inputs or outputs.
        def proposal(z):
            q, p = np.split(z, 2)
            s = copy.copy(state)
            s.position = q
            s.momentum = p
            s.update(self.distr)
            prop, info = self.propose(s, step_size, num_steps)
            if info.success:
                zp = np.hstack((prop.position, prop.momentum))
            else:
                zp = np.full(z.shape, np.nan)
            return zp

        z = np.hstack((state.position, state.momentum))
        Jac = jacobian(proposal, delta)(z)
        det = np.abs(np.linalg.det(Jac))
        err = np.maximum(np.abs(det - np.exp(logdetact)), 1e-16)
        err = np.log10(err)
        return err


def jacobian(func: Callable, delta: float):
    """Finite differences approximation to the Jacobian."""
    def jacfn(z):
        num_dims = len(z)
        Jac = np.zeros((num_dims, num_dims))
        for j in range(num_dims):
            pert = np.zeros(num_dims)
            pert[j] = 0.5 * delta
            zh = func(z + pert)
            zl = func(z - pert)
            Jac[j] = (np.hstack(zh) - np.hstack(zl)) / delta
        return Jac
    return jacfn
