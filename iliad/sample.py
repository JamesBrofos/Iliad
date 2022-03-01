import time
from typing import Tuple

import numpy as np

from iliad.hamiltonian import hamiltonian
from iliad.integrators.euler_maruyama import euler_maruyama
from iliad.integrators.info import Info
from iliad.integrators.states import State
from iliad.proposals import Proposal, ProposalInfo
from iliad.statistics.normal import rvs


def langevin_metropolis_hastings(
        proposal: Proposal,
        state: State,
        step_size: float,
        unif: float
) -> Tuple[State, np.ndarray, bool]:
    """Computes the Metropolis-Hastings accept-reject decision to the
    manifold-adjusted Langevin algorithm. The step-size is interpreted as the
    discretization of time employed by the Euler-Maruyama integrator.

    Args:
        proposal: A proposal operator to advance the state of the Markov chain.
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        step_size: The integration step-size.
        unif: Uniform random number for determining the accept-reject decision.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        q: The position variable in the constrained space.
        accept: Whether or not the proposal was accepted.

    """
    start = time.time()
    q, fldj = proposal.distr.inverse_transform(state.position)
    ildj = -fldj
    new_state, mh = euler_maruyama(state, proposal.distr, step_size, proposal.langevin_type)
    new_q, new_fldj = proposal.distr.inverse_transform(new_state.position)
    logu = np.log(unif)
    metropolis = mh - new_fldj - ildj
    accprob = 1.0 if metropolis > 0 else np.exp(metropolis)
    accept = logu < metropolis
    proposal.update_acceptance_rate(metropolis)
    elapsed = time.time() - start
    if accept:
        state = new_state
        q = new_q
    return state, q, new_q, accept, accprob, elapsed

def hamiltonian_metropolis_hastings(
        proposal: Proposal,
        state: State,
        step_size: float,
        num_steps: int,
        unif: float,
        alpha: float,
        check_prob: float
) -> Tuple[State, Info, np.ndarray, bool]:
    """Computes the Metropolis-Hastings accept-reject criterion given a proposal, a
    current state of the chain, a integration step-size, and a number of
    itnegration steps. We also provide a uniform random variable for
    determining the accept-reject criterion and the inverse transformation
    function for transforming parameters from an unconstrained space to a
    constrained space.

    Args:
        proposal: A proposal operator to advance the state of the Markov chain.
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        step_size: The integration step-size.
        num_steps: The number of integration steps.
        unif: Uniform random number for determining the accept-reject decision.
        alpha: The partial momentum refreshment rate.
        check_prob: Probability to compute reversibility and volume preservation
            statistics for the proposal.

    Returns:
        state: An augmented state object with the updated position and momentum
            and values for the log-posterior and metric and their gradients.
        info: An information object with the updated number of fixed point
            iterations and boolean indicator for successful integration.
        q: The position variable in the constrained space.
        accept: Whether or not the proposal was accepted.

    """
    # Start time of the transition computation.
    start = time.time()
    # Sample momentum from conditional distribution and compute the associated
    # Hamiltonian energy.
    beta = np.sqrt(1-alpha**2)
    p = state.momentum
    n = rvs(state.sqrtm_metric)
    state.momentum = alpha*p + beta*n
    state.velocity = state.inv_metric@state.momentum
    ham = hamiltonian(
        state.momentum,
        state.log_posterior,
        state.logdet_metric,
        state.velocity
    )
    q, fldj = proposal.distr.inverse_transform(state.position)
    ildj = -fldj
    new_state, prop_info = proposal.propose(state, step_size, num_steps)
    new_chol, new_logdet = new_state.sqrtm_metric, new_state.logdet_metric
    new_q, new_fldj = proposal.distr.inverse_transform(new_state.position)
    new_ham = hamiltonian(
        new_state.momentum,
        new_state.log_posterior,
        new_state.logdet_metric,
        new_state.velocity
    )
    # Notice the relevant choice of sign when the Jacobian determinant of the
    # forward or inverse transform is used.
    #
    # Write this expression as,
    # (exp(-new_ham) / exp(new_fldj)) / (exp(-ham) * exp(ildj))
    #
    # See the following resource for understanding the Metropolis-Hastings
    # correction with a Jacobian determinant correction [1].
    #
    # [1] https://wiki.helsinki.fi/download/attachments/48865399/ch7-rev.pdf
    logu = np.log(unif)
    metropolis = ham - new_ham - new_fldj - ildj + prop_info.logdet
    accprob = 1.0 if metropolis > 0 else np.exp(metropolis)
    success = prop_info.success
    accept = np.logical_and(logu < metropolis, success)
    proposal.update_acceptance_rate(accprob, success)
    # Compute the elapsed time.
    elapsed = time.time() - start

    # Randomly check the properties of reversibility and volume preservation.
    # The time taken to compute the reversibility and volume preservation check
    # is excluded from the time elapsed for this iteration.
    random_check = np.random.uniform() < check_prob
    if random_check and not prop_info.invalid:
        proposal.update_detailed_balance(state, step_size, num_steps, prop_info)

    # Negation of the momentum a second time. This transformation keeps the
    # target distribution invariant and can be used in conjunction with partial
    # momentum refreshment.
    #
    # Nota bene: This modification must be applied after the random check of
    # detailed balance if the computed log-determinant involves the momentum
    # (such is the case of the Lagrangian proposal).
    state.momentum *= -1.0
    state.velocity *= -1.0

    # Apply the accept reject decision.
    if accept:
        state = new_state
        q = new_q
    return state, q, new_q, accept, accprob, elapsed

class SampleInfo:
    def __init__(self, sample: np.ndarray, proposal: np.ndarray, accepted: bool, accprob: float, elapsed: float):
        self.sample = sample
        self.proposal = proposal
        self.accepted = accepted
        self.accprob = accprob
        self.elapsed = elapsed

def sample(
        q: np.ndarray,
        step_size: float,
        num_steps: int,
        proposal: Proposal,
        alpha: float=0.0,
        check_prob: float=0.0,
        langevin_prob: float=0.0
) -> Tuple[np.ndarray, ProposalInfo]:
    """Draw samples from the target density using Hamiltonian Monte Carlo. This
    function requires that one specify a Hamiltonian energy, a proposal
    operator, and a function to sample momenta. This function is implemented as
    a generator so as to yield samples from the target distribution when
    requested.

    Args:
        q: The position variable.
        step_size: The integration step-size.
        num_steps: The number of integration steps.
        proposal: A proposal operator to advance the state of the Markov chain.
        alpha: Parameter controlling the partial refreshment of the momentum
            variable.
        check_prob: Probability to compute reversibility and volume preservation
            statistics for the proposal.
        langevin_prob: Probability to employ manifold MALA instead of single-step
            Riemannian manolf HMC.

    Returns:
        q: The next position variable.
        accept: Whether or not the proposal is accepted.
        elapsed: The time to compute the next state of the chain (excluding
            computation time for the metrics).

    """
    # Transform the position variable if, for instance, an unconstrained
    # representation is required.
    qt, ildj = proposal.distr.forward_transform(q)
    state = proposal.first_state(qt)
    del qt, ildj
    if langevin_prob > 0.0:
        if proposal.langevin_type is None:
            raise ValueError("Cannot employ manifold MALA with an incompatible HMC proposal.")
        lb = 2
    else:
        lb = 1

    while True:
        mh_unif, lv_unif = np.random.uniform(size=(2, ))
        if lv_unif < langevin_prob:
            # Metropolis-adjusted Langevin algorithm.
            state, q, prop_q, accepted, accprob, elapsed = langevin_metropolis_hastings(
                proposal,
                state,
                step_size,
                mh_unif
            )
        else:
            # HMC with a randomized number of integration steps.
            ns = np.random.randint(lb, num_steps + 1)
            state, q, prop_q, accepted, accprob, elapsed = hamiltonian_metropolis_hastings(
                proposal,
                state,
                step_size,
                ns,
                mh_unif,
                alpha,
                check_prob
            )
        s = SampleInfo(q, prop_q, accepted, accprob, elapsed)
        yield s
