import numpy as np

from iliad.statistics.normal import logpdf


def hamiltonian(
        momentum: np.ndarray,
        log_posterior: float,
        logdet: float,
        velocity: np.ndarray
) -> float:
    """Hamiltonian for sampling from the distribution.

    Args:
        momentum: The momentum variable at which to evaluate the Hamiltonian.
        log_posterior: The value of the log-posterior representing the negative
            potential energy of the system.
        logdet: The log-determinant of the covariance matrix.
        velocity: The velocity variable, which is the inverse metric multiplied
            into the momentum.

    Returns:
        H: The value of the Hamiltonian, representing the total energy of the
            system; this is the sum of the potential and kinetic energies.

    """
    U = -log_posterior
    K = -logpdf(momentum, logdet, velocity)
    H = U + K
    return H
