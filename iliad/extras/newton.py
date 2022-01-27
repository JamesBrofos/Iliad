import numpy as np

from iliad.linalg import solve_psd

from odyssey.distribution import Distribution


def newton_raphson(q: np.ndarray, distr: Distribution, tol: float=1e-10) -> np.ndarray:
    """Implements the Newton-Raphson algorithm to find the maximum a posteriori of
    the posterior.

    Args:
        q: Initial guess for the location of the maximum of the posterior.
        distr: Distribution whose maximum a posteriori should be located.
        tol: The convergence tolerance for Newton iterations.

    Returns:
        q: The maximizer of the posterior density.

    """
    delta = np.inf
    while delta > tol:
        _, g, G, _ = distr.riemannian_quantities(q)
        q += solve_psd(G, g)[0]
        delta = np.abs(g).max()
    return q
