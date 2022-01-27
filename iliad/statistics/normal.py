from typing import Optional, Union

import numpy as np


def rvs(sqrtm: np.ndarray, mu: Optional[Union[np.ndarray, float]]=0.0) -> np.ndarray:
    """Generates a single draw from a (by default) zero-mean multivariate normal
    distribution whose covariance is provided by its square root factor.

    Args:
        sqrtm: A square root factor of the covariance matrix of the multivariate
            normal distribution.
        mu: The mean of the normal distribution.

    Returns:
        z: Random sample from the normal distribution.

    """
    z = sqrtm.dot(np.random.normal(size=sqrtm.shape[0])) + mu
    return z

def logpdf(z: np.ndarray, logdet: float, inv_z: np.ndarray) -> float:
    """Computes the log-density of the zero-mean multivariate normal density when
    provided with the log-determinant of the covariance matrix and the inverse
    covariance matrix.

    Args:
        z: Location at which to evaluate the log-density.
        logdet: The log-determinant of the covariance matrix.
        inv_z: The inverse of the covariance matrix of the multivariate normal
            distribution applied to the vector `z`.

    Returns:
        ld: The log-density of the zero-mean multivariate normal at the specified
            input location.

    """
    n = len(z)
    ld = -0.5*n*np.log(2*np.pi) - 0.5*logdet - 0.5*np.matmul(z, inv_z)
    return ld
