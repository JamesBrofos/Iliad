import numpy as np
from scipy.spatial.distance import cdist

K = lambda x, y, bw: np.exp(-0.5*cdist(x, y, 'sqeuclidean') / bw**2)


def mmd(x: np.ndarray, y: np.ndarray, bw: float) -> float:
    """Computes the maximum mean discrepancy between two samples. This is a measure
    of the similarity of two distributions that generate the input samples.

    Args:
        x: First set of samples.
        y: Second set of samples.
        bw: Bandwidth parameter to use in computing the squared exponential
            kernel.

    Returns:
        u: An unbiased estimator of the maximum mean discrepancy.

    """
    m = len(x)
    n = len(y)
    a = 0.0
    b = 0.0
    c = 0.0
    for i in range(m):
        xp = x[[i]]
        Ka = K(xp, x, bw)
        Kc = K(xp, y, bw)
        a += np.sum(Ka) - Ka[0, i]
        c += np.sum(Kc)

    for i in range(n):
        yp = y[[i]]
        Kb = K(yp, y, bw)
        b += np.sum(Kb) - Kb[0, i]

    a /= m*(m-1)
    b /= n*(n-1)
    c /= -0.5*m*n
    u = a + b + c
    return u
