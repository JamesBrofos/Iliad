from typing import Optional

import numpy as np
import scipy.linalg as spla


def solve_psd(A: np.ndarray, rhs: Optional[np.ndarray]=None):
    """Solve the system `A x = rhs` under the assumption that `A` is positive
    definite. The method implemented is to compute the Cholesky factorization
    of `A` and solve the system via forward-backward substitution.

    Args:
        A: Left-hand side of the linear system.
        rhs: Right-hand side of the linear system.

    Returns:
        x: Solution of the linear system.
        L: The Cholesky factor of the left-hand side of the linear system.

    """
    if rhs is None:
        rhs = np.eye(len(A))
    L = np.linalg.cholesky(A)
    x = spla.cho_solve((L, True), rhs)
    return x, L
