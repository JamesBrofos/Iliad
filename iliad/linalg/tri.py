from typing import Optional

import numpy as np
import scipy.linalg as spla
import scipy.sparse as spsr


def solve_tri(tri: np.ndarray, rhs: Optional[np.ndarray]=None) -> np.ndarray:
    """The special structure of a tridiagonal matrix permits it to be used in
    solving a linear system in linear time instead of the usual cubic time.

    Args:
        tri: Tridiagonal matrix.
        rhs: Right-hand side of the linear system.

    Returns:
        x: The solution of the linear system involving a tridiagonal matrix.
        C: The Cholesky factor of the tridiagonal matrix.

    """
    if rhs is None:
        rhs = np.eye(tri.shape[-1])
    L = spla.cholesky_banded(tri)
    x = spla.cho_solve_banded((L, False), rhs)
    C = spsr.diags([L[0, 1:], L[1]], [-1, 0])
    return x, C

def chol_tri(tri: np.ndarray) -> np.ndarray:
    """The special structure of a tridiagonal matrix permits its Cholesky factor to
    be computed in linear time instead of cubic time.

    Args:
        tri: Tridiagonal matrix.

    Returns:
        C: The Cholesky factorization of the tridiagonal matrix.

    """
    c = spla.cholesky_banded(tri)
    C = spsr.diags([c[0, 1:], c[1]], [-1, 0])
    return C
