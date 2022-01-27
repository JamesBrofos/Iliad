import numpy as np

from .eigen import eigen_to_matrix


def sqrtm(eigen: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Computes the matrix square root of the positive definite matrix. Computes an
    eigen-decomposition and defines the square root in terms of the square root
    of the eigenvalues.

    Args:
        eigen: Eigenvalues of the matrix whose square-root is to be computed.
        vectors: Eigenvectors of the matrix whose square-root is to be computed.

    Returns:
        square_root: The matrix square root.

    """
    eigen_sqrt = np.sqrt(eigen)
    square_root = eigen_to_matrix(eigen_sqrt, vectors)
    return square_root
