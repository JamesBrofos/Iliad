import numpy as np


def eigen_to_matrix(eigen: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Computes the vector whose eigen-decomposition is provided. Assumes that the
    eigenvectors are orthonormal.

    Args:
        eigen: Eigenvalues of the matrix.
        vectors: Eigenvectors of the matrix.

    Returns:
        matrix: The matrix with the provided eigen-decomposition.

    """
    matrix = np.dot(vectors, (eigen*vectors).T)
    matrix = np.real(matrix)
    return matrix
