from typing import Tuple

import numpy as np

from iliad.linalg import eigen_to_matrix


def coth(x: np.ndarray) -> np.ndarray:
    """Implements the hyperbolic cotangent function."""
    idx = np.abs(x) > 1e-8
    out = np.zeros_like(x)
    out[idx] = np.reciprocal(np.tanh(x[idx]))
    return out

def softabs(lam: np.ndarray, alpha: float) -> np.ndarray:
    """The softabs transformation that applies a smooth absolute value-like
    transformation to the input. The sharpness of the transformation about zero
    is controlled by the parameter `alpha`.

    Args:
        lam: Values to smoothly transform under the softabs operation.
        alpha: The softabs smoothness parameter.

    Returns:
        slam: The transformed values under the softabs operation.

    """
    slam = lam * coth(alpha * lam)
    return slam

csch = lambda z: np.reciprocal(np.sinh(z))
cschsq = lambda z: np.square(csch(z))
softabs_deriv = lambda l, alpha: coth(alpha * l) - alpha * l * cschsq(alpha * l)

def j_matrix(l: np.ndarray, lt: np.ndarray, alpha: float) -> np.ndarray:
    """Helper function to compute the `J` matrix in the derivation of the softabs
    metric. Some care must be taken to appropriately handle repeated eigenvalues.

    See [1] for treatment of the softabs metric.

    [1] https://arxiv.org/abs/1212.4693

    Args:
        l: The original eigenvalues.
        lt: The eigenvalues transformed by the softabs function.
        alpha: Parameter controlling the sharpness of the softabs function.

    Returns:
        J: The auxiliary `J` matrix derived in the softabs metric.

    """
    j_den_p = l - np.expand_dims(l, -1)
    deg = np.abs(j_den_p) < 1e-10
    j_den_p = np.where(deg, np.ones_like(j_den_p), j_den_p)
    deriv = softabs_deriv(l, alpha)
    deriv = deriv.repeat(l.size).reshape((l.size, -1)).T
    j_num_p = lt - np.expand_dims(lt, -1)
    j_num_p = np.where(deg, deriv, j_num_p)
    J = j_num_p / j_den_p
    return J

def force(momentum: np.ndarray,
          grad_log_posterior: np.ndarray,
          dH: np.ndarray,
          l: np.ndarray,
          lt: np.ndarray,
          inv_lt: np.ndarray,
          U: np.ndarray,
          alpha: float) -> np.ndarray:
    """Computes the time derivative of the momentum for a SoftAbs Riemannian
    metric.

    Args:
        momentum: The momentum of the system.
        grad_log_posterior: The gradient of the log-posterior from which to
            sample.
        dH: The Jacobian of the Hessian from which the SoftAbs metric is
            computed.
        l: The eigenvalues of the Hessian.
        lt: The softabs transformation of the eigenvalues of the Hessian; these
            are the eigenvalues of the metric.
        inv_lt: The reciprocals of the eigenvalues of the metric, which are the
            eigenvalues of the inverse metric.
        U: The eigenvectors of the Hessian.
        alpha: The SoftAbs sharpness parameter.

    Returns:
        force: The time derivative of the momentum.

    """
    UT = U.T
    R = np.diag(inv_lt)
    D = np.diag(UT@momentum / lt)
    J = j_matrix(l, lt, alpha)
    UTdH = UT@dH
    grad_log_det = np.trace(U@(R*J)@UTdH, axis1=1, axis2=2)
    grad_quadratic = -np.trace(U@D@J@D@UTdH, axis1=1, axis2=2)
    force = grad_log_posterior - 0.5*grad_log_det - 0.5*grad_quadratic
    return force

def decomposition(H: np.ndarray, alpha: float) -> Tuple[np.ndarray, ...]:
    """Given a Hessian matrix and the prescribed sharpness parameter of the SoftAbs
    transformation, computes the eigen-decomposition of the Hessian to
    construct the SoftAbs metric, its inverse, and associated quantities.

    Args:
        H: Hessian matrix.
        alpha: SoftAbs sharpness parameter.

    Returns:
        l: Eigenvalues of the Hessian.
        U: Eigenvectors of the Hessian.
        lt: Eigenvalues of the Hessian transformed according to the SoftAbs
            function, which is a smooth approximation to the absolute value.
        inv_lt: The reciprocal of the eigenvalues of the metric, which are the
            eigenvalues of the inverse metric.
        metric: The SoftAbs metric.
        inv_metric: The inverse of the SoftAbs metric.

    """
    l, U = np.linalg.eigh(H)
    lt = softabs(l, alpha)
    inv_lt = np.reciprocal(lt)
    metric = eigen_to_matrix(lt, U)
    inv_metric = eigen_to_matrix(inv_lt, U)
    return l, U, lt, inv_lt, metric, inv_metric
