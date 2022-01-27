import numpy as np


def velocity(inv_metric: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    """Computes the velocity given the inverse Riemannian metric and the momentum
    of the particle.

    Args:
        inv_metric: The inverse of the Riemannian metric.
        momentum: The momentum variable.

    Returns:
        vel: The time derivative of position.

    """
    vel = np.matmul(inv_metric, momentum)
    return vel

def force(velocity: np.ndarray, grad_log_posterior: np.ndarray, jac_metric: np.ndarray, grad_log_det: np.ndarray) -> np.ndarray:
    """Computes the force acting on a particle, which is the time derivative of the
    particle's momentum given the state of the particle. The force is also the
    negative gradient of the Hamiltonian with respect to position.

    Args:
        velocity: The velocity variable.
        grad_log_posterior: The gradient of the log-posterior.
        jac_metric: The Jacobian of the metric tensor.
        grad_log_det: The gradient of the log-determinant of the Riemannian metric.

    Returns:
        force: The time derivative of momentum.

    """
    qform = 0.5 * velocity@np.einsum('ijk,j->ik', jac_metric, velocity)
    force = grad_log_posterior - grad_log_det + qform
    return force

def grad_logdet(inv_metric: np.ndarray, jac_metric: np.ndarray, num_dims: int) -> np.ndarray:
    """Computes the gradient of the log-determinant of the Riemannian metric.

    Args:
        inv_metric: The inverse of the Riemannian metric.
        jac_metric: The Jacobian of the metric tensor.
        num_dims: The number of dimensions of the position variable.

    Returns:
        grad_logdet: The gradient of the log-determinant of the Riemannian
            metric.

    """
    A = np.einsum('ij,jlk->ilk', inv_metric, jac_metric)
    grad_logdet = 0.5*np.trace(A, axis1=0, axis2=1)
    return grad_logdet
