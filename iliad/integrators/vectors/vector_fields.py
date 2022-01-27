from typing import Callable, Tuple

import numpy as np

from odyssey.distribution import Distribution

from iliad.integrators.fields import riemannian, softabs
from iliad.linalg import solve_psd


def riemannian_velocity_and_force(distr: Distribution) -> Tuple[Callable]:
    """Function to produce functions for computing the velocity and force vector
    fields given a Riemannian Hamiltonian system.

    Args:
        distr: Distribution from which to construct the Riemannian vector field.

    Returns:
        velocity_vector: Function computing the time derivative of position.
        force_vector: Function computing the time derivative of momentum.

    """
    def velocity_vector(q, p):
        G = distr.riemannian_metric(q)
        vel, _ = solve_psd(G, p)
        return vel

    def force_vector(q, p):
        num_dims = len(q)
        lp, glp, metric, jac_metric = distr.riemannian_quantities(q)
        inv_metric, _ = solve_psd(metric)
        grad_logdet_metric = riemannian.grad_logdet(inv_metric, jac_metric, num_dims)
        vel = inv_metric@p
        force = riemannian.force(vel, glp, jac_metric, grad_logdet_metric)
        return force
    return velocity_vector, force_vector

def softabs_vector_field(distr: Distribution, alpha: float) -> Callable:
    """Computes the velocity and force, which are the time derivatives of position
    and momentum, given the auxiliaries of a SoftAbs vector field, which
    includes the gradient of the log-posterior, the Hessian of the
    log-posterior, and the Jacobian of the Hessian.

    Args:
        distr: The distribution from which to assemble the vector field.
        alpha: SoftAbs sharpness parameter.

    Returns:
        vector_field: A function returning the time derivatives of position and
            momentum.

    """
    def vector_field(q, p):
        _, glp, H, dH = distr.softabs_quantities(q)
        l, U, lt, inv_lt, metric, inv_metric = softabs.decomposition(H, alpha)
        velocity = riemannian.velocity(inv_metric, p)
        force = softabs.force(p, glp, dH, l, lt, inv_lt, U, alpha)
        return velocity, force
    return vector_field

def riemannian_vector_field(distr: Distribution) -> Callable:
    """Computes the velocity and force, which are the time derivatives of position
    and momentum, given the auxiliaries of a Riemannian vector field, which
    includes the gradient of the log-posterior, the Riemannian metric, and the
    Jacobian of the Riemannian metric.

    Args:
        distr: The distribution from which to assemble the vector field.

    Returns:
        vector_field: A function returning the time derivatives of position and
            momentum.

    """
    def vector_field(q, p):
        num_dims = len(q)
        lp, glp, metric, jac_metric = distr.riemannian_quantities(q)
        inv_metric, _ = solve_psd(metric)
        dld = riemannian.grad_logdet(inv_metric, jac_metric, num_dims)
        velocity = riemannian.velocity(inv_metric, p)
        force = riemannian.force(velocity, glp, jac_metric, dld)
        return velocity, force
    return vector_field
