import numpy as np


def cond(val: tuple, thresh: float, max_iters: int) -> bool:
    """Terminate the fixed point iteration when the change between the previous and
    current iterate is below tolerance or (more dangerously) if the maximum
    number of iterations has been exceeded.

    Args:
        val: Tuple containing the value of the variable whose fixed point is sought
            and auxiliary data such as the number of fixed point iterations so far
            and the deviation of the current fixed point proposal from the previous
            one.
        thresh: Convergence threshold for the fixed point iteration.
        max_iters: Maximum number of fixed point iterations.

    Returns:
        c: A boolean indicator of whether the fixed point iteration should continue
            if true or terminate if false.

    """
    delta, num_iters = val[-2:]
    c = np.logical_and(np.any(np.abs(delta) > thresh), num_iters < max_iters)
    return c
