from typing import Tuple

import numpy as np


def expected_mahalanobis(sigma, diff, accprob):
    sigma_inv = np.linalg.inv(sigma)
    maha = np.sum((diff@sigma_inv) * diff, axis=-1)
    emaha = maha * accprob
    not_nan = ~np.isnan(emaha)
    emaha = emaha[not_nan]
    return emaha, not_nan

def jump_statistics(sigma: np.ndarray, diff: np.ndarray, accprob: np.ndarray) -> Tuple[float, float]:
    """Computes statistics related to the squared jump distance between states of
    the Markov chain. This is the difference (under a Mahalanobis norm) between
    the proposal state and the current state of the chain, modulated by the
    acceptance probability of the proposal.

    Args:
        sigma: The covariance matrix of the samples, which is the inverse
            Mahalanobis matrix.
        diff: The difference between the current state of the chain and the
            proposal state.
        accprob: The probability of accepting the proposal.

    Returns:
        esjd: The expected squared jump distance.
        msjd: The median squared jump distance.

    """
    emaha, _ = expected_mahalanobis(sigma, diff, accprob)
    esjd = np.mean(emaha)
    msjd = np.median(emaha)
    return esjd, msjd

def timed_jump_statistics(sigma: np.ndarray, diff: np.ndarray, accprob: np.ndarray, elapsed: np.ndarray) -> Tuple[float, float]:
    emaha, not_nan = expected_mahalanobis(sigma, diff, accprob)
    emaha_div_elapsed = emaha / elapsed[not_nan]
    tesjd = np.mean(emaha_div_elapsed)
    tmsjd = np.median(emaha_div_elapsed)
    return tesjd, tmsjd
