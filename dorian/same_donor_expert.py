"""DORIAN expert 1: deterministic residual transfer within the same donor."""

import numpy as np


def same_donor_residual_prediction(target_tissue_mean, observed_tissue_residuals):
    """Return the target-tissue mean plus the donor's mean observed residual.

    Parameters
    ----------
    target_tissue_mean
        Training-derived mean expression vector for the missing target tissue.
    observed_tissue_residuals
        Tissue-centered residual vectors from the same donor's observed tissues.
    """

    target_mean = np.asarray(target_tissue_mean, dtype=np.float32)
    residuals = np.asarray(observed_tissue_residuals, dtype=np.float32)
    if residuals.ndim != 2 or residuals.shape[0] == 0:
        raise ValueError("At least one observed-tissue residual vector is required.")
    return target_mean + residuals.mean(axis=0)

