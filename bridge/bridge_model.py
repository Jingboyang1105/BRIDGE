"""The final BRIDGE prediction rule that combines its two experts."""

import numpy as np


def combine_expert_predictions(donor_level_average_prediction, sample_matching_prediction, expert_weights):
    """Combine the two full expression predictions using record-specific weights."""

    donor_level_average = np.asarray(donor_level_average_prediction, dtype=np.float32)
    sample_matching = np.asarray(sample_matching_prediction, dtype=np.float32)
    weights = np.asarray(expert_weights, dtype=np.float32)
    if donor_level_average.shape != sample_matching.shape:
        raise ValueError("The two expert predictions must have identical shapes.")
    if weights.shape[-1] != 2:
        raise ValueError("BRIDGE requires one weight for each of its two experts.")
    return (
        weights[..., 0, None] * donor_level_average
        + weights[..., 1, None] * sample_matching
    )

