"""The final DORIAN prediction rule that combines its two experts."""

import numpy as np


def combine_expert_predictions(same_donor_prediction, memory_prediction, expert_weights):
    """Combine the two full expression predictions using record-specific weights."""

    same_donor = np.asarray(same_donor_prediction, dtype=np.float32)
    memory = np.asarray(memory_prediction, dtype=np.float32)
    weights = np.asarray(expert_weights, dtype=np.float32)
    if same_donor.shape != memory.shape:
        raise ValueError("The two expert predictions must have identical shapes.")
    if weights.shape[-1] != 2:
        raise ValueError("DORIAN requires one weight for each of its two experts.")
    return weights[..., 0, None] * same_donor + weights[..., 1, None] * memory

