"""Minimal demonstration of DORIAN's two-expert prediction interface."""

import numpy as np

from dorian import predict_missing_tissue


same_donor_prediction = np.array([1.2, 2.0, 3.1], dtype=np.float32)
sample_matching_prediction = np.array([1.5, 1.5, 3.2], dtype=np.float32)

prediction = predict_missing_tissue(
    same_donor_prediction=same_donor_prediction,
    memory_prediction=sample_matching_prediction,
    mixer_weights=np.array([0.6, 0.4], dtype=np.float32),
)

print(prediction)

