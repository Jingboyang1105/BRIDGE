"""Readable, manuscript-facing implementation of BRIDGE."""

from .adaptive_mixer import AdaptivePredictionMixer
from .bridge_model import combine_expert_predictions
from .sample_matching_expert import SampleMatchingEncoder, RecordDataset
from .predict_missing_tissue import predict_missing_tissue
from .donor_level_average_expert import donor_level_average_residual_prediction

__all__ = [
    "AdaptivePredictionMixer",
    "combine_expert_predictions",
    "SampleMatchingEncoder",
    "predict_missing_tissue",
    "RecordDataset",
    "donor_level_average_residual_prediction",
]

