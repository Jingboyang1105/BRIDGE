"""Small readable helpers for the final missing-tissue prediction step."""

from .bridge_model import combine_expert_predictions


def predict_missing_tissue(donor_level_average_prediction, sample_matching_prediction, mixer_weights):
    """Return BRIDGE's final imputed target-tissue expression profile."""

    return combine_expert_predictions(
        donor_level_average_prediction=donor_level_average_prediction,
        sample_matching_prediction=sample_matching_prediction,
        expert_weights=mixer_weights,
    )

