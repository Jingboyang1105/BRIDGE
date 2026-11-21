"""Small readable helpers for the final missing-tissue prediction step."""

from .dorian_model import combine_expert_predictions


def predict_missing_tissue(same_donor_prediction, memory_prediction, mixer_weights):
    """Return DORIAN's final imputed target-tissue expression profile."""

    return combine_expert_predictions(
        same_donor_prediction=same_donor_prediction,
        memory_prediction=memory_prediction,
        expert_weights=mixer_weights,
    )

