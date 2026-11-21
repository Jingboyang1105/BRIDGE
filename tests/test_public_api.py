import numpy as np

from dorian import predict_missing_tissue


def test_public_api_combines_two_expert_predictions():
    same_donor = np.array([1.2, 2.0, 3.1], dtype=np.float32)
    sample_matching = np.array([1.5, 1.5, 3.2], dtype=np.float32)

    prediction = predict_missing_tissue(
        same_donor_prediction=same_donor,
        memory_prediction=sample_matching,
        mixer_weights=np.array([0.6, 0.4], dtype=np.float32),
    )

    np.testing.assert_allclose(
        prediction,
        0.6 * same_donor + 0.4 * sample_matching,
        rtol=1e-6,
    )
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()

