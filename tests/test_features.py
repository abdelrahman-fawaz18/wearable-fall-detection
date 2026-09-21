import numpy as np

from wearable_fall_detection.features import autocorrelation, extract_features, feature_names


def test_autocorrelation_of_linear_signal_is_one() -> None:
    assert np.isclose(autocorrelation(np.arange(10), lag=3), 1.0)


def test_constant_signal_has_zero_autocorrelation() -> None:
    assert autocorrelation(np.ones(20), lag=2) == 0.0


def test_feature_vector_has_expected_schema() -> None:
    rng = np.random.default_rng(7)
    features = extract_features(rng.normal(size=(50, 6)))
    assert features.shape == (84,)
    assert len(feature_names()) == 84
    assert np.isfinite(features).all()
