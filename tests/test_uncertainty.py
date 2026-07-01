import pytest

from src.uncertainty import normalize_probabilities, predictive_entropy, route_case, top2_margin


def test_uncertainty_confident_case_routes_without_review():
    routed = route_case("accepted", [0.01, 0.02, 0.92, 0.03, 0.02])
    assert routed["manual_review"] is False
    assert routed["confidence"] == pytest.approx(0.92)
    assert predictive_entropy([1, 0, 0, 0, 0]) < 0.01
    assert top2_margin([0.7, 0.2, 0.1]) == pytest.approx(0.5)


def test_uncertainty_missing_model_routes_to_review():
    routed = route_case("accepted", None)
    assert routed["manual_review"] is True
    assert routed["reason"] == "model_missing"


def test_probability_validation():
    with pytest.raises(ValueError):
        normalize_probabilities([0, 0, 0])