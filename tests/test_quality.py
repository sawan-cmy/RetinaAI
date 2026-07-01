from src.quality_check import assess_quality


def test_quality_accepts_synthetic_retina(synthetic_retina):
    result = assess_quality(synthetic_retina)
    assert result.status == "accepted"
    assert result.retina_visibility_score > 0


def test_quality_rejects_dark_image(dark_image):
    result = assess_quality(dark_image)
    assert result.status == "rejected"
    assert "image_too_dark" in result.reasons