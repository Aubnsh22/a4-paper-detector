import cv2
import numpy as np
import pytest

from a4_detector.a4_detection import A4DetectionResult
from a4_detector.reference import reference_assessment
from a4_detector.product_a4 import detect_product_reference


def checkerboard_capture(foot_length_px=300):
    image = np.zeros((1280, 960, 3), np.uint8)
    for y in range(0, 1280, 80):
        for x in range(0, 960, 80):
            image[y:y+80, x:x+80] = (175, 197, 217) if (x//80 + y//80) % 2 else (25, 30, 30)
    corners = np.array([[200, 440], [536, 440], [536, 915], [200, 915]], np.float32)
    cv2.fillConvexPoly(image, corners.astype(int), (224, 222, 204))
    cv2.ellipse(image, (690, 680), (48, foot_length_px//2), 0, 0, 360, (115, 146, 182), -1)
    return image, corners


@pytest.mark.parametrize("foot_length_px", [200, 280, 400])
def test_floor_connected_reference_recovers_observed_paper_for_different_foot_sizes(foot_length_px):
    image, expected = checkerboard_capture(foot_length_px)
    bad_corners = expected.copy()
    bad_corners[0] -= [85, 55]
    original = A4DetectionResult(success=True, corners=bad_corners, confidence=.79,
                                boundary_residual_px=47, corner_uncertainty_px=4, visible_corner_count=4)
    assert not reference_assessment(image, original)["valid"]
    actual, assessment = detect_product_reference(image, initial=original)
    assert assessment["valid"]
    assert actual.boundary_hypothesis == "isolated_color_boundary"
    assert np.max(np.linalg.norm(actual.corners - expected, axis=1)) < 3
    assert assessment["alignment"]["tilt_deg"] < 3
    assert actual.visible_corner_count == 4 and not actual.three_corner_reconstruction


def test_accepted_reference_keeps_exact_original_calibration(monkeypatch):
    image, corners = checkerboard_capture()
    original = A4DetectionResult(success=True, corners=corners, confidence=.9,
                                boundary_residual_px=1, corner_uncertainty_px=1, visible_corner_count=4)
    def unexpected(*args, **kwargs):
        pytest.fail("A valid existing calibration must not be replaced")
    monkeypatch.setattr("a4_detector.product_a4._find_candidates", unexpected)
    actual, assessment = detect_product_reference(image, initial=original)
    assert actual is original
    assert assessment["valid"]


@pytest.mark.parametrize("foot_length_px", [200, 400])
def test_paper_with_same_saturation_and_brightness_as_floor_uses_observed_colour_edges(foot_length_px):
    image, expected = checkerboard_capture(foot_length_px)
    # Exchanging blue and red preserves HSV saturation/value. All brightness
    # masks merge the sheet with the warm tiles; the colour boundary is visible.
    floor = np.all(image == (175, 197, 217), axis=2)
    image[floor] = (204, 222, 224)
    missing = A4DetectionResult(message="No reliable reference.")
    actual, assessment = detect_product_reference(image, initial=missing)
    assert assessment["valid"]
    assert actual.boundary_hypothesis == "isolated_chromatic_boundary"
    assert np.max(np.linalg.norm(actual.corners - expected, axis=1)) < 3
    assert actual.visible_corner_count == 4 and not actual.three_corner_reconstruction


@pytest.mark.parametrize("condition", ["missing", "multiple", "cropped", "tilted"])
def test_fallback_does_not_certify_missing_ambiguous_or_unusable_paper(condition):
    image = np.full((1280, 960, 3), (40, 55, 65), np.uint8)
    if condition == "multiple":
        for x in (80, 540):
            cv2.rectangle(image, (x, 400), (x+300, 824), (224, 222, 204), -1)
    elif condition == "cropped":
        cv2.rectangle(image, (-30, 400), (300, 867), (224, 222, 204), -1)
    elif condition == "tilted":
        quad = np.array([[210, 460], [410, 460], [540, 930], [80, 930]], np.int32)
        cv2.fillConvexPoly(image, quad, (224, 222, 204))
    missing = A4DetectionResult(message="No reliable reference.")
    actual, assessment = detect_product_reference(image, initial=missing)
    assert not assessment["valid"]
    assert actual is missing
