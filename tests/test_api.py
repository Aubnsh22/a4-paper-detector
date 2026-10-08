"""Integration tests for acceptance, coordinates and CLI behavior."""
from dataclasses import replace
import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from a4_detector import compute_homography, detect_a4
from a4_detector.a4_detection import A4DetectionResult
from a4_detector.capture_policy import POLICY
from a4_detector.reference import reference_assessment


def paper(landscape=False):
    image = np.full((960, 1280, 3), 65, np.uint8)
    width, height = (446, 315) if landscape else (315, 446)
    corners = np.array([[160, 180], [160 + width, 180],
                        [160 + width, 180 + height], [160, 180 + height]], np.float32)
    cv2.fillConvexPoly(image, corners.astype(int), (245, 245, 245))
    return image, corners


@pytest.mark.parametrize("landscape", [False, True])
def test_reference_returns_source_corners_and_usable_plane_mapping(landscape):
    image, expected = paper(landscape)
    original = image.copy()
    result = detect_a4(image)
    assert result.valid, result.messages
    assert np.max(np.linalg.norm(result.corners - expected, axis=1)) < 4
    assert np.array_equal(image, original)
    assert result.detection.visible_corner_count == 4
    assert not result.detection.three_corner_reconstruction
    homography = compute_homography(result.corners)
    mapped = cv2.perspectiveTransform(result.corners.reshape(1, 4, 2), homography)[0]
    assert np.linalg.norm(mapped[1] - mapped[0]) == pytest.approx(297 if landscape else 210, abs=.01)
    assert np.linalg.norm(mapped[2] - mapped[1]) == pytest.approx(210 if landscape else 297, abs=.01)
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["image_size_wh"] == [1280, 960]
    assert payload["confidence_calibrated"] is False
    assert payload["coordinate_units"] == "pixels"


def test_no_reference_returns_no_usable_corners():
    result = detect_a4(np.full((960, 1280, 3), 65, np.uint8))
    assert not result.valid
    assert result.corners is None
    assert result.to_dict()["corners"] is None
    assert "a4_missing" in result.reasons
    json.dumps(result.to_dict(), allow_nan=False)


def test_small_source_image_is_rejected_even_when_paper_is_detected():
    image, _ = paper()
    result = detect_a4(cv2.resize(image, (640, 480)))
    assert not result.valid
    assert "resolution" in result.reasons
    assert result.corners is None


@pytest.mark.parametrize("image", [None, np.zeros((10, 10, 3), np.uint8),
    np.zeros((100, 100), np.uint8), np.zeros((100, 100, 4), np.uint8),
    np.zeros((100, 100, 3), np.float32), np.zeros((0, 100, 3), np.uint8)])
def test_invalid_image_contract_has_a_clear_error(image):
    with pytest.raises(ValueError, match="uint8 BGR"):
        detect_a4(image)


@pytest.mark.parametrize("change,reason", [
    ({"visible_corner_count": 3, "three_corner_reconstruction": True}, "a4_corners"),
    ({"confidence": .1}, "a4_boundary"),
    ({"corner_uncertainty_px": 30}, "a4_boundary"),
    ({"boundary_residual_px": 30}, "a4_boundary"),
])
def test_detected_shape_must_pass_reference_checks(change, reason):
    image, corners = paper()
    baseline = A4DetectionResult(success=True, corners=corners, confidence=.9,
        visible_corner_count=4, corner_uncertainty_px=1, boundary_residual_px=1)
    assessment = reference_assessment(image, replace(baseline, **change))
    assert not assessment["valid"]
    assert reason in assessment["reasons"]


def test_public_api_applies_the_apps_final_tilt_limit(monkeypatch):
    image, corners = paper()
    initial = A4DetectionResult(success=True, corners=corners, confidence=.9,
        visible_corner_count=4, corner_uncertainty_px=1, boundary_residual_px=1)
    assessment = reference_assessment(image, initial)
    assert assessment["valid"]
    assessment["alignment"]["tilt_deg"] = POLICY["alignmentExitDeg"] + 1
    monkeypatch.setattr("a4_detector.api.detect_a4_paper", lambda *args, **kwargs: initial)
    monkeypatch.setattr("a4_detector.api.detect_product_reference", lambda *args, **kwargs: (initial, assessment))
    result = detect_a4(image)
    assert not result.valid
    assert result.corners is None
    assert "tilt" in result.reasons


def test_cli_writes_json_and_overlay_in_a_unicode_directory(tmp_path):
    directory = tmp_path / "référence"
    directory.mkdir()
    image, _ = paper()
    photo = directory / "paper.png"
    photo.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    output = directory / "result.json"
    overlay = directory / "overlay.png"
    completed = subprocess.run([sys.executable, "-m", "a4_detector", str(photo),
        "--output", str(output), "--overlay", str(overlay)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["valid"]
    decoded = cv2.imdecode(np.frombuffer(overlay.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape
    assert not np.array_equal(decoded, image)


def test_cli_rejection_and_bad_input_exit_codes(tmp_path):
    photo = tmp_path / "blank.png"
    cv2.imwrite(str(photo), np.full((800, 800, 3), 65, np.uint8))
    rejected = subprocess.run([sys.executable, "-m", "a4_detector", str(photo)], capture_output=True, text=True)
    assert rejected.returncode == 1
    assert not json.loads(rejected.stdout)["valid"]
    missing = subprocess.run([sys.executable, "-m", "a4_detector", str(tmp_path / "absent.png")], capture_output=True, text=True)
    assert missing.returncode == 2
    assert "a4-detect:" in missing.stderr
    protected = subprocess.run([sys.executable, "-m", "a4_detector", str(photo), "--output", str(photo)], capture_output=True, text=True)
    assert protected.returncode == 2
    assert photo.read_bytes().startswith(b"\x89PNG")
