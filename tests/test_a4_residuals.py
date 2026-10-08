import cv2
import numpy as np

from a4_detector.a4_detection import (
    _boundary_residuals,
    current_a4_policy,
    detect_a4_paper,
)


def test_boundary_residuals_are_small_for_matching_quad() -> None:
    corners = np.array([[100, 100], [310, 100], [310, 397], [100, 397]], dtype=np.float32)
    contour = corners.reshape(-1, 1, 2)
    boundary, reprojection, uncertainty = _boundary_residuals(corners, contour)
    assert boundary == 0.0
    assert reprojection == 0.0
    assert uncertainty == 0.0


def test_boundary_residuals_increase_for_wrong_corner() -> None:
    corners = np.array([[100, 100], [310, 100], [310, 397], [100, 397]], dtype=np.float32)
    contour = corners.reshape(-1, 1, 2)
    wrong = corners.copy()
    wrong[0] += [40, 35]
    boundary, reprojection, _ = _boundary_residuals(wrong, contour)
    assert boundary > 0.0
    assert reprojection > 0.0


def test_a4_result_exposes_residual_diagnostics() -> None:
    image = np.full((500, 450, 3), 40, dtype=np.uint8)
    corners = np.array([[100, 80], [310, 80], [310, 377], [100, 377]], dtype=np.int32)
    cv2.fillConvexPoly(image, corners, (220, 220, 220))
    result = detect_a4_paper(image)
    assert result.success
    assert result.boundary_hypothesis
    assert result.visible_corner_count in {3, 4}
    assert result.boundary_residual_px is not None
    assert result.corner_uncertainty_px is not None


def test_a4_policy_defaults_to_baseline(monkeypatch) -> None:
    monkeypatch.delenv("FOOT_A4_RESIDUAL_POLICY", raising=False)
    assert current_a4_policy() == "baseline"
