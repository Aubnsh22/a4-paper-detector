"""Recover a visible reference when a floor-connected paper mask is unreliable.

The established detector remains the first choice. A fallback isolates sizeable
low-saturation components at several exposures; every replacement must satisfy
the same product reference limits and have strong observed boundary evidence.
"""
from __future__ import annotations

import cv2
import numpy as np

from .a4_detection import (
    A4DetectionResult,
    _boundary_residuals,
    _candidate_blocker,
    _find_candidates,
    _refine_corners_from_image,
    _validate_edge_continuity_strict,
    detect_a4_paper,
)
from .reference import reference_assessment
from .capture_policy import POLICY


def _uniform_paper_rescue(image: np.ndarray, original: A4DetectionResult,
                          reference: dict) -> tuple[A4DetectionResult, dict] | None:
    """Recover a fully observed sheet when patterned floor corrupts the paper mask.

    The contour and all four corners must come from image edges.  This path
    only relaxes agreement with the brightness-derived paper mask, which can
    follow pale checkerboard tiles instead of the sheet.
    """
    area = float(image.shape[0] * image.shape[1])
    plausible = []
    for candidate in original.candidates:
        scores = candidate.score_components
        # The brightness mask can miss a real sheet on a checkerboard or on a
        # blue surface. In that case independent image edges still locate all
        # four sides. Moderate edge support needs an especially uniform,
        # bright interior; a darker sheet needs stronger boundary support.
        edge = scores.get("strict_edge_continuity", 0.0)
        bright_uniform = (
            edge >= 0.75
            and scores.get("interior_brightness", 0.0) >= 0.80
            and scores.get("interior_uniformity", 0.0) >= 0.90
        )
        strong_edge = (
            edge >= 0.92
            and 0.60 <= scores.get("interior_brightness", 0.0) < 0.80
            and scores.get("interior_uniformity", 0.0) >= 0.80
        )
        if not (
            _candidate_blocker(candidate) == "Detected corners do not follow the visible paper boundary."
            and candidate.is_fully_visible and not candidate.projected_corner
            and candidate.visible_corner_count == 4
            and 0.08 <= candidate.area / area <= 0.35
            and candidate.score >= 0.74
            and (bright_uniform or strong_edge)
            and scores.get("rectified_uniformity", 0.0) >= 0.90
            and scores.get("visible_ratio_prior", 0.0) >= 0.98
            and scores.get("rectified_a4_ratio", 0.0) >= 0.98
            and scores.get("internal_line_score", 0.0) >= 0.95
            and candidate.boundary_residual_px <= 3.0
            and candidate.corner_uncertainty_px <= 3.5
        ):
            continue
        plausible.append(candidate)
    if not plausible:
        return None
    plausible.sort(key=lambda candidate: candidate.score, reverse=True)
    best = plausible[0]
    for alternative in plausible[1:]:
        intersection, _ = cv2.intersectConvexConvex(best.corners, alternative.corners)
        smaller = min(best.area, alternative.area)
        if intersection / max(smaller, 1.0) < 0.80:
            return None
    observed = A4DetectionResult(
        success=True,
        corners=_refine_corners_from_image(image, best.corners),
        confidence=best.score,
        message="A4 detected from four visible paper edges.",
        orientation="portrait" if best.side_lengths[1] + best.side_lengths[3]
        >= best.side_lengths[0] + best.side_lengths[2] else "landscape",
        boundary_hypothesis="uniform_edge_boundary",
        boundary_residual_px=best.boundary_residual_px,
        reprojection_residual_px=best.reprojection_residual_px,
        corner_uncertainty_px=best.corner_uncertainty_px,
        visible_corner_count=4,
    )
    checked = reference_assessment(image, observed)
    if not checked["valid"] or checked["alignment"]["tilt_deg"] > POLICY["alignmentExitDeg"]:
        return None
    checked["reference_diagnostics"].update(
        detector="uniform_edge_boundary", original_reasons=reference["reasons"],
        original_detection_message=original.message,
    )
    return observed, checked


def _dominant_edge_rescue(image: np.ndarray, original: A4DetectionResult,
                          reference: dict) -> tuple[A4DetectionResult, dict] | None:
    """Resolve a contour-score tie when only one sheet has visible edges."""
    if original.success:
        return None
    candidates = sorted(original.candidates, key=lambda item: item.score, reverse=True)
    if not candidates:
        return None
    best = candidates[0]
    scores = best.score_components
    if not (
        not _candidate_blocker(best)
        and best.is_fully_visible and not best.projected_corner
        and best.visible_corner_count == 4 and best.score >= 0.75
        and scores.get("strict_edge_continuity", 0.0) >= 0.95
        and scores.get("rectified_uniformity", 0.0) >= 0.80
        and scores.get("visible_ratio_prior", 0.0) >= 0.95
        and best.boundary_residual_px <= 4.0
        and best.corner_uncertainty_px <= 4.0
    ):
        return None
    for alternative in candidates[1:]:
        if (alternative.score >= best.score * 0.92
                and alternative.score_components.get("strict_edge_continuity", 0.0) >= 0.30):
            return None
    observed = A4DetectionResult(
        success=True,
        corners=_refine_corners_from_image(image, best.corners),
        confidence=best.score,
        message="A4 detected from four visible paper edges.",
        orientation="portrait" if best.side_lengths[1] + best.side_lengths[3]
        >= best.side_lengths[0] + best.side_lengths[2] else "landscape",
        boundary_hypothesis="dominant_edge_boundary",
        boundary_residual_px=best.boundary_residual_px,
        reprojection_residual_px=best.reprojection_residual_px,
        corner_uncertainty_px=best.corner_uncertainty_px,
        visible_corner_count=4,
    )
    checked = reference_assessment(image, observed)
    if not checked["valid"] or checked["alignment"]["tilt_deg"] > POLICY["alignmentExitDeg"]:
        return None
    checked["reference_diagnostics"].update(
        detector="dominant_edge_boundary", original_reasons=reference["reasons"],
        original_detection_message=original.message,
    )
    return observed, checked


def _isolated_candidates(binary: np.ndarray, image: np.ndarray, edges: np.ndarray):
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = binary.size
    contours = [contour for contour in contours
                if POLICY["minA4AreaFraction"] * area <= cv2.contourArea(contour)
                <= POLICY["maxA4AreaFraction"] * area]
    if not contours:
        return []
    components = np.zeros_like(binary)
    cv2.drawContours(components, contours, -1, 255, cv2.FILLED)
    candidates, _, _ = _find_candidates(components, image, edges)
    return candidates


def _strong_boundary(candidate, continuity: float | None = None) -> bool:
    scores = candidate.score_components
    edge = scores["strict_edge_continuity"] if continuity is None else continuity
    if candidate.projected_corner or _candidate_blocker(candidate):
        return False
    if (edge >= 0.85 and scores["interior_uniformity"] >= 0.65
            and scores["rectified_uniformity"] >= 0.75
            and scores["visible_ratio_prior"] >= 0.90):
        return True
    # A shadow across an otherwise complete sheet can lower interior
    # uniformity. Accept that shape only with nearly continuous image edges
    # and tight observed corner/boundary fits.
    return (candidate.is_fully_visible and candidate.visible_corner_count == 4
            and edge >= 0.95 and candidate.score >= 0.72
            and scores["interior_uniformity"] >= 0.55
            and scores["rectified_uniformity"] >= 0.75
            and scores["visible_ratio_prior"] >= 0.98
            and candidate.boundary_residual_px <= 4.0
            and candidate.corner_uncertainty_px <= 4.0)


def detect_product_reference(image: np.ndarray, *, initial: A4DetectionResult | None = None,
                             assessment: dict | None = None) -> tuple[A4DetectionResult, dict]:
    original = initial if initial is not None else detect_a4_paper(image, collect_debug=False)
    reference = assessment if assessment is not None else reference_assessment(image, original)
    if reference["valid"]:
        return original, reference

    h, w = image.shape[:2]
    if min(h, w) < POLICY["minImageShortEdgePx"]:
        return original, reference
    recovered = _uniform_paper_rescue(image, original, reference)
    if recovered is not None:
        return recovered
    recovered = _dominant_edge_rescue(image, original, reference)
    if recovered is not None:
        return recovered
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 20, 60)
    # Pale tiles may match the sheet's luminance while differing in colour.
    # Require an observed image edge, including chromatic boundaries.
    colour_edges = cv2.Canny(cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0), 10, 30)
    edges = cv2.bitwise_or(edges, colour_edges)
    seen: list[np.ndarray] = []
    accepted: list[tuple[A4DetectionResult, dict]] = []
    for saturation in (20, 30, 40, 50, 65):
        for brightness in (140, 170, 200):
            binary = ((hsv[:, :, 1] < saturation) & (hsv[:, :, 2] > brightness)).astype(np.uint8) * 255
            for candidate in _isolated_candidates(binary, image, edges):
                if any(np.max(np.linalg.norm(candidate.corners - quad, axis=1)) < 4 for quad in seen):
                    continue
                if not _strong_boundary(candidate):
                    continue
                observed = A4DetectionResult(
                    success=True,
                    corners=_refine_corners_from_image(image, candidate.corners),
                    confidence=candidate.score,
                    message="A4 detected from its visible paper boundary.",
                    orientation="portrait" if candidate.side_lengths[1] + candidate.side_lengths[3]
                    >= candidate.side_lengths[0] + candidate.side_lengths[2] else "landscape",
                    boundary_hypothesis="isolated_color_boundary",
                    boundary_residual_px=candidate.boundary_residual_px,
                    reprojection_residual_px=candidate.reprojection_residual_px,
                    corner_uncertainty_px=candidate.corner_uncertainty_px,
                    visible_corner_count=candidate.visible_corner_count,
                )
                checked = reference_assessment(image, observed)
                if not checked["valid"] or checked["alignment"]["tilt_deg"] > POLICY["alignmentExitDeg"]:
                    continue
                checked["reference_diagnostics"].update(
                    detector="isolated_color_boundary", saturation_limit=saturation,
                    brightness_minimum=brightness, original_reasons=reference["reasons"],
                    original_detection_message=original.message,
                )
                accepted.append((observed, checked))
                # A weak contour must not suppress a later threshold whose
                # similar corners have complete observed boundary support.
                seen.append(candidate.corners.copy())

    if not accepted:
        # Cool paper and warm pale tiles can share both saturation and brightness.
        # Colour contrast may isolate the sheet, but never supplies missing edges.
        difference = image[:, :, 0].astype(np.int16) - image[:, :, 2].astype(np.int16)
        chroma = np.clip(difference + 128, 0, 255).astype(np.uint8)
        edges = edges | cv2.Canny(cv2.GaussianBlur(chroma, (5, 5), 0), 10, 30)
        difference = cv2.GaussianBlur(difference.astype(np.float32), (5, 5), 0)
        for threshold in (0, 4, 8, 12, 16):
            for brightness in (140, 170, 200):
                binary = ((difference > threshold) & (hsv[:, :, 1] < 65)
                          & (hsv[:, :, 2] > brightness)).astype(np.uint8) * 255
                for candidate in _isolated_candidates(binary, image, edges):
                    if any(np.max(np.linalg.norm(candidate.corners - quad, axis=1)) < 4 for quad in seen):
                        continue
                    corners = _refine_corners_from_image(chroma, candidate.corners)
                    continuity = _validate_edge_continuity_strict(edges, corners, None)
                    if not _strong_boundary(candidate, continuity):
                        continue
                    boundary, _, uncertainty = _boundary_residuals(corners, candidate.contour, image=image)
                    observed = A4DetectionResult(
                        success=True, corners=corners, confidence=candidate.score,
                        message="A4 detected from its visible paper boundary.",
                        orientation="portrait" if candidate.side_lengths[1] + candidate.side_lengths[3]
                        >= candidate.side_lengths[0] + candidate.side_lengths[2] else "landscape",
                        boundary_hypothesis="isolated_chromatic_boundary",
                        boundary_residual_px=boundary,
                        reprojection_residual_px=candidate.reprojection_residual_px,
                        corner_uncertainty_px=uncertainty,
                        visible_corner_count=candidate.visible_corner_count,
                    )
                    checked = reference_assessment(image, observed)
                    if not checked["valid"] or checked["alignment"]["tilt_deg"] > POLICY["alignmentExitDeg"]:
                        continue
                    checked["reference_diagnostics"].update(
                        detector="isolated_chromatic_boundary", blue_minus_red_minimum=threshold,
                        brightness_minimum=brightness, original_reasons=reference["reasons"],
                        original_detection_message=original.message,
                    )
                    accepted.append((observed, checked))
                    seen.append(candidate.corners.copy())

    accepted.sort(key=lambda item: item[0].confidence, reverse=True)
    if not accepted:
        return original, reference
    best, checked = accepted[0]
    for alternative, _ in accepted[1:]:
        intersection, _ = cv2.intersectConvexConvex(best.corners, alternative.corners)
        smaller_area = min(abs(cv2.contourArea(best.corners)), abs(cv2.contourArea(alternative.corners)))
        same_sheet = intersection / max(smaller_area, 1) >= 0.8
        if not same_sheet and alternative.confidence >= best.confidence * 0.92:
            # Multiple plausible physical sheets cannot establish which is A4.
            return original, reference
    return best, checked
