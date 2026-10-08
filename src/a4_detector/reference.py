"""A4 reference acceptance checks extracted from the capture application."""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .camera_geometry import valid_quad, visual_alignment
from .capture_policy import POLICY, messages


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (ValueError, TypeError):
        return default



def reference_assessment(image_bgr: np.ndarray, a4: Any) -> dict:
    h, w = image_bgr.shape[:2]
    reasons = []
    checks = {"resolution": min(h, w) >= POLICY["minImageShortEdgePx"], "a4": False, "geometry": False}
    if not checks["resolution"]:
        reasons.append("resolution")
    corners = getattr(a4, "corners", None)
    result = {"policy_version": POLICY["version"], "image_size_wh": [w, h], "checks": checks,
              "corners": None, "alignment": None}
    if not getattr(a4, "success", False) or corners is None or not valid_quad(corners, (h, w)):
        reasons.append("a4_missing")
    else:
        quad = np.asarray(corners, float)
        result["corners"] = (quad / [w, h]).tolist()
        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        min_side = float(min(sides))
        area_fraction = abs(cv2.contourArea(quad.astype(np.float32))) / (h * w)
        margin = min(float(quad.min()), float((w - 1 - quad[:, 0]).min()), float((h - 1 - quad[:, 1]).min()))
        if (getattr(a4, "visible_corner_count", 0) != 4 or getattr(a4, "three_corner_reconstruction", True)
                or margin < min(h, w) * POLICY["frameMarginFraction"]):
            reasons.append("a4_corners")
        if min_side < POLICY["minA4ShortEdgePx"] or area_fraction < POLICY["minA4AreaFraction"]:
            reasons.append("a4_small")
        if area_fraction > POLICY["maxA4AreaFraction"]:
            reasons.append("a4_large")
        if (not finite(getattr(a4, "confidence", None)) >= POLICY["minA4Score"]
                or not finite(getattr(a4, "corner_uncertainty_px", None)) <= POLICY["maxA4CornerUncertaintyPx"]
                or not finite(getattr(a4, "boundary_residual_px", None)) <= POLICY["maxA4BoundaryResidualPx"]):
            reasons.append("a4_boundary")
        checks["a4"] = not any(reason.startswith("a4_") for reason in reasons)
        try:
            alignment = visual_alignment(quad, (h, w))
            result["alignment"] = alignment
            checks["geometry"] = alignment["residual_px"] / max(min_side, 1) <= POLICY["maxPoseResidualFraction"]
            if not checks["geometry"]:
                reasons.append("pose")
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            reasons.append("pose")
        result["reference_diagnostics"] = {"short_side_px": min_side, "area_fraction": area_fraction,
            "score": finite(getattr(a4, "confidence", None), 0), "semantics": "diagnostic_not_probability"}
    result.update(valid=all(checks.values()), reasons=reasons, messages=messages(reasons))
    return result
