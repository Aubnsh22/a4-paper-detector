"""Application API with the source application's A4 acceptance checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .a4_detection import A4DetectionResult, detect_a4_paper
from .capture_policy import POLICY, messages
from .product_a4 import detect_product_reference


@dataclass
class ReferenceResult:
    """Detection diagnostics and a separate decision about reference validity."""

    detection: A4DetectionResult
    assessment: dict[str, Any]

    @property
    def valid(self) -> bool:
        return bool(self.assessment["valid"])

    @property
    def corners(self) -> np.ndarray | None:
        """Accepted TL, TR, BR, BL coordinates in source-image pixels."""
        return self.detection.corners if self.valid else None

    @property
    def confidence(self) -> float:
        """Candidate ranking score, not a calibrated probability."""
        return float(self.detection.confidence)

    @property
    def reasons(self) -> list[str]:
        return list(self.assessment["reasons"])

    @property
    def messages(self) -> list[str]:
        return list(self.assessment["messages"])

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible result; rejected references have null corners."""
        return {
            "valid": self.valid,
            "corners": self.corners.tolist() if self.corners is not None else None,
            "coordinate_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "coordinate_units": "pixels",
            "image_size_wh": self.assessment["image_size_wh"],
            "confidence": self.confidence,
            "confidence_calibrated": False,
            "orientation": self.detection.orientation if self.detection.success else None,
            "reasons": self.reasons,
            "messages": self.messages,
            "checks": dict(self.assessment["checks"]),
            "policy_version": self.assessment["policy_version"],
            "boundary_hypothesis": self.detection.boundary_hypothesis,
            "boundary_residual_px": self.detection.boundary_residual_px,
            "corner_uncertainty_px": self.detection.corner_uncertainty_px,
            "visible_corner_count": self.detection.visible_corner_count,
            "three_corner_reconstruction": self.detection.three_corner_reconstruction,
            "alignment": self.assessment["alignment"],
        }


def detect_a4(image: np.ndarray, *, collect_debug: bool = False) -> ReferenceResult:
    """Detect a paper reference and apply four-corner, size and pose checks.

    Input must be a nonempty uint8 BGR array shaped (height, width, 3).
    A valid result follows the original application's A4 checks, including
    its 12-degree assumed-focal tilt limit. It does not validate a foot.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 24
    ):
        raise ValueError("Expected a uint8 BGR image with three channels and sides of at least 24 pixels.")
    initial = detect_a4_paper(image, collect_debug=collect_debug)
    detection, assessment = detect_product_reference(image, initial=initial)
    # The original application checks tilt after reference_assessment. Apply
    # the same A4-only gate here so callers have one acceptance decision.
    alignment = assessment.get("alignment")
    if alignment is not None and not alignment["tilt_deg"] <= POLICY["alignmentExitDeg"]:
        assessment["checks"]["geometry"] = False
        assessment["reasons"] = list(dict.fromkeys([*assessment["reasons"], "tilt"]))
        assessment["messages"] = messages(assessment["reasons"])
        assessment["valid"] = False
    if collect_debug and detection is not initial:
        detection.debug_images = initial.debug_images
    return ReferenceResult(detection, assessment)
