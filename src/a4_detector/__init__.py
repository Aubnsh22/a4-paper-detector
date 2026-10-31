"""Detect and validate an A4 reference in a BGR image."""
from .api import ReferenceResult, detect_a4
from .a4_detection import A4DetectionResult, detect_a4_paper, order_corners
from .geometry import compute_homography

__version__ = "0.2.0"
__all__ = [
    "A4DetectionResult",
    "ReferenceResult",
    "compute_homography",
    "detect_a4",
    "detect_a4_paper",
    "order_corners",
]
