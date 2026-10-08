"""A4 dimensions and image-to-plane homography helpers."""
from typing import Optional

import cv2
import numpy as np


A4_WIDTH_MM: float = 210.0



A4_HEIGHT_MM: float = 297.0



A4_REAL_COORDS: np.ndarray = np.array(
    [
        [0.0, 0.0],
        [A4_WIDTH_MM, 0.0],
        [A4_WIDTH_MM, A4_HEIGHT_MM],
        [0.0, A4_HEIGHT_MM],
    ],
    dtype=np.float32,
)



A4_LANDSCAPE_COORDS: np.ndarray = np.array(
    [
        [0.0, 0.0],
        [A4_HEIGHT_MM, 0.0],
        [A4_HEIGHT_MM, A4_WIDTH_MM],
        [0.0, A4_WIDTH_MM],
    ],
    dtype=np.float32,
)



def compute_homography(
    image_corners: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute a 3×3 perspective transformation matrix that maps image-pixel
    coordinates of the four A4 corners to real-world millimetre coordinates.

    Parameters
    ----------
    image_corners : np.ndarray
        4×2 pixel coordinates in order: TL, TR, BR, BL.

    Returns
    -------
    np.ndarray or None
        The 3×3 transformation matrix, or *None* if OpenCV cannot compute it.
    """
    src = np.array(image_corners, dtype=np.float32).reshape(4, 2)
    dst = a4_real_coordinates(src)
    H = cv2.getPerspectiveTransform(src, dst)
    return H



def a4_real_coordinates(image_corners: np.ndarray) -> np.ndarray:
    """Return the correct physical A4 coordinates for an image quadrilateral.

    The previous implementation always mapped the first image edge to
    210 mm.  That is only correct for portrait images.  When a sheet is
    rotated, that same edge represents the 297 mm side and the old mapping
    introduced a large, systematic length error.

    The mean of each opposite side pair is used instead of one side alone so
    mild perspective does not change the orientation decision.
    """
    corners = np.asarray(image_corners, dtype=np.float64).reshape(4, 2)
    edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    image_width = float((edge_lengths[0] + edge_lengths[2]) / 2.0)
    image_height = float((edge_lengths[1] + edge_lengths[3]) / 2.0)

    if image_width > image_height:
        return A4_LANDSCAPE_COORDS.copy()
    return A4_REAL_COORDS.copy()
