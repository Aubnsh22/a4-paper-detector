"""
Automatic A4 paper detection using OpenCV contour analysis.

This module is independent of Streamlit. It detects convex quadrilateral
candidates, rectifies each candidate to an A4-ratio preview, and scores paper
likeness using geometry, homography stability, edge evidence, brightness,
uniformity, and internal-line penalties.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import os


A4_RATIO: float = 297.0 / 210.0
A4_RATIO_TOLERANCE: float = 0.20
MIN_AREA_FRACTION: float = 0.001
MAX_AREA_FRACTION: float = 0.90
BORDER_MARGIN_PX: int = 5
MIN_RECTANGULARITY: float = 0.65
MAX_OPPOSITE_SIDE_RATIO: float = 2.35
MIN_SIDE_FRACTION: float = 0.025
MAX_HOMOGRAPHY_CONDITION: float = 1.0e6
CONFIDENCE_THRESHOLD: float = 0.48
AMBIGUITY_RATIO: float = 0.92
FRONTO_PARALLEL_SIDE_RATIO: float = 1.18
MIN_FRONTO_PARALLEL_A4_PRIOR: float = 0.75
MIN_EDGE_CONTINUITY: float = 0.30
MAX_CORNER_DISTANCE_RATIO: float = 1.8
MAX_CORNER_TO_MASK_DIST_FRAC: float = 0.08
# A foot can legitimately occlude a substantial part of the sheet. Corner,
# contour, aspect-ratio, and homography checks still protect candidate quality.
MIN_INTERIOR_PAPER_FRAC: float = 0.38
MIN_RECTIFIED_UNIFORMITY: float = 0.30
MAX_SIDE_IMBALANCE_RATIO: float = 2.0
MAX_BOUNDARY_RESIDUAL_PX: float = 12.0
MAX_CORNER_UNCERTAINTY_PX: float = 8.0
UNCERTAIN_MESSAGE: str = (
    "A4 detection is uncertain. Retake the photo with the complete paper "
    "visible and the camera closer to above."
)


@dataclass
class A4Candidate:
    """One quadrilateral candidate for the A4 paper."""

    corners: np.ndarray
    contour: np.ndarray
    area: float
    perimeter: float
    rectangularity: float
    aspect_ratio: float
    ratio_similarity: float
    is_convex: bool
    is_fully_visible: bool
    side_lengths: Tuple[float, float, float, float]
    score: float = 0.0
    rejection_reason: str = ""
    score_components: Dict[str, float] = field(default_factory=dict)
    rectified: Optional[np.ndarray] = None
    homography_condition: float = float("inf")
    projected_corner: bool = False
    boundary_hypothesis: str = "unknown"
    boundary_residual_px: float = float("inf")
    reprojection_residual_px: float = float("inf")
    corner_uncertainty_px: float = float("inf")
    visible_corner_count: int = 4


@dataclass
class A4DetectionResult:
    """Full output of the detection pipeline."""

    success: bool = False
    corners: Optional[np.ndarray] = None
    confidence: float = 0.0
    message: str = ""
    orientation: str = "portrait"
    candidates: List[A4Candidate] = field(default_factory=list)
    debug_images: Dict[str, np.ndarray] = field(default_factory=dict)
    boundary_hypothesis: str = "unknown"
    boundary_residual_px: Optional[float] = None
    reprojection_residual_px: Optional[float] = None
    corner_uncertainty_px: Optional[float] = None
    visible_corner_count: Optional[int] = None
    three_corner_reconstruction: bool = False


def current_a4_policy() -> str:
    """Return the reversible residual policy for controlled experiments."""
    return os.getenv("FOOT_A4_RESIDUAL_POLICY", "baseline").strip().lower() or "baseline"


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four 2-D points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _side_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _quad_side_lengths(corners: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        _side_length(corners[0], corners[1]),
        _side_length(corners[1], corners[2]),
        _side_length(corners[2], corners[3]),
        _side_length(corners[3], corners[0]),
    )


def _project_fourth_corner(points: np.ndarray) -> Optional[np.ndarray]:
    """Infer a missing rectangle corner from three observed corners.

    This is an affine approximation, so it is only used as a low-confidence
    fallback for mild perspective. The vertex with the most right-angled
    neighbourhood is treated as the shared corner of the two visible sides.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 3 or not np.all(np.isfinite(pts)):
        return None
    best = None
    for middle in range(3):
        others = [index for index in range(3) if index != middle]
        u = pts[others[0]] - pts[middle]
        v = pts[others[1]] - pts[middle]
        denom = max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-6)
        rightness = abs(float(np.dot(u, v)) / denom)
        candidate = pts[others[0]] + pts[others[1]] - pts[middle]
        score = rightness
        if best is None or score < best[0]:
            best = (score, candidate)
    if best is None or best[0] > 0.45:
        return None
    return best[1].astype(np.float32)


def _project_fourth_corner_from_edges(
    points: np.ndarray,
    edge_image: np.ndarray,
) -> Optional[np.ndarray]:
    """Refine a missing corner by intersecting visible paper-edge lines."""
    pts = np.asarray(points, dtype=np.float32).reshape(3, 2)
    affine = _project_fourth_corner(pts)
    if affine is None or edge_image is None:
        return affine
    best = None
    for middle in range(3):
        others = [index for index in range(3) if index != middle]
        a, b, m = pts[others[0]], pts[others[1]], pts[middle]
        # The missing corner is adjacent to a and b. Search along the two
        # rays from those observed corners toward the affine estimate.
        fitted = []
        yy, xx = np.where(np.asarray(edge_image) > 0)
        pixels = np.column_stack((xx, yy)).astype(np.float32)
        for origin in (a, b):
            target = affine - origin
            target_length = float(np.linalg.norm(target))
            if target_length < 10.0:
                fitted = []
                break
            direction = target / target_length
            offsets = pixels - origin
            projection = offsets @ direction
            perpendicular = np.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
            selected = pixels[(projection >= -12.0) & (projection <= target_length * 1.25) & (perpendicular <= 10.0)]
            if len(selected) < 20:
                fitted = []
                break
            line = np.asarray(cv2.fitLine(selected.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01), dtype=np.float32).reshape(-1)
            fitted.append(line)
        if len(fitted) != 2:
            continue
        d1, o1 = fitted[0][:2], fitted[0][2:4]
        d2, o2 = fitted[1][:2], fitted[1][2:4]
        cross = float(d1[0] * d2[1] - d1[1] * d2[0])
        if abs(cross) < 1e-4:
            continue
        delta = o2 - o1
        t = float((delta[0] * d2[1] - delta[1] * d2[0]) / cross)
        intersection = o1 + t * d1
        if np.linalg.norm(intersection - affine) <= max(target_length * 0.45, 30.0):
            best = intersection.astype(np.float32)
            break
    return best if best is not None else affine


def _estimate_width_height(corners: np.ndarray) -> Tuple[float, float]:
    top, right, bottom, left = _quad_side_lengths(corners)
    return (top + bottom) / 2.0, (left + right) / 2.0


def _refine_corners_subpixel(
    corners: np.ndarray,
    edges: np.ndarray,
    search_radius: int = 12,
) -> np.ndarray:
    """Snap each corner to the strongest nearby edge point for sub-pixel accuracy.

    For each corner, examines a local neighbourhood and shifts it toward the
    densest cluster of edge pixels.  This corrects the typical 2-6 px drift
    introduced by ``approxPolyDP`` polygon simplification.
    """
    refined = corners.astype(np.float32).copy()
    img_h, img_w = edges.shape[:2]
    for i in range(4):
        cx, cy = float(corners[i][0]), float(corners[i][1])
        x0 = max(0, int(cx) - search_radius)
        y0 = max(0, int(cy) - search_radius)
        x1 = min(img_w, int(cx) + search_radius + 1)
        y1 = min(img_h, int(cy) + search_radius + 1)
        patch = edges[y0:y1, x0:x1].astype(np.float32)
        if patch.size == 0 or np.max(patch) == 0:
            continue
        ys, xs = np.where(patch > 0)
        if len(xs) < 3:
            continue
        weights = patch[ys, xs]
        total = float(np.sum(weights))
        if total < 1e-6:
            continue
        new_x = float(np.sum(xs * weights)) / total + x0
        new_y = float(np.sum(ys * weights)) / total + y0
        dist = np.hypot(new_x - cx, new_y - cy)
        if dist <= search_radius:
            refined[i] = [new_x, new_y]
    return refined


def _refine_corners_from_image(
    image: np.ndarray,
    corners: np.ndarray,
    max_shift: float = 6.0,
) -> np.ndarray:
    """Refine a selected paper quadrilateral to sub-pixel corner positions.

    ``approxPolyDP`` and binary masks commonly leave a corner several pixels
    inside the real paper edge.  That error is amplified when the measured
    foot is outside the sheet and the homography is extrapolated.  OpenCV's
    corner refinement uses the original grayscale gradients and is safer than
    averaging every nearby edge pixel (which can include floor texture).

    If the image does not contain a stable corner, or refinement would move a
    point implausibly far, the original detection is retained.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    refined = original.copy()
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    try:
        cv2.cornerSubPix(
            gray,
            refined,
            winSize=(7, 7),
            zeroZone=(-1, -1),
            criteria=criteria,
        )
    except cv2.error:
        return original

    shifts = np.linalg.norm(refined - original, axis=1)
    if not np.all(np.isfinite(refined)) or float(np.max(shifts)) > max_shift:
        return original

    contour = refined.astype(np.float32).reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1.0:
        return original
    return refined


def _refine_quad_from_long_edges(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Replace contour corners that drift onto nearby floor-tile lines.

    The paper can be partly covered by the foot, so its contour may merge with
    a bright floor patch. Long Canny line segments still reveal the visible
    paper edges. Fit one compatible line per side and intersect adjacent
    lines; if the evidence is insufficient, retain the original quadrilateral.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = (
        np.zeros_like(gray)
        if image.ndim == 2
        else cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    )
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 105)
    raw_lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 180.0,
        threshold=55,
        minLineLength=80,
        maxLineGap=28,
    )
    if raw_lines is None:
        return corners

    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    line_data: list[tuple[np.ndarray, float, float, np.ndarray, np.ndarray]] = []
    for segment in raw_lines.reshape(-1, 4):
        p0 = segment[:2].astype(np.float32)
        p1 = segment[2:].astype(np.float32)
        length = float(np.linalg.norm(p1 - p0))
        if length < 80.0:
            continue
        line = np.cross(np.array([p0[0], p0[1], 1.0]), np.array([p1[0], p1[1], 1.0]))
        norm = float(np.hypot(line[0], line[1]))
        if norm < 1e-6:
            continue
        line = line / norm
        angle = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        line_data.append((line, angle, length, p0, p1))

    selected: list[np.ndarray] = []
    for index in range(4):
        start = pts[index]
        end = pts[(index + 1) % 4]
        expected = end - start
        expected_length = float(np.linalg.norm(expected))
        if expected_length < 1e-6:
            selected.append(np.cross(np.r_[start, 1.0], np.r_[end, 1.0]))
            continue
        expected_angle = float(np.arctan2(expected[1], expected[0]))
        expected_line = np.cross(np.r_[start, 1.0], np.r_[end, 1.0]).astype(np.float64)
        expected_line /= max(float(np.hypot(expected_line[0], expected_line[1])), 1e-6)
        midpoint = (start + end) * 0.5
        max_distance = max(90.0, expected_length * 0.35)
        options: list[tuple[float, np.ndarray]] = []
        quad_center = pts.mean(axis=0)
        for line, angle, length, p0, p1 in line_data:
            angle_delta = abs((angle - expected_angle + np.pi / 2.0) % np.pi - np.pi / 2.0)
            if angle_delta > np.deg2rad(14.0) or length < expected_length * 0.22:
                continue
            distance = abs(float(line @ np.r_[midpoint, 1.0]))
            if distance > max_distance:
                continue
            # A true paper edge should have a bright, low-saturation region
            # on its inward side. Tile/grout lines usually have similar floor
            # appearance on both sides even when they are longer.
            inward_sign = 1.0 if float(line @ np.r_[quad_center, 1.0]) >= 0.0 else -1.0
            inward = inward_sign * line[:2]
            samples = p0[None, :] + np.linspace(0.12, 0.88, 24)[:, None] * (p1 - p0)[None, :]
            inside = np.rint(samples + inward[None, :] * 12.0).astype(int)
            outside = np.rint(samples - inward[None, :] * 12.0).astype(int)
            valid = (
                (inside[:, 0] >= 0) & (inside[:, 0] < gray.shape[1])
                & (inside[:, 1] >= 0) & (inside[:, 1] < gray.shape[0])
                & (outside[:, 0] >= 0) & (outside[:, 0] < gray.shape[1])
                & (outside[:, 1] >= 0) & (outside[:, 1] < gray.shape[0])
            )
            if np.any(valid):
                ix, iy = inside[valid, 0], inside[valid, 1]
                ox, oy = outside[valid, 0], outside[valid, 1]
                inside_brightness = float(np.mean(gray[iy, ix]))
                outside_brightness = float(np.mean(gray[oy, ox]))
                inside_saturation = float(np.mean(saturation[iy, ix]))
                contrast = float(
                    np.clip((inside_brightness - outside_brightness - 5.0) / 50.0, 0.0, 1.0)
                )
                whiteness = float(np.clip(1.0 - inside_saturation / 100.0, 0.0, 1.0))
                paper_evidence = 0.80 * contrast + 0.20 * whiteness
            else:
                paper_evidence = 0.0
            score = (
                length
                * np.exp(-0.75 * distance / max(max_distance, 1.0))
                * (0.10 + 2.50 * paper_evidence)
            )
            options.append((score, line))
        if options:
            selected.append(max(options, key=lambda item: item[0])[1])
        else:
            selected.append(expected_line)

    intersections: list[np.ndarray] = []
    for index in range(4):
        point = np.cross(selected[(index - 1) % 4], selected[index])
        if abs(float(point[2])) < 1e-6:
            return pts
        intersections.append((point[:2] / point[2]).astype(np.float32))
    refined = order_corners(np.asarray(intersections, dtype=np.float32))
    if not np.all(np.isfinite(refined)) or not cv2.isContourConvex(refined.reshape(-1, 1, 2)):
        return pts
    if _quad_area(refined) < _quad_area(pts) * 0.45:
        return pts
    return refined


def _quad_area(corners: np.ndarray) -> float:
    area = 0.0
    for i in range(len(corners)):
        j = (i + 1) % len(corners)
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def _is_fully_visible(
    corners: np.ndarray,
    img_h: int,
    img_w: int,
    margin: int = BORDER_MARGIN_PX,
) -> bool:
    for pt in corners:
        if pt[0] < margin or pt[1] < margin:
            return False
        if pt[0] > img_w - margin or pt[1] > img_h - margin:
            return False
    return True


def _polygon_mask(shape: Tuple[int, int], corners: np.ndarray, scale: float = 1.0) -> np.ndarray:
    pts = corners.astype(np.float32)
    if scale != 1.0:
        center = pts.mean(axis=0)
        pts = center + (pts - center) * scale
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
    return mask


def _ratio_similarity(ratio: float, tolerance: float) -> float:
    ratio = max(ratio, 1.0 / max(ratio, 1e-6))
    deviation = abs(ratio - A4_RATIO) / A4_RATIO
    return float(max(0.0, 1.0 - deviation / tolerance))


# ---------------------------------------------------------------------------
# White-paper-first detection
# ---------------------------------------------------------------------------


def _detect_paper_region(
    image: np.ndarray,
    collect_debug: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Detect the bright, low-saturation paper region as a mask and contour.

    Returns (paper_mask, largest_contour, debug_images) or (None, None, {}) if
    no suitable paper region is found.
    """
    debug: Dict[str, np.ndarray] = {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_v = float(np.mean(v))
    mean_s = float(np.mean(s))
    v_thresh = max(int(min(mean_v + 40, 255) * 0.65), 120)
    s_thresh = min(int(max(mean_s + 20, 30) * 1.2), 80)

    bright = (v > v_thresh).astype(np.uint8) * 255
    low_sat = (s < s_thresh).astype(np.uint8) * 255
    combined = cv2.bitwise_and(bright, low_sat)

    # On checkerboard/tiled floors, the adaptive threshold can connect bright
    # floor patches to the sheet and pull a corner toward a grout intersection.
    # A stricter white-paper pass is safer when it produces a plausible A4
    # quadrilateral; the adaptive pass remains the fallback for dark or tinted
    # paper.
    strict_v_thresh = max(v_thresh, 170)
    strict_s_thresh = min(s_thresh, 65)
    strict = ((v > strict_v_thresh) & (s < strict_s_thresh)).astype(np.uint8) * 255
    strict = cv2.morphologyEx(
        strict,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    strict = cv2.morphologyEx(
        strict,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    strict_labels, strict_stats, _ = cv2.connectedComponentsWithStats(strict, connectivity=8)[1:]
    strict_components = sorted(
        range(1, len(strict_stats)),
        key=lambda label: int(strict_stats[label, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    image_area = float(image.shape[0] * image.shape[1])
    for label in strict_components:
        component_area = int(strict_stats[label, cv2.CC_STAT_AREA])
        if component_area / image_area < max(MIN_AREA_FRACTION, 0.01):
            break
        component = ((strict_labels == label) * 255).astype(np.uint8)
        component_contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not component_contours:
            continue
        component_contour = max(component_contours, key=cv2.contourArea)
        component_corners = _derive_corners_from_paper_mask(component, component_contour)
        if component_corners is None or not _validate_corner_distances(component_corners):
            continue
        component_corners = _refine_quad_from_long_edges(image, component_corners)
        sides = _quad_side_lengths(component_corners)
        width_est, height_est = _estimate_width_height(component_corners)
        aspect = max(width_est, height_est) / max(min(width_est, height_est), 1.0)
        if max(sides) / max(min(sides), 1.0) <= MAX_OPPOSITE_SIDE_RATIO and _ratio_similarity(
            aspect, tolerance=0.90
        ) >= 0.55:
            if collect_debug:
                debug["strict_paper_region_mask"] = component.copy()
            return component, component_contour, debug

    # A foot and its shadow can split a white sheet into several disconnected
    # bright regions. Reassemble only sizeable, nearby components around the
    # dominant non-border region, then fit the outer oriented rectangle. This
    # preserves the real paper boundary even when its middle is occluded.
    merge_primary = None
    for label in strict_components:
        x, y, width, height, area = strict_stats[label].tolist()
        touches_border = (
            x <= BORDER_MARGIN_PX
            or y <= BORDER_MARGIN_PX
            or x + width >= image.shape[1] - BORDER_MARGIN_PX
            or y + height >= image.shape[0] - BORDER_MARGIN_PX
        )
        if not touches_border and min(width, height) >= 20:
            merge_primary = label
            break
    if merge_primary is not None:
        px, py, pw, ph, parea = strict_stats[merge_primary].tolist()
        primary_center = np.array([px + pw * 0.5, py + ph * 0.5], dtype=np.float32)
        selected_labels = [merge_primary]
        for label in strict_components:
            if label == merge_primary:
                continue
            x, y, width, height, area = strict_stats[label].tolist()
            if area < max(200.0, parea * 0.04):
                continue
            if width > pw * 2.5 or height > ph * 1.15:
                continue
            center = np.array([x + width * 0.5, y + height * 0.5], dtype=np.float32)
            if abs(float(center[0] - primary_center[0])) > pw * 1.45:
                continue
            if abs(float(center[1] - primary_center[1])) > ph * 0.60:
                continue
            if (
                x <= BORDER_MARGIN_PX
                or y <= BORDER_MARGIN_PX
                or x + width >= image.shape[1] - BORDER_MARGIN_PX
                or y + height >= image.shape[0] - BORDER_MARGIN_PX
            ):
                continue
            selected_labels.append(label)

        if len(selected_labels) >= 2:
            points = np.vstack(
                [
                    np.column_stack(np.where(strict_labels == label)[::-1])
                    for label in selected_labels
                ]
            ).astype(np.float32)
            oriented = cv2.minAreaRect(points)
            rw, rh = oriented[1]
            if min(rw, rh) >= 20.0:
                aspect = max(rw, rh) / max(min(rw, rh), 1.0)
                if _ratio_similarity(aspect, tolerance=0.90) >= 0.55:
                    merged_corners = order_corners(cv2.boxPoints(oriented))
                    merged_mask = np.zeros_like(strict)
                    cv2.fillConvexPoly(
                        merged_mask,
                        np.rint(merged_corners).astype(np.int32),
                        255,
                    )
                    merged_contours, _ = cv2.findContours(
                        merged_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if merged_contours:
                        if collect_debug:
                            debug["merged_paper_region_mask"] = merged_mask.copy()
                        return (
                            merged_mask,
                            max(merged_contours, key=cv2.contourArea),
                            debug,
                        )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    if collect_debug:
        debug["paper_region_mask"] = cleaned.copy()

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8,
    )
    img_area = float(image.shape[0] * image.shape[1])

    candidates_info: List[Tuple[int, float, float]] = []
    for lbl in range(1, n_labels):
        comp_area = float(stats[lbl, cv2.CC_STAT_AREA])
        comp_frac = comp_area / img_area
        if comp_frac < MIN_AREA_FRACTION or comp_frac > MAX_AREA_FRACTION:
            continue
        cw = float(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = float(stats[lbl, cv2.CC_STAT_HEIGHT])
        if min(cw, ch) < 20:
            continue
        candidates_info.append((lbl, comp_area, comp_frac))

    if not candidates_info:
        return None, None, debug

    candidates_info.sort(key=lambda x: x[1], reverse=True)
    best_lbl = candidates_info[0][0]
    paper_mask = ((labels == best_lbl) * 255).astype(np.uint8)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, kernel3)

    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, debug

    largest_contour = max(contours, key=cv2.contourArea)
    if collect_debug:
        debug["paper_region_largest"] = cv2.cvtColor(paper_mask, cv2.COLOR_GRAY2BGR)

    return paper_mask, largest_contour, debug


def _derive_corners_from_paper_mask(
    paper_mask: np.ndarray,
    contour: np.ndarray,
) -> Optional[np.ndarray]:
    """Derive 4 A4 corners from the paper region mask contour."""
    peri = cv2.arcLength(contour, True)
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            if cv2.isContourConvex(approx):
                return order_corners(approx.reshape(4, 2))

    hull = cv2.convexHull(contour)
    hull_peri = cv2.arcLength(hull, True)
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        approx = cv2.approxPolyDP(hull, eps * hull_peri, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))

    return None


def _validate_edge_continuity_strict(
    edges: np.ndarray,
    corners: np.ndarray,
    mask: Optional[np.ndarray],
) -> float:
    """Strict edge continuity: require strong edges along each quad side.

    Each side must have edges along at least MIN_EDGE_CONTINUITY fraction of
    its length.  If a paper mask is provided, edges are further gated to only
    count if they lie on or near the mask boundary.
    """
    per_side_scores: List[float] = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        length = max(int(_side_length(a, b)), 1)
        hits = 0
        for t in np.linspace(0.0, 1.0, length):
            p = a * (1.0 - t) + b * t
            x, y = int(round(p[0])), int(round(p[1]))
            y0 = max(0, y - 2)
            y1 = min(edges.shape[0], y + 3)
            x0 = max(0, x - 2)
            x1 = min(edges.shape[1], x + 3)
            if y0 >= y1 or x0 >= x1:
                continue
            patch = edges[y0:y1, x0:x1]
            if np.any(patch > 0):
                hits += 1
        per_side_scores.append(hits / length)

    mean_score = float(np.mean(per_side_scores))
    min_side_score = float(np.min(per_side_scores))
    if min_side_score < MIN_EDGE_CONTINUITY * 0.5:
        return min_side_score
    return mean_score


def _validate_corner_distances(corners: np.ndarray) -> bool:
    """Reject if one corner is much farther from the centroid than the others."""
    center = corners.mean(axis=0)
    dists = np.linalg.norm(corners - center, axis=1)
    median_dist = float(np.median(dists))
    if median_dist < 1.0:
        return True
    max_ratio = float(np.max(dists)) / median_dist
    return max_ratio <= MAX_CORNER_DISTANCE_RATIO


def _validate_corner_to_mask_boundary(
    corners: np.ndarray,
    paper_mask: np.ndarray,
) -> Tuple[bool, List[float]]:
    """Check each corner is close to the paper mask boundary.

    Returns (all_ok, per_corner_distances).
    """
    mask_dilated = cv2.dilate(
        paper_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    dist_transform = cv2.distanceTransform(
        (255 - mask_dilated), cv2.DIST_L2, 5,
    )
    img_diag = float(np.hypot(paper_mask.shape[1], paper_mask.shape[0]))
    max_dist = MAX_CORNER_TO_MASK_DIST_FRAC * img_diag

    distances: List[float] = []
    for pt in corners:
        x = int(np.clip(round(pt[0]), 0, paper_mask.shape[1] - 1))
        y = int(np.clip(round(pt[1]), 0, paper_mask.shape[0] - 1))
        distances.append(float(dist_transform[y, x]))

    return all(d <= max_dist for d in distances), distances


def _validate_interior_paper_fraction(
    corners: np.ndarray,
    paper_mask: np.ndarray,
) -> float:
    """Fraction of the quad interior that overlaps the paper mask."""
    interior_mask = _polygon_mask(paper_mask.shape, corners, scale=0.90)
    intersection = cv2.bitwise_and(interior_mask, paper_mask)
    interior_area = float(np.count_nonzero(interior_mask))
    if interior_area < 1.0:
        return 0.0
    return float(np.count_nonzero(intersection)) / interior_area


def _validate_rectified_uniformity(rectified: np.ndarray) -> float:
    """Score how paper-like the rectified image is (bright, uniform, low-sat)."""
    if rectified is None or rectified.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    v_mean = float(np.mean(v_ch))
    s_mean = float(np.mean(s_ch))
    g_std = float(np.std(gray))
    brightness_score = float(np.clip(v_mean / 255.0, 0.0, 1.0))
    saturation_score = float(np.clip(1.0 - s_mean / 128.0, 0.0, 1.0))
    uniformity_score = float(np.clip(1.0 - g_std / 55.0, 0.0, 1.0))
    return 0.4 * brightness_score + 0.3 * saturation_score + 0.3 * uniformity_score


def _validate_same_contour_origin(
    corners: np.ndarray,
    contours: List[np.ndarray],
) -> bool:
    """All 4 corners must lie on or very close to the same contour."""
    if not contours:
        return False
    best_count = 0
    for cnt in contours:
        cnt_pts = cnt.reshape(-1, 2).astype(np.float32)
        hits = 0
        for c in corners:
            dists = np.linalg.norm(cnt_pts - c.reshape(1, 2), axis=1)
            if np.min(dists) < 8.0:
                hits += 1
        best_count = max(best_count, hits)
    return best_count >= 3


def _per_corner_mask_distance(
    corners: np.ndarray,
    paper_mask: np.ndarray,
) -> List[float]:
    """Distance of each corner from the paper mask boundary."""
    mask_dilated = cv2.dilate(
        paper_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    dist_transform = cv2.distanceTransform(
        (255 - mask_dilated), cv2.DIST_L2, 5,
    )
    distances: List[float] = []
    for pt in corners:
        x = int(np.clip(round(pt[0]), 0, paper_mask.shape[1] - 1))
        y = int(np.clip(round(pt[1]), 0, paper_mask.shape[0] - 1))
        distances.append(float(dist_transform[y, x]))
    return distances


def _rectified_paper_score(rectified: np.ndarray) -> float:
    """Post-rectification check: most of the image should be uniform paper."""
    if rectified is None or rectified.size == 0:
        return 0.0
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    bright = (v > int(max(float(np.mean(v)) * 0.6, 100))).astype(np.uint8) * 255
    total = float(bright.shape[0] * bright.shape[1])
    if total < 1.0:
        return 0.0
    bright_frac = float(np.count_nonzero(bright)) / total
    std = float(np.std(gray))
    uniformity = float(np.clip(1.0 - std / 60.0, 0.0, 1.0))
    return 0.5 * bright_frac + 0.5 * uniformity


def _preprocess(
    image: np.ndarray,
    collect_debug: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    debug: Dict[str, np.ndarray] = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if collect_debug:
        debug["grayscale"] = gray

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if collect_debug:
        debug["blurred"] = blurred

    median = float(np.median(blurred))
    low = max(int(0.5 * median), 30)
    high = max(min(int(1.5 * median), 255), low + 40)
    edges = cv2.Canny(blurred, low, high)
    if collect_debug:
        debug["canny"] = edges

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if collect_debug:
        debug["morphological"] = closed
    return closed, debug


def _threshold_contour_images(
    image: np.ndarray,
    collect_debug: bool = False,
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    debug: Dict[str, np.ndarray] = {}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    masks = [
        cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
        cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel),
    ]

    if collect_debug:
        debug["otsu_light_foreground"] = masks[0]
        debug["otsu_dark_foreground"] = masks[1]
    return masks, debug


def _rectify_candidate(
    image: np.ndarray,
    corners: np.ndarray,
    orientation: str,
    scale: int = 2,
) -> Tuple[Optional[np.ndarray], float, float]:
    if orientation == "portrait":
        dst_w, dst_h = 210 * scale, 297 * scale
    else:
        dst_w, dst_h = 297 * scale, 210 * scale

    dst_pts = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_pts)
    if H is None or not np.all(np.isfinite(H)):
        return None, float("inf"), 0.0

    condition = float(np.linalg.cond(H))
    if not np.isfinite(condition):
        return None, float("inf"), 0.0

    rectified = cv2.warpPerspective(image, H, (dst_w, dst_h))
    return rectified, condition, max(dst_w, dst_h) / min(dst_w, dst_h)


def _edge_continuity(edges: np.ndarray, corners: np.ndarray) -> float:
    scores: List[float] = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        length = max(int(_side_length(a, b)), 1)
        hits = 0
        for t in np.linspace(0.0, 1.0, length):
            p = a * (1.0 - t) + b * t
            x, y = int(round(p[0])), int(round(p[1]))
            y0, y1 = max(0, y - 2), min(edges.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(edges.shape[1], x + 3)
            if np.any(edges[y0:y1, x0:x1] > 0):
                hits += 1
        scores.append(hits / length)
    return float(np.mean(scores))


def _point_to_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    denominator = max(float(np.dot(segment, segment)), 1e-6)
    relative = np.asarray(points, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    projection = np.clip((relative @ segment) / denominator, 0.0, 1.0)
    closest = np.asarray(start, dtype=np.float32) + projection[:, None] * segment
    return np.linalg.norm(np.asarray(points, dtype=np.float32) - closest, axis=1)


def _boundary_residuals(
    corners: np.ndarray,
    contour: np.ndarray,
    *,
    image: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate contour-to-quad, rectified-edge, and corner uncertainty.

    The values are diagnostics, not calibrated probabilities.  A contour
    point is compared with the nearest physical quad edge; the same points are
    then projected into the ideal A4 rectangle to expose perspective/edge
    disagreement.  Corner uncertainty is the local edge-centroid movement.
    """
    quad = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return float("inf"), float("inf"), float("inf")
    distances = np.column_stack(
        [_point_to_segment_distance(points, quad[i], quad[(i + 1) % 4]) for i in range(4)]
    )
    nearest = np.min(distances, axis=1)
    boundary_residual = float(np.percentile(nearest, 90))
    try:
        orientation = "portrait" if _estimate_width_height(quad)[1] >= _estimate_width_height(quad)[0] else "landscape"
        destination = np.array(
            [[0, 0], [209, 0], [209, 296], [0, 296]] if orientation == "portrait" else [[0, 0], [296, 0], [296, 209], [0, 209]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(quad, destination)
        projected = cv2.perspectiveTransform(points[None, :, :], homography)[0]
        expected = np.column_stack(
            [
                np.minimum(np.abs(projected[:, 0]), np.abs(projected[:, 0] - destination[1, 0])),
                np.minimum(np.abs(projected[:, 1]), np.abs(projected[:, 1] - destination[2, 1])),
            ]
        )
        reprojection_residual = float(np.percentile(np.min(expected, axis=1), 90))
    except cv2.error:
        reprojection_residual = float("inf")

    corner_uncertainty = 0.0
    if image is not None:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
        refined = _refine_corners_subpixel(quad, edges)
        corner_uncertainty = float(np.mean(np.linalg.norm(refined - quad, axis=1)))
    return boundary_residual, reprojection_residual, corner_uncertainty


def _appearance_scores(
    image: np.ndarray,
    edges: np.ndarray,
    corners: np.ndarray,
) -> Dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inner_mask = _polygon_mask(gray.shape, corners, scale=0.88)
    full_mask = _polygon_mask(gray.shape, corners, scale=1.0)
    outer_mask = _polygon_mask(gray.shape, corners, scale=1.18)
    surrounding_mask = cv2.subtract(outer_mask, full_mask)

    inside = gray[inner_mask > 0]
    outside = gray[surrounding_mask > 0]
    inside_mean = float(np.mean(inside)) if inside.size else 0.0
    inside_std = float(np.std(inside)) if inside.size else 255.0
    outside_std = float(np.std(outside)) if outside.size else inside_std

    brightness = inside_mean / 255.0
    absolute_uniformity = max(0.0, 1.0 - inside_std / 55.0)
    relative_uniformity = float(np.clip((outside_std - inside_std + 20.0) / 60.0, 0.0, 1.0))
    internal_edge_density = float(np.mean(edges[inner_mask > 0] > 0)) if inside.size else 1.0

    return {
        "edge_continuity": _edge_continuity(edges, corners),
        "interior_brightness": float(brightness),
        "interior_uniformity": float(0.65 * absolute_uniformity + 0.35 * relative_uniformity),
        "internal_line_score": float(max(0.0, 1.0 - internal_edge_density / 0.08)),
    }


def _find_candidates(
    contour_image: np.ndarray,
    image: np.ndarray,
    edge_image: np.ndarray,
    collect_debug: bool = False,
    debug_prefix: str = "",
) -> Tuple[List[A4Candidate], Dict[str, np.ndarray], List[str]]:
    img_h, img_w = image.shape[:2]
    image_area = float(img_h * img_w)
    image_diag = float(np.hypot(img_w, img_h))
    debug: Dict[str, np.ndarray] = {}

    contours, _ = cv2.findContours(contour_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if collect_debug:
        dbg_all = image.copy()
        cv2.drawContours(dbg_all, contours[:40], -1, (0, 255, 0), 2)
        debug[f"{debug_prefix}all_contours"] = dbg_all

    candidates: List[A4Candidate] = []
    reject_reasons: List[str] = []

    for i, cnt in enumerate(contours):
        cnt_area = cv2.contourArea(cnt)
        frac = cnt_area / image_area
        if frac < MIN_AREA_FRACTION or frac > MAX_AREA_FRACTION:
            if i < 10:
                reject_reasons.append(f"{debug_prefix}contour {i}: area {frac:.3f} outside bounds")
            continue

        peri = cv2.arcLength(cnt, True)
        approx = None
        projected_corner = False
        last_len = 0
        for eps_factor in [0.008, 0.012, 0.016, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15]:
            temp_approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
            last_len = len(temp_approx)
            if last_len == 4:
                approx = temp_approx
                break

        if approx is None:
            # If one sheet corner is occluded, the visible paper contour may
            # simplify to a triangle. Infer the fourth corner only from a
            # strong three-point rectangle and treat it as lower confidence.
            triangle = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(triangle) == 3:
                triangle_points = triangle.reshape(3, 2)
                inferred = _project_fourth_corner_from_edges(triangle_points, edge_image)
                if inferred is not None:
                    approx = np.vstack(
                        [triangle.reshape(3, 2), inferred.reshape(1, 2)]
                    ).astype(np.float32).reshape(4, 1, 2)
                    projected_corner = True
            if approx is None:
                if i < 10:
                    reject_reasons.append(f"{debug_prefix}contour {i}: not four-corner ({last_len})")
                continue

        if not cv2.isContourConvex(approx):
            hull = cv2.convexHull(approx)
            if len(hull) == 4:
                approx = hull
            else:
                if i < 10:
                    reject_reasons.append(f"{debug_prefix}contour {i}: not convex")
                continue

        corners = order_corners(approx.reshape(4, 2))
        sides = _quad_side_lengths(corners)
        width_est, height_est = _estimate_width_height(corners)
        min_side = min(sides)
        if min(width_est, height_est) < 1 or min_side / image_diag < MIN_SIDE_FRACTION:
            if i < 10:
                reject_reasons.append(f"{debug_prefix}contour {i}: edge nearly collapsed")
            continue

        side_ratio = max(sides) / max(min_side, 1.0)
        if side_ratio > MAX_OPPOSITE_SIDE_RATIO:
            if i < 10:
                reject_reasons.append(f"{debug_prefix}contour {i}: extreme side ratio {side_ratio:.2f}")
            continue

        quad_a = _quad_area(corners)
        (_, _), (rw, rh), _ = cv2.minAreaRect(approx)
        rect_area = float(rw * rh) if rw * rh > 0 else 1.0
        rectangularity = min(quad_a / rect_area, 1.0)

        longer = max(width_est, height_est)
        shorter = min(width_est, height_est)
        aspect = longer / shorter
        visible_ratio_prior = _ratio_similarity(aspect, tolerance=0.90)
        orientation = "portrait" if height_est >= width_est else "landscape"

        rectified, homography_condition, rectified_ratio = _rectify_candidate(
            image, corners, orientation,
        )
        if rectified is None or homography_condition > MAX_HOMOGRAPHY_CONDITION:
            if i < 10:
                reject_reasons.append(f"{debug_prefix}contour {i}: unstable homography")
            continue

        rectified_a4_ratio = _ratio_similarity(rectified_ratio, tolerance=A4_RATIO_TOLERANCE)
        appearance = _appearance_scores(image, edge_image, corners)
        boundary_residual, reprojection_residual, corner_uncertainty = _boundary_residuals(
            corners,
            cnt,
            image=image,
        )

        strict_ec = _validate_edge_continuity_strict(edge_image, corners, None)

        rect_score = _rectified_paper_score(rectified)

        candidate = A4Candidate(
            corners=corners,
            contour=cnt,
            area=quad_a,
            perimeter=peri,
            rectangularity=rectangularity,
            aspect_ratio=aspect,
            ratio_similarity=rectified_a4_ratio,
            is_convex=True,
            is_fully_visible=_is_fully_visible(corners, img_h, img_w),
            side_lengths=sides,
            rectified=rectified,
            homography_condition=homography_condition,
            projected_corner=projected_corner,
            boundary_hypothesis=debug_prefix.rstrip("_") or "edge_contour",
            boundary_residual_px=boundary_residual,
            reprojection_residual_px=reprojection_residual,
            corner_uncertainty_px=corner_uncertainty,
            visible_corner_count=3 if projected_corner else 4,
        )
        candidate.score_components = {
            "area": min((quad_a / image_area) / 0.10, 1.0),
            "convexity": 1.0,
            "edge_continuity": appearance["edge_continuity"],
            "strict_edge_continuity": strict_ec,
            "interior_brightness": appearance["interior_brightness"],
            "interior_uniformity": appearance["interior_uniformity"],
            "homography_stability": max(0.0, 1.0 - np.log10(max(homography_condition, 1.0)) / 6.0),
            "rectified_a4_ratio": rectified_a4_ratio,
            "distance_from_borders": 1.0 if candidate.is_fully_visible else 0.0,
            "visible_ratio_prior": visible_ratio_prior,
            "internal_line_score": appearance["internal_line_score"],
            "interior_paper_fraction": 0.0,
            "rectified_uniformity": rect_score,
            "corner_to_mask_dist_ok": True,
            "projected_corner_prior": 0.55 if projected_corner else 1.0,
            "boundary_residual_score": float(np.clip(1.0 - boundary_residual / MAX_BOUNDARY_RESIDUAL_PX, 0.0, 1.0)),
            "corner_uncertainty_score": float(np.clip(1.0 - corner_uncertainty / MAX_CORNER_UNCERTAINTY_PX, 0.0, 1.0)),
        }
        candidate.score = _score_candidate(candidate, image_area)
        if projected_corner:
            candidate.score *= 0.78
        candidates.append(candidate)

    if collect_debug:
        dbg_candidates = image.copy()
        for idx, c in enumerate(sorted(candidates, key=lambda x: x.score, reverse=True)):
            pts = c.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(dbg_candidates, [pts], True, (255, 0, 255), 2)
            cv2.putText(
                dbg_candidates,
                f"{idx}:{c.score:.2f}",
                tuple(c.corners[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        debug[f"{debug_prefix}all_candidates_scored"] = dbg_candidates

    return candidates, debug, reject_reasons


def _dedupe_candidates(candidates: List[A4Candidate]) -> List[A4Candidate]:
    unique: List[A4Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        duplicate = False
        for kept in unique:
            center_distance = float(
                np.linalg.norm(candidate.corners.mean(axis=0) - kept.corners.mean(axis=0))
            )
            area_ratio = min(candidate.area, kept.area) / max(candidate.area, kept.area)
            corner_distance = float(
                np.mean(np.linalg.norm(candidate.corners - kept.corners, axis=1))
            )
            if center_distance < 20.0 and area_ratio > 0.85 and corner_distance < 30.0:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def _score_candidate(candidate: A4Candidate, image_area: float) -> float:
    c = candidate.score_components
    score = (
        0.12 * c.get("area", 0.0)
        + 0.05 * c.get("convexity", 0.0)
        + 0.12 * c.get("edge_continuity", 0.0)
        + 0.11 * c.get("interior_brightness", 0.0)
        + 0.11 * c.get("interior_uniformity", 0.0)
        + 0.12 * c.get("homography_stability", 0.0)
        + 0.12 * c.get("rectified_a4_ratio", 0.0)
        + 0.06 * c.get("distance_from_borders", 0.0)
        + 0.04 * c.get("visible_ratio_prior", 0.0)
        + 0.08 * c.get("internal_line_score", 0.0)
        + 0.07 * c.get("interior_paper_fraction", 0.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def _is_nearly_fronto_parallel(candidate: A4Candidate) -> bool:
    top, right, bottom, left = candidate.side_lengths
    horizontal_ratio = max(top, bottom) / max(min(top, bottom), 1.0)
    vertical_ratio = max(left, right) / max(min(left, right), 1.0)
    return horizontal_ratio < FRONTO_PARALLEL_SIDE_RATIO and vertical_ratio < FRONTO_PARALLEL_SIDE_RATIO


def _candidate_blocker(candidate: A4Candidate) -> str:
    if not candidate.is_fully_visible:
        return "The complete A4 paper must be visible."

    min_side = min(candidate.side_lengths)
    if min_side <= 0:
        return "Camera angle is too extreme. Retake the photo from above."

    if max(candidate.side_lengths) / max(min_side, 1.0) > MAX_OPPOSITE_SIDE_RATIO:
        return "Camera angle is too extreme. Retake the photo from above."

    if candidate.homography_condition > MAX_HOMOGRAPHY_CONDITION:
        return "Camera angle is too extreme. Retake the photo from above."

    if current_a4_policy() == "strict_v1":
        if candidate.boundary_residual_px > MAX_BOUNDARY_RESIDUAL_PX:
            return "A4 boundary support is inconsistent. Retake the photo with the physical paper edge visible."
        if candidate.corner_uncertainty_px > MAX_CORNER_UNCERTAINTY_PX:
            return "A4 corner localization is uncertain. Retake the photo with all corners in focus."
        # A projected corner is valid only as an explicitly lower-confidence
        # three-corner reconstruction; strict mode never silently treats it as
        # equivalent to four observed physical corners.
        if candidate.projected_corner:
            return "Only three A4 corners are visible; review or provide a complete reference sheet."

    side_imbalance = max(candidate.side_lengths) / max(min(candidate.side_lengths), 1.0)
    if side_imbalance > MAX_SIDE_IMBALANCE_RATIO:
        return "Detected shape is not a valid A4 rectangle."

    if (
        _is_nearly_fronto_parallel(candidate)
        and candidate.score_components.get("visible_ratio_prior", 0.0) < MIN_FRONTO_PARALLEL_A4_PRIOR
    ):
        return "The detected paper does not appear to be A4."

    if not _validate_corner_distances(candidate.corners):
        return "Corner positions are inconsistent with a single paper."

    if not bool(candidate.score_components.get("corner_to_mask_dist_ok", True)):
        components = candidate.score_components
        # Strong edge-derived quadrilaterals can lose their outer paper
        # pixels in a dark carpet shadow or when the foot occludes a corner.
        # Accept that case only when every independent geometric cue agrees;
        # weak or projected candidates still fail closed.
        strong_visible_boundary = (
            candidate.is_fully_visible
            and candidate.score >= 0.70
            and components.get("edge_continuity", 0.0) >= 0.88
            and components.get("strict_edge_continuity", 0.0) >= 0.88
            and components.get("visible_ratio_prior", 0.0) >= 0.90
            and components.get("rectified_a4_ratio", 0.0) >= 0.95
            and components.get("interior_paper_fraction", 0.0) >= 0.45
            and components.get("rectified_uniformity", 0.0) >= 0.70
            and not candidate.projected_corner
        )
        if not strong_visible_boundary:
            return "Detected corners do not follow the visible paper boundary."

    paper_frac = candidate.score_components.get("interior_paper_fraction", 0.0)
    if paper_frac > 0.0 and paper_frac < MIN_INTERIOR_PAPER_FRAC:
        return "Detected area includes too much non-paper region."

    rect_score = candidate.score_components.get("rectified_uniformity", 0.0)
    if rect_score > 0.0 and rect_score < MIN_RECTIFIED_UNIFORMITY:
        return "Rectified region is not sufficiently paper-like."

    if candidate.score < CONFIDENCE_THRESHOLD:
        # Strong geometric/visibility evidence is sufficient for the
        # side-by-side layout even when shadows slightly lower the aggregate
        # confidence score.
        visible = candidate.score_components.get("visible_ratio_prior", 0.0)
        rectified = candidate.score_components.get("rectified_a4_ratio", 0.0)
        interior = candidate.score_components.get("interior_paper_fraction", 0.0)
        if not (candidate.score >= 0.80 and visible >= 0.90 and rectified >= 0.95 and interior >= 0.70):
            return UNCERTAIN_MESSAGE

    return ""


def _select_best(
    candidates: List[A4Candidate],
    reject_reasons: Optional[List[str]] = None,
) -> Tuple[Optional[A4Candidate], str]:
    if not candidates:
        reason_str = " | ".join(reject_reasons or ["No large four-corner shapes found."])
        if any("extreme side ratio" in reason for reason in (reject_reasons or [])):
            return None, "Camera angle is too extreme. Retake the photo from above."
        return None, f"No valid A4 paper detected. Debug info: {reason_str}"

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    passable = [c for c in ranked if _candidate_blocker(c) == ""]
    if not passable:
        if len(ranked) >= 2 and ranked[1].score / max(ranked[0].score, 1e-6) >= AMBIGUITY_RATIO:
            return None, UNCERTAIN_MESSAGE
        top_reason = _candidate_blocker(ranked[0])
        if top_reason == "The complete A4 paper must be visible.":
            return None, top_reason
        return None, UNCERTAIN_MESSAGE if ranked[0].score < CONFIDENCE_THRESHOLD else top_reason

    best = passable[0]
    if len(passable) >= 2:
        second = passable[1]
        if second.score / max(best.score, 1e-6) >= AMBIGUITY_RATIO:
            strong_best = (
                best.score >= 0.82
                and best.score_components.get("visible_ratio_prior", 0.0) >= 0.90
                and best.score_components.get("rectified_a4_ratio", 0.0) >= 0.95
            )
            if not strong_best:
                return None, UNCERTAIN_MESSAGE

    return best, ""


def _detect_orientation(corners: np.ndarray) -> str:
    width, height = _estimate_width_height(corners)
    return "portrait" if height >= width else "landscape"


def _debug_rejection_panel(
    image: np.ndarray,
    candidates: List[A4Candidate],
    reject_reasons: List[str],
    message: str,
) -> np.ndarray:
    panel = image.copy()
    lines = [f"Final: {message}"] + reject_reasons[:8]
    y = 24
    for line in lines:
        cv2.putText(panel, line[:95], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)
        y += 20
    for idx, c in enumerate(sorted(candidates, key=lambda x: x.score, reverse=True)[:12]):
        pts = c.corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(panel, [pts], True, (0, 255, 255), 2)
        cv2.putText(panel, f"{idx}:{c.score:.2f}", tuple(c.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
    return panel


def _has_similar_a4_components(masks: List[np.ndarray], image_area: float) -> bool:
    """Detect duplicate, similarly sized A4-like blobs missed by contour scoring."""
    components: List[Tuple[float, float]] = []
    for mask in masks:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, n_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            frac = area / image_area
            if frac < MIN_AREA_FRACTION or frac > 0.20:
                continue
            w = float(stats[label, cv2.CC_STAT_WIDTH])
            h = float(stats[label, cv2.CC_STAT_HEIGHT])
            if min(w, h) < 10:
                continue
            ratio = max(w, h) / min(w, h)
            if _ratio_similarity(ratio, tolerance=0.20) > 0.85:
                components.append((area, ratio))

    components = sorted(components, reverse=True)
    if len(components) < 2:
        return False
    return components[1][0] / max(components[0][0], 1.0) >= AMBIGUITY_RATIO


def detect_a4_paper(
    image: np.ndarray,
    *,
    collect_debug: bool = False,
) -> A4DetectionResult:
    result = A4DetectionResult()
    all_debug: Dict[str, np.ndarray] = {}
    if collect_debug:
        all_debug["original"] = image.copy()

    edges, prep_debug = _preprocess(image, collect_debug)
    all_debug.update(prep_debug)

    img_h, img_w = image.shape[:2]
    image_area = float(img_h * img_w)

    paper_mask, paper_contour, paper_debug = _detect_paper_region(image, collect_debug)
    all_debug.update(paper_debug)

    candidates: List[A4Candidate] = []
    all_reject_reasons: List[str] = []
    edge_contours: List[np.ndarray] = []

    if paper_mask is not None and paper_contour is not None:
        pm_corners = _derive_corners_from_paper_mask(paper_mask, paper_contour)
        if pm_corners is not None and _validate_corner_distances(pm_corners):
            pm_corners = _refine_quad_from_long_edges(image, pm_corners)
            pm_area = _quad_area(pm_corners)
            pm_frac = pm_area / image_area
            if MIN_AREA_FRACTION <= pm_frac <= MAX_AREA_FRACTION:
                sides = _quad_side_lengths(pm_corners)
                width_est, height_est = _estimate_width_height(pm_corners)
                min_side = min(sides)
                side_ratio = max(sides) / max(min_side, 1.0)
                orientation = "portrait" if height_est >= width_est else "landscape"

                rectified, homography_condition, rectified_ratio = _rectify_candidate(
                    image, pm_corners, orientation,
                )
                if rectified is not None and homography_condition <= MAX_HOMOGRAPHY_CONDITION:
                    strict_ec = _validate_edge_continuity_strict(edges, pm_corners, paper_mask)
                    interior_frac = _validate_interior_paper_fraction(pm_corners, paper_mask)
                    rect_score = _rectified_paper_score(rectified)
                    corner_dists = _per_corner_mask_distance(pm_corners, paper_mask)
                    corners_ok, _ = _validate_corner_to_mask_boundary(pm_corners, paper_mask)

                    (_, _), (rw, rh), _ = cv2.minAreaRect(
                        pm_corners.astype(np.int32).reshape(-1, 1, 2),
                    )
                    rect_area = float(rw * rh) if rw * rh > 0 else 1.0
                    rectangularity = min(pm_area / rect_area, 1.0)
                    longer = max(width_est, height_est)
                    shorter = min(width_est, height_est)
                    aspect = longer / shorter
                    visible_ratio_prior = _ratio_similarity(aspect, tolerance=0.90)
                    rectified_a4_ratio = _ratio_similarity(
                        rectified_ratio, tolerance=A4_RATIO_TOLERANCE,
                    )
                    appearance = _appearance_scores(image, edges, pm_corners)
                    boundary_residual, reprojection_residual, corner_uncertainty = _boundary_residuals(
                        pm_corners,
                        paper_contour,
                        image=image,
                    )

                    pm_candidate = A4Candidate(
                        corners=pm_corners,
                        contour=paper_contour,
                        area=pm_area,
                        perimeter=cv2.arcLength(paper_contour, True),
                        rectangularity=rectangularity,
                        aspect_ratio=aspect,
                        ratio_similarity=rectified_a4_ratio,
                        is_convex=True,
                        is_fully_visible=_is_fully_visible(pm_corners, img_h, img_w),
                        side_lengths=sides,
                        rectified=rectified,
                        homography_condition=homography_condition,
                        boundary_hypothesis="paper_mask",
                        boundary_residual_px=boundary_residual,
                        reprojection_residual_px=reprojection_residual,
                        corner_uncertainty_px=corner_uncertainty,
                        visible_corner_count=4,
                    )
                    pm_candidate.score_components = {
                        "area": min((pm_area / image_area) / 0.10, 1.0),
                        "convexity": 1.0,
                        "edge_continuity": appearance["edge_continuity"],
                        "strict_edge_continuity": strict_ec,
                        "interior_brightness": appearance["interior_brightness"],
                        "interior_uniformity": appearance["interior_uniformity"],
                        "homography_stability": max(
                            0.0, 1.0 - np.log10(max(homography_condition, 1.0)) / 6.0,
                        ),
                        "rectified_a4_ratio": rectified_a4_ratio,
                        "distance_from_borders": 1.0 if pm_candidate.is_fully_visible else 0.0,
                        "visible_ratio_prior": visible_ratio_prior,
                        "internal_line_score": appearance["internal_line_score"],
                        "interior_paper_fraction": interior_frac,
                        "rectified_uniformity": rect_score,
                        "corner_to_mask_dist_ok": corners_ok,
                        "boundary_residual_score": float(np.clip(1.0 - boundary_residual / MAX_BOUNDARY_RESIDUAL_PX, 0.0, 1.0)),
                        "corner_uncertainty_score": float(np.clip(1.0 - corner_uncertainty / MAX_CORNER_UNCERTAINTY_PX, 0.0, 1.0)),
                    }
                    pm_candidate.score = _score_candidate(pm_candidate, image_area)
                    candidates.append(pm_candidate)

                    if collect_debug:
                        pm_dbg = image.copy()
                        pts = pm_corners.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(pm_dbg, [pts], True, (0, 255, 0), 3)
                        for ci, (c, d) in enumerate(zip(pm_corners, corner_dists)):
                            cv2.circle(pm_dbg, tuple(c.astype(int)), 8, (0, 200, 0), -1)
                            cv2.putText(
                                pm_dbg,
                                f"c{ci}:{d:.1f}px",
                                (int(c[0]) + 10, int(c[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                            )
                        cv2.putText(
                            pm_dbg,
                            f"paper_frac:{interior_frac:.2f} ec:{strict_ec:.2f}",
                            (10, img_h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                        )
                        all_debug["paper_mask_candidate"] = pm_dbg

    if collect_debug and paper_mask is not None:
        mask_bgr = cv2.cvtColor(paper_mask, cv2.COLOR_GRAY2BGR)
        all_debug["paper_mask_final"] = mask_bgr
        if paper_contour is not None:
            cv2.drawContours(mask_bgr, [paper_contour], -1, (0, 255, 0), 2)

    edge_candidates, edge_debug, edge_reject = _find_candidates(
        edges, image, edges, collect_debug, debug_prefix="edges_",
    )
    candidates.extend(edge_candidates)
    all_reject_reasons.extend(edge_reject)
    all_debug.update(edge_debug)

    # Low-contrast white paper on a pale patterned floor is often connected
    # to the background by the bright-region mask. A softer fixed Canny pass
    # followed by a wider close recovers the paper's four long boundary edges
    # without relaxing the normal candidate validation rules.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rescue_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 20, 60)
    rescue_closed = cv2.morphologyEx(
        rescue_edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
    )
    rescue_candidates, rescue_debug, rescue_reject = _find_candidates(
        rescue_closed,
        image,
        rescue_edges,
        collect_debug,
        debug_prefix="rescue_edges_",
    )
    candidates.extend(rescue_candidates)
    all_reject_reasons.extend(rescue_reject)
    all_debug.update(rescue_debug)
    if collect_debug:
        all_debug["rescue_edges"] = rescue_edges
        all_debug["rescue_edges_closed"] = rescue_closed

    for cnt in edge_candidates:
        edge_contours.append(cnt.contour)

    threshold_images, threshold_debug = _threshold_contour_images(image, collect_debug)
    similar_component_ambiguity = _has_similar_a4_components(
        threshold_images, float(image.shape[0] * image.shape[1])
    )
    all_debug.update(threshold_debug)
    for idx, threshold_image in enumerate(threshold_images):
        extra_candidates, extra_debug, extra_reject_reasons = _find_candidates(
            threshold_image,
            image,
            edges,
            collect_debug,
            debug_prefix=f"threshold_{idx}_",
        )
        candidates.extend(extra_candidates)
        all_reject_reasons.extend(extra_reject_reasons)
        all_debug.update(extra_debug)

    if paper_mask is not None and paper_contour is not None:
        for cand in candidates:
            if cand in [c for c in candidates if hasattr(c, '_pm_validated')]:
                continue
            int_frac = _validate_interior_paper_fraction(cand.corners, paper_mask)
            cand.score_components["interior_paper_fraction"] = int_frac

            corners_ok, _ = _validate_corner_to_mask_boundary(cand.corners, paper_mask)
            cand.score_components["corner_to_mask_dist_ok"] = corners_ok
            cand.score = _score_candidate(cand, image_area)

            if not corners_ok:
                all_reject_reasons.append(
                    f"candidate: one or more corners far from paper boundary"
                )

    candidates = _dedupe_candidates(candidates)
    result.candidates = candidates

    best, message = _select_best(candidates, all_reject_reasons)
    if best is not None and similar_component_ambiguity:
        best = None
        message = UNCERTAIN_MESSAGE
    if best is None:
        result.success = False
        result.message = message
        if collect_debug:
            all_debug["rejection_reasons"] = _debug_rejection_panel(
                image, candidates, all_reject_reasons, message,
            )
        result.debug_images = all_debug
        return result

    # Improve the calibration corners before they are used to measure the
    # heel/toe points.  The candidate score and detection diagnostics remain
    # based on the original contour, while the returned coordinates get the
    # higher precision available from the source image.
    calibrated_corners = _refine_corners_from_image(image, best.corners)

    result.success = True
    result.corners = calibrated_corners
    result.confidence = best.score
    result.orientation = _detect_orientation(best.corners)
    result.boundary_hypothesis = best.boundary_hypothesis
    result.boundary_residual_px = float(best.boundary_residual_px)
    result.reprojection_residual_px = float(best.reprojection_residual_px)
    result.corner_uncertainty_px = float(best.corner_uncertainty_px)
    result.visible_corner_count = int(best.visible_corner_count)
    result.three_corner_reconstruction = bool(best.projected_corner)
    result.message = (
        "A4 paper detected successfully. "
        "The system recognizes an A4-shaped paper and assumes the user "
        "placed a real A4 sheet."
    )

    if collect_debug:
        selected = image.copy()
        for idx, c in enumerate(sorted(candidates, key=lambda x: x.score, reverse=True)):
            pts = c.corners.astype(np.int32).reshape((-1, 1, 2))
            color = (0, 255, 0) if c is best else (255, 0, 255)
            cv2.polylines(selected, [pts], True, color, 2)
            cv2.putText(
                selected,
                f"{idx}:{c.score:.2f}",
                tuple(c.corners[0].astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        all_debug["selected_candidate"] = selected
        if best.rectified is not None:
            all_debug["rectified_paper_preview"] = best.rectified
        all_debug["rejection_reasons"] = _debug_rejection_panel(
            image, candidates, all_reject_reasons, "selected candidate accepted"
        )

    result.debug_images = all_debug
    return result
