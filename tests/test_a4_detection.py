"""
Unit tests for a4_detector.a4_detection
===============================
Run with:
    pytest tests/test_a4_detection.py -v
"""

import pytest
import numpy as np
import cv2

from a4_detector.a4_detection import (
    detect_a4_paper,
    _quad_area,
    _quad_side_lengths,
    _estimate_width_height,
    order_corners,
    _project_fourth_corner,
    _project_fourth_corner_from_edges,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_synthetic_image(corners: list, img_size=(800, 600), bg_color=50, fg_color=200):
    """Create a synthetic BGR image with a filled polygon."""
    img = np.full((img_size[1], img_size[0], 3), bg_color, dtype=np.uint8)
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (fg_color, fg_color, fg_color))
    return img


def create_tiled_floor(img_size=(900, 700), base=115, line=65, tile=110):
    """Create a simple tiled floor with visible grout lines."""
    img = np.full((img_size[1], img_size[0], 3), base, dtype=np.uint8)
    for x in range(0, img_size[0], tile):
        cv2.line(img, (x, 0), (x, img_size[1]), (line, line, line), 3)
    for y in range(0, img_size[1], tile):
        cv2.line(img, (0, y), (img_size[0], y), (line, line, line), 3)
    return img


def draw_paper(img, corners, color=230):
    pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillConvexPoly(img, pts, (color, color, color))
    cv2.polylines(img, [pts], True, (245, 245, 245), 2)
    return img


def _assert_close(a: float, b: float, tol: float = 1.0) -> None:
    assert abs(a - b) < tol, f"{a:.4f} != {b:.4f} (tol={tol})"


# ---------------------------------------------------------------------------
# Tests for internal geometry helpers
# ---------------------------------------------------------------------------

def test_order_corners():
    # Disordered square
    pts = np.array([
        [100, 100],  # BR
        [0, 100],    # BL
        [0, 0],      # TL
        [100, 0],    # TR
    ])
    ordered = order_corners(pts)
    assert np.allclose(ordered[0], [0, 0])
    assert np.allclose(ordered[1], [100, 0])
    assert np.allclose(ordered[2], [100, 100])
    assert np.allclose(ordered[3], [0, 100])

def test_quad_area():
    corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]])
    area = _quad_area(corners)
    _assert_close(area, 5000.0)

def test_quad_side_lengths():
    corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]])
    t, r, b, l = _quad_side_lengths(corners)
    _assert_close(t, 100.0)
    _assert_close(r, 50.0)
    _assert_close(b, 100.0)
    _assert_close(l, 50.0)

def test_estimate_width_height():
    corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]])
    w, h = _estimate_width_height(corners)
    _assert_close(w, 100.0)
    _assert_close(h, 50.0)


def test_project_fourth_corner_from_three_rectangle_points():
    points = np.array([[100, 100], [310, 100], [100, 397]], dtype=np.float32)
    inferred = _project_fourth_corner(points)
    assert inferred is not None
    assert np.linalg.norm(inferred - np.array([310, 397], dtype=np.float32)) < 1.0


def test_project_fourth_corner_uses_visible_edge_lines():
    points = np.array([[100, 100], [300, 110], [110, 380]], dtype=np.float32)
    edges = np.zeros((500, 500), dtype=np.uint8)
    cv2.line(edges, (300, 110), (310, 390), 255, 3)
    cv2.line(edges, (110, 380), (310, 390), 255, 3)
    inferred = _project_fourth_corner_from_edges(points, edges)
    assert inferred is not None
    assert np.linalg.norm(inferred - np.array([310, 390], dtype=np.float32)) < 20.0


# ---------------------------------------------------------------------------
# Detection tests (using synthetic images)
# ---------------------------------------------------------------------------

class TestA4Detection:

    def test_clean_a4_paper_portrait(self):
        """Perfect A4 proportion in portrait."""
        # 210 x 297 ratio
        corners = [[100, 100], [310, 100], [310, 397], [100, 397]]
        img = create_synthetic_image(corners)
        res = detect_a4_paper(img)
        assert res.success is True
        assert res.orientation == "portrait"
        assert res.confidence > 0.7
        # Should match input roughly
        assert np.linalg.norm(res.corners[0] - [100, 100]) < 5

    def test_clean_a4_paper_landscape(self):
        """Perfect A4 proportion in landscape."""
        corners = [[100, 100], [397, 100], [397, 310], [100, 310]]
        img = create_synthetic_image(corners)
        res = detect_a4_paper(img)
        assert res.success is True
        assert res.orientation == "landscape"

    def test_rotated_a4_paper(self):
        """A4 paper rotated by 45 degrees."""
        # A simple rotation test might be hard to hand-code coordinates for,
        # so we build a clean rectangle and use cv2.warpAffine.
        base = create_synthetic_image([[200, 100], [410, 100], [410, 397], [200, 397]], img_size=(800, 800))
        center = (400, 400)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated = cv2.warpAffine(base, M, (800, 800))

        res = detect_a4_paper(rotated)
        assert res.success is True

    def test_perspective_distorted_a4_paper(self):
        """A4 paper with perspective distortion."""
        corners = [[150, 150], [350, 180], [330, 450], [120, 400]]
        img = create_synthetic_image(corners)
        res = detect_a4_paper(img)
        assert res.success is True
        # Distorted but still roughly within tolerance

    def test_dark_floor_light_floor(self):
        corners = [[100, 100], [310, 100], [310, 397], [100, 397]]

        # Dark floor
        img_dark = create_synthetic_image(corners, bg_color=10, fg_color=255)
        assert detect_a4_paper(img_dark).success is True

        # Light floor (paper is darker)
        img_light = create_synthetic_image(corners, bg_color=255, fg_color=100)
        assert detect_a4_paper(img_light).success is True

    def test_non_a4_shaped_rectangle(self):
        """Square instead of A4."""
        corners = [[100, 100], [300, 100], [300, 300], [100, 300]]
        img = create_synthetic_image(corners)
        res = detect_a4_paper(img)
        assert res.success is False
        assert "does not appear to be A4" in res.message

    def test_paper_touching_borders(self):
        """Paper touches the edge of the image, should fail full visibility check."""
        corners = [[2, 100], [210, 100], [210, 397], [2, 397]]
        img = create_synthetic_image(corners)
        res = detect_a4_paper(img)
        assert res.success is False
        assert "complete A4 paper must be visible" in res.message

    def test_extreme_perspective(self):
        """Highly distorted trapezoid that fails opposite-side ratio check."""
        # Top side = 200, Bottom side = 80 -> ratio = 2.5 (max allowed is 2.0 in select_best)
        corners = [[100, 100], [300, 100], [240, 300], [160, 300]]
        img = create_synthetic_image(corners)
        res = detect_a4_paper(img)
        assert res.success is False
        assert "Camera angle is too extreme" in res.message

    def test_no_paper(self):
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        res = detect_a4_paper(img)
        assert res.success is False
        assert "No valid A4 paper" in res.message

    def test_multiple_rectangles_ambiguity(self):
        """Two identical A4 shapes should cause ambiguity."""
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        # First A4
        c1 = np.array([[50, 50], [155, 50], [155, 198], [50, 198]])
        cv2.fillPoly(img, [c1], (200, 200, 200))
        # Second A4 (identical size)
        c2 = np.array([[300, 50], [405, 50], [405, 198], [300, 198]])
        cv2.fillPoly(img, [c2], (200, 200, 200))

        res = detect_a4_paper(img)
        assert res.success is False
        assert "uncertain" in res.message

    def test_multiple_rectangles_clear_winner(self):
        """One A4 shape and one tiny arbitrary shape."""
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        # Real A4
        c1 = np.array([[100, 100], [310, 100], [310, 397], [100, 397]])
        cv2.fillPoly(img, [c1], (200, 200, 200))
        # Small noise square
        c2 = np.array([[500, 100], [520, 100], [520, 120], [500, 120]])
        cv2.fillPoly(img, [c2], (200, 200, 200))

        res = detect_a4_paper(img)
        assert res.success is True

    def test_strongly_tilted_but_valid_a4(self):
        """A fully visible A4 under strong but not collapsed perspective is accepted."""
        corners = [[150, 120], [380, 175], [315, 500], [115, 380]]
        img = create_synthetic_image(corners, img_size=(800, 650), bg_color=70, fg_color=235)
        res = detect_a4_paper(img, collect_debug=True)
        assert res.success is True
        assert res.confidence >= 0.48
        assert "rectified_paper_preview" in res.debug_images

    def test_a4_on_tiled_floor(self):
        """A bright low-texture A4 should win over floor tile rectangles."""
        img = create_tiled_floor()
        paper = [[210, 115], [430, 150], [390, 470], [160, 405]]
        draw_paper(img, paper, color=238)
        res = detect_a4_paper(img)
        assert res.success is True
        assert np.linalg.norm(res.corners.mean(axis=0) - np.array(paper).mean(axis=0)) < 45

    def test_a4_with_reflections(self):
        """Mild reflective streaks on the paper should not make detection fail."""
        paper = [[170, 105], [390, 120], [370, 430], [145, 405]]
        img = create_synthetic_image(paper, img_size=(800, 600), bg_color=80, fg_color=225)
        cv2.ellipse(img, (260, 250), (85, 18), -18, 0, 360, (255, 255, 255), -1)
        cv2.GaussianBlur(img, (5, 5), 0, dst=img)
        res = detect_a4_paper(img)
        assert res.success is True

    def test_multiple_floor_rectangles_are_uncertain(self):
        """Several similarly plausible floor rectangles should not be accepted as paper."""
        img = create_tiled_floor(img_size=(800, 600), base=170, line=60, tile=140)
        res = detect_a4_paper(img)
        assert res.success is False
        assert "uncertain" in res.message or "No valid A4 paper" in res.message

    def test_one_clear_paper_candidate_among_several_rectangles(self):
        """Multiple rectangles are allowed when one paper candidate clearly scores best."""
        img = create_tiled_floor()
        # Add darker tile-like rectangles.
        cv2.rectangle(img, (35, 35), (145, 190), (125, 125, 125), -1)
        cv2.rectangle(img, (610, 250), (735, 425), (130, 130, 130), -1)
        paper = [[245, 105], [470, 135], [430, 465], [195, 405]]
        draw_paper(img, paper, color=245)
        res = detect_a4_paper(img)
        assert res.success is True
        assert np.linalg.norm(res.corners.mean(axis=0) - np.array(paper).mean(axis=0)) < 50

    def test_truly_extreme_invalid_perspective(self):
        """Reject a visible but almost collapsed paper edge."""
        corners = [[120, 110], [420, 120], [255, 360], [235, 355]]
        img = create_synthetic_image(corners, img_size=(800, 600), bg_color=60, fg_color=235)
        res = detect_a4_paper(img)
        assert res.success is False
        assert "too extreme" in res.message or "uncertain" in res.message

    def test_paper_on_tiled_floor_no_corner_bleed(self):
        """Regression: floor-tile intersection must never be selected as a paper corner.

        The old pipeline could combine paper edges with floor grout lines into
        one contour, producing a quad whose top-left corner was a tile intersection
        far from the paper.  The new pipeline must reject that quad and find the
        actual paper corners instead.
        """
        img = create_tiled_floor(img_size=(900, 700), base=100, line=55, tile=120)

        paper_tl = [250, 160]
        paper_tr = [470, 190]
        paper_br = [430, 490]
        paper_bl = [200, 450]
        paper_corners = [paper_tl, paper_tr, paper_br, paper_bl]
        draw_paper(img, paper_corners, color=240)

        res = detect_a4_paper(img, collect_debug=True)
        assert res.success is True, f"Detection failed: {res.message}"

        paper_center = np.mean(paper_corners, axis=0)
        detected_center = res.corners.mean(axis=0)
        center_dist = float(np.linalg.norm(detected_center - paper_center))
        assert center_dist < 60, (
            f"Detected center {detected_center} is too far from paper center "
            f"{paper_center} (dist={center_dist:.1f}px).  "
            f"A floor-tile corner may have been selected."
        )

        paper_arr = np.array(paper_corners, dtype=np.float32)
        for i in range(4):
            dists_to_paper = np.linalg.norm(paper_arr - res.corners[i], axis=1)
            min_dist = float(np.min(dists_to_paper))
            assert min_dist < 50, (
                f"Corner {i} at {res.corners[i]} is {min_dist:.1f}px from the "
                f"nearest paper corner — likely a floor-tile intersection."
            )

    def test_paper_near_tile_intersection_rejects_far_corner(self):
        """Place a paper corner near a tile intersection to stress-test rejection.

        One paper corner intentionally sits very close to a floor grout
        intersection.  The detector must still pick the correct paper corner
        and not 'jump' to the tile intersection.
        """
        img = create_tiled_floor(img_size=(900, 700), base=110, line=60, tile=120)

        paper_corners = [[242, 122], [460, 150], [420, 450], [195, 410]]
        draw_paper(img, paper_corners, color=235)

        res = detect_a4_paper(img, collect_debug=True)
        assert res.success is True, f"Detection failed: {res.message}"

        paper_arr = np.array(paper_corners, dtype=np.float32)
        max_min_corner_dist = 0.0
        for i in range(4):
            dists = np.linalg.norm(paper_arr - res.corners[i], axis=1)
            max_min_corner_dist = max(max_min_corner_dist, float(np.min(dists)))
        assert max_min_corner_dist < 55, (
            f"Worst corner distance from paper is {max_min_corner_dist:.1f}px — "
            f"a floor corner was likely selected instead of a paper corner."
        )
