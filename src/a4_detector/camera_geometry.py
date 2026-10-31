"""Camera state validation and pinhole pose, without single-sheet self-calibration.

All world coordinates are millimetres. A4 is z=0; positive endpoint height
points toward the camera, independent of the recovered plane-normal sign.
An assumed focal length is suitable for guidance/research, never ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import a4_real_coordinates


def valid_quad(corners: np.ndarray, shape: tuple[int, int]) -> bool:
    quad = np.asarray(corners, dtype=np.float64)
    if quad.shape != (4, 2) or not np.all(np.isfinite(quad)):
        return False
    h, w = shape
    return bool(
        len(np.unique(quad, axis=0)) == 4
        and np.all(quad >= 0) and np.all(quad[:, 0] < w) and np.all(quad[:, 1] < h)
        and cv2.isContourConvex(quad.astype(np.float32))
        and abs(cv2.contourArea(quad.astype(np.float32))) > 100
    )


def centered_camera(shape: tuple[int, int], focal_px: float) -> np.ndarray:
    if not np.isfinite(focal_px) or focal_px <= 0:
        raise ValueError("Focal length must be positive and finite.")
    h, w = shape
    return np.array([[focal_px, 0, (w - 1) / 2], [0, focal_px, (h - 1) / 2], [0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size_wh: tuple[int, int]
    camera_id: str
    crop_state: str
    rms_px: float
    view_count: int

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != "camera_calibration_v1":
            raise ValueError("Unknown calibration schema.")
        result = cls(np.asarray(data["camera_matrix"], float), np.asarray(data["distortion"], float),
                     tuple(data["image_size_wh"]), str(data["camera_id"]), str(data["crop_state"]),
                     float(data["rms_px"]), int(data["view_count"]))
        result.for_image(tuple(reversed(result.image_size_wh)), result.camera_id, result.crop_state)
        return result

    def for_image(self, shape: tuple[int, int], camera_id: str, crop_state: str) -> tuple[np.ndarray, np.ndarray]:
        k = np.asarray(self.camera_matrix, float)
        d = np.asarray(self.distortion, float).reshape(-1)
        w, h = self.image_size_wh
        if (k.shape != (3, 3) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(d))
                or d.size not in (4, 5, 8, 12, 14) or min(w, h, *shape) <= 0
                or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-6
                or abs(k[1, 0]) > 1e-6 or min(k[0, 0], k[1, 1]) <= 0
                or not (0 <= k[0, 2] < w and 0 <= k[1, 2] < h)
                or not np.isfinite(self.rms_px) or not 0 <= self.rms_px <= 1.5 or self.view_count < 8):
            raise ValueError("Calibration is missing or unstable; use at least eight diverse calibration views.")
        if not self.camera_id or not self.crop_state or (camera_id, crop_state) != (self.camera_id, self.crop_state):
            raise ValueError("Camera, zoom, crop or stabilization state does not match the calibration.")
        sy, sx = shape[0] / h, shape[1] / w
        if not np.isclose(sx, sy, rtol=0.002):
            raise ValueError("Calibration aspect ratio does not match the image; crop/orientation is unknown.")
        scaled = k.copy()
        scaled[0, 0] *= sx
        scaled[1, 1] *= sy
        scaled[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
        scaled[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
        return scaled, d


@dataclass(frozen=True)
class PlanePose:
    camera: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    center: np.ndarray
    residual_px: float
    tilt_deg: float
    solver: str = "ippe_refined"

    @property
    def height_mm(self) -> float:
        return abs(float(self.center[2]))

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        points = cv2.undistortPoints(np.asarray(pixels, float).reshape(-1, 1, 2), self.camera, self.distortion)
        return np.column_stack((points.reshape(-1, 2), np.ones(len(points))))

    def points_at_heights(self, pixels: np.ndarray, heights_mm: np.ndarray) -> np.ndarray:
        rays = self.rays(pixels) @ self.rotation
        heights = np.broadcast_to(np.asarray(heights_mm, float), (len(rays),))
        if not np.all(np.isfinite(heights)) or np.any(heights < 0) or np.any(heights >= self.height_mm):
            raise ValueError("Endpoint heights must lie between the reference plane and camera.")
        if np.any(np.abs(rays[:, 2]) < 1e-9):
            raise ValueError("Endpoint ray is parallel to the plane.")
        scale = (np.sign(self.center[2]) * heights - self.center[2]) / rays[:, 2]
        if np.any(scale <= 0):
            raise ValueError("Endpoint is behind the camera.")
        return self.center + rays * scale[:, None]

    def world_from_z_depth(self, pixels: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_mm, float).reshape(-1)
        if not np.all(np.isfinite(depth)) or np.any(depth <= 0):
            raise ValueError("Invalid metric camera z-depth.")
        return (self.rays(pixels) * depth[:, None] - self.translation) @ self.rotation


def solve_a4_pose(corners: np.ndarray, shape: tuple[int, int], camera: np.ndarray,
                  distortion: np.ndarray | None = None) -> PlanePose:
    if not valid_quad(corners, shape):
        raise ValueError("All four distinct, ordered, visible A4 corners are required.")
    k = np.asarray(camera, float)
    if (k.shape != (3, 3) or not np.all(np.isfinite(k)) or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-6 or abs(k[1, 0]) > 1e-6):
        raise ValueError("Camera matrix is invalid.")
    distortion = np.zeros(5) if distortion is None else np.asarray(distortion, float).reshape(-1)
    if distortion.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(distortion)):
        raise ValueError("Lens distortion is invalid.")
    objects = np.column_stack((a4_real_coordinates(corners), np.zeros(4))).astype(float)
    pixels = np.asarray(corners, float)
    count, rotations, translations, _ = cv2.solvePnPGeneric(objects, pixels, k, distortion, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    def evaluate(rv, tv, solver):
        if not np.all(np.isfinite(rv)) or not np.all(np.isfinite(tv)):
            return
        rv, tv = cv2.solvePnPRefineLM(objects, pixels, k, distortion, rv.copy(), tv.copy())
        rotation, _ = cv2.Rodrigues(rv)
        translation = tv.reshape(3)
        center = -rotation.T @ translation
        if not np.all(np.isfinite(center)) or np.min(objects @ rotation[2] + translation[2]) <= 0:
            return
        projected = cv2.projectPoints(objects, rv, tv, k, distortion)[0].reshape(-1, 2)
        residual = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
        tilt = float(np.degrees(np.arccos(np.clip(abs(rotation[2, 2]), 0, 1))))
        candidates.append(PlanePose(k, distortion, rotation, translation, center, residual, tilt, solver))
    for rv, tv in zip(rotations, translations) if count else ():
        evaluate(rv, tv, "ippe_refined")
    if not candidates:
        # OpenCV IPPE can return NaNs for exact synthetic symmetries. An
        # independent planar initialization still has to pass cheirality and
        # reprojection checks; missing or invalid corners are never recovered.
        found, rv, tv = cv2.solvePnP(objects, pixels, k, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if found:
            evaluate(rv, tv, "iterative_fallback")
    if not candidates:
        raise ValueError("A4 pose is not physically recoverable.")
    return min(candidates, key=lambda item: item.residual_px)


def visual_alignment(corners: np.ndarray, shape: tuple[int, int]) -> dict:
    """Conservative focal sensitivity range, NOT a calibrated pose estimate.

    The browser has no reliable focal/crop metadata. Require near-overhead
    agreement over a wide focal range instead of optimizing focal to the sheet.
    """
    poses = [solve_a4_pose(corners, shape, centered_camera(shape, max(shape) * factor))
             for factor in (0.55, 0.85, 1.2, 1.6)]
    representative = poses[1]
    normal = representative.rotation[:, 2] * np.sign(representative.rotation[2, 2])
    return {
        "source": "a4_visual_focal_sweep", "calibrated": False,
        "tilt_deg": float(max(p.tilt_deg for p in poses)),
        "tilt_range_deg": [float(min(p.tilt_deg for p in poses)), float(max(p.tilt_deg for p in poses))],
        "pitch_deg": float(np.degrees(np.arctan2(normal[1], normal[2]))),
        "roll_deg": float(np.degrees(np.arctan2(normal[0], normal[2]))),
        "residual_px": float(max(p.residual_px for p in poses)),
        "focal_is_assumed": True,
    }
