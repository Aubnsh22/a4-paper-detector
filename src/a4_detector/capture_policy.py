"""Versioned engineering limits for A4 reference validation."""
from __future__ import annotations

import json
from importlib.resources import files

POLICY = json.loads(files(__package__).joinpath("reference_policy.json").read_text(encoding="utf-8"))

MESSAGES = {
    'a4_missing': 'A4 corners are not fully visible.',
    'a4_corners': 'Show all four physical A4 corners.',
    'a4_boundary': 'Reference geometry is uncertain. Keep the paper flat and its edges clear.',
    'a4_small': 'Move the phone closer while keeping the foot and all A4 corners visible.',
    'a4_large': 'Move the phone up to include the full foot and A4.',
    'resolution': 'Image resolution is insufficient. Use the rear camera at a higher resolution.',
    'tilt': 'Keep the phone only slightly behind the heel, with the whole A4 visible.',
    'pose': 'Reference geometry is uncertain. Keep the A4 flat and move the phone slightly closer to above it.',
}


def messages(reasons: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(MESSAGES.get(reason, "Reference rejected; retake the photo.") for reason in reasons))
