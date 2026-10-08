"""Command-line detection with JSON output and an optional corner overlay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from .api import detect_a4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect and validate a physical A4 reference in a photo.")
    parser.add_argument("image", type=Path, help="Input photo readable by OpenCV, such as JPEG or PNG.")
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout.")
    parser.add_argument("--overlay", type=Path, help="Write a PNG or JPEG showing accepted corners.")
    args = parser.parse_args(argv)
    try:
        targets = [p.resolve() for p in (args.image, args.output, args.overlay) if p is not None]
        if len(set(targets)) != len(targets):
            raise ValueError("Input image, JSON output and overlay must use different paths.")
        content = np.frombuffer(args.image.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(content, cv2.IMREAD_COLOR)
        result = detect_a4(image)
        payload = json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n"
        if args.overlay is not None:
            overlay = image.copy()
            if result.corners is not None:
                corners = np.rint(result.corners).astype(np.int32)
                cv2.polylines(overlay, [corners], True, (55, 170, 70), 3)
                for label, point in zip(("TL", "TR", "BR", "BL"), corners):
                    cv2.circle(overlay, tuple(point), 5, (55, 170, 70), -1)
                    cv2.putText(overlay, label, tuple(point + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, .65, (55, 170, 70), 2)
            else:
                cv2.putText(overlay, "Reference rejected", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 200), 2)
            encoded, data = cv2.imencode(args.overlay.suffix, overlay)
            if not encoded:
                raise ValueError("Could not encode the overlay image.")
            args.overlay.parent.mkdir(parents=True, exist_ok=True)
            args.overlay.write_bytes(data.tobytes())
        if args.output is None:
            sys.stdout.write(payload)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        return 0 if result.valid else 1
    except (OSError, ValueError, cv2.error) as exc:
        print(f"a4-detect: {exc}", file=sys.stderr)
        return 2
