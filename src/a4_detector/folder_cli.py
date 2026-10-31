"""Run the A4 reference detector on every supported image in a folder."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .api import detect_a4

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _overlay(image: np.ndarray, result) -> np.ndarray:
    output = image.copy()
    corners = result.corners
    color = (48, 180, 76) if result.valid else (40, 60, 220)
    if corners is not None:
        points = np.rint(corners).astype(np.int32)
        cv2.polylines(output, [points], True, color, 4, cv2.LINE_AA)
        for label, point in zip(("TL", "TR", "BR", "BL"), points):
            cv2.circle(output, tuple(point), 7, color, -1, cv2.LINE_AA)
            cv2.putText(output, label, tuple(point + [9, -9]), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(output, "A4 not accepted", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 2, cv2.LINE_AA)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare A4 detection results across a folder of photos.")
    parser.add_argument("folder", type=Path, help="Folder of JPEG, PNG, TIFF, BMP, or WebP images.")
    parser.add_argument("--output-dir", type=Path, default=Path("a4-results"), help="Directory for annotated images and reports (default: a4-results).")
    parser.add_argument("--recursive", action="store_true", help="Include images in subfolders and preserve their folder structure.")
    args = parser.parse_args(argv)

    if not args.folder.is_dir():
        print(f"a4-detect-folder: not a folder: {args.folder}", file=sys.stderr)
        return 2
    output_dir = args.output_dir.resolve()
    source_images = args.folder.rglob("*") if args.recursive else args.folder.iterdir()
    images = sorted(
        p for p in source_images
        if p.is_file() and p.suffix.lower() in SUPPORTED
        and output_dir not in p.resolve().parents
    )
    if not images:
        print("a4-detect-folder: no supported images found", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accepted = 0
    for path in images:
        relative = path.relative_to(args.folder)
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        row = {"image": relative.as_posix(), "valid": False, "reasons": "", "messages": "", "corners_px": "", "elapsed_ms": "", "error": ""}
        if image is None:
            row["error"] = "Could not decode image"
        else:
            start = time.perf_counter()
            try:
                result = detect_a4(image)
                row.update({
                    "valid": result.valid,
                    "reasons": "; ".join(result.reasons),
                    "messages": "; ".join(result.messages),
                    "corners_px": json.dumps(result.corners.tolist() if result.corners is not None else None),
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                })
                accepted += int(result.valid)
                target = output_dir / "annotated" / relative.with_suffix(".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                ok, encoded = cv2.imencode(".png", _overlay(image, result))
                if not ok:
                    raise OSError("Could not encode annotated image")
                target.write_bytes(encoded.tobytes())
                detail = output_dir / "results" / relative.with_suffix(".json")
                detail.parent.mkdir(parents=True, exist_ok=True)
                detail.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
            except (ValueError, cv2.error, OSError) as exc:
                row["error"] = str(exc)
        rows.append(row)

    fields = ["image", "valid", "reasons", "messages", "corners_px", "elapsed_ms", "error"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"input_folder": str(args.folder.resolve()), "images": len(rows), "accepted": accepted, "rejected": len(rows) - accepted, "results": rows}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Processed {len(rows)} images: {accepted} accepted, {len(rows) - accepted} rejected or unreadable.")
    print(f"Annotated images: {output_dir / 'annotated'}")
    print(f"Comparison report: {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
