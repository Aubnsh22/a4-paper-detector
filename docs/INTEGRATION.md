# Integration

## Image contract

`detect_a4(image, collect_debug=False)` takes a NumPy `uint8` array of shape
`(height, width, 3)` in BGR order. It raises `ValueError` for malformed inputs.
Images smaller than 24 pixels on either side are invalid inputs; otherwise,
an image below the reference resolution limit returns a rejected result.

For Pillow or another RGB source, convert to BGR:

```python
image_bgr = cv2.cvtColor(np.asarray(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
```

Apply any EXIF orientation before detection. Use the same decoded image and
dimensions throughout your processing pipeline. If you resize or crop first,
the returned coordinates belong to that resized or cropped image.

Detection is synchronous CPU work. In an asynchronous web application, run
it in your application's worker or executor rather than on the event loop.
Set appropriate upload and image-size limits in the host application.

## Result

| Field | Meaning |
| --- | --- |
| `valid` | Whether the A4 reference passed the configured checks |
| `corners` | Accepted `(4, 2)` pixel coordinates, or `None` after rejection |
| `confidence` | Candidate ranking score in `[0, 1]`; not a probability |
| `reasons` | Machine-readable rejection codes |
| `messages` | Readable rejection messages |
| `detection` | Core detection result, including any rejected candidate |
| `assessment` | Detailed checks, normalized diagnostic corners, and alignment |
| `to_dict()` | JSON-compatible result with accepted pixel coordinates |

Corner order is top-left, top-right, bottom-right, bottom-left in the image.
The top-left convention uses the smallest coordinate sum; corners are not
named physical markers on the paper. Orientation and short/long-side
assignment are inferred from projected side lengths and can be ambiguous
under strong perspective or diagonal rotation.

For overlays, draw `result.corners` on the same image passed to detection.
`assessment["corners"]` uses normalized diagnostic coordinates and can be
present after rejection; it must not be treated as an accepted reference.

With `collect_debug=True`, `result.detection.debug_images` contains intermediate
images from the core pass. If colour-boundary recovery replaces the candidate,
those images still describe the initial pass. They are not included in JSON.

## Plane coordinates

After checking `result.valid`, the helper below maps pixels onto the assumed
A4 plane in millimetres:

```python
import cv2
import numpy as np
from a4_detector import detect_a4, compute_homography

result = detect_a4(image_bgr)
if not result.valid:
    raise ValueError("; ".join(result.messages))

homography = compute_homography(result.corners)
points_px = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
points_mm = cv2.perspectiveTransform(points_px, homography)[0]
distance_mm = float(np.linalg.norm(points_mm[1] - points_mm[0]))
```

Supply `x1`, `y1`, `x2`, and `y2` from your own landmark detector. The helper
selects portrait or landscape dimensions from the average opposite side
lengths. It assumes the physical sheet is exactly 210 x 297 mm.

This transformation applies to the paper plane. A point above that plane,
such as an elevated toe, has parallax; mapping its image position through a
homography does not recover its true projection onto the support surface.
Error can also grow outside the paper boundary. The package does not correct
lens distortion or certify a final foot length.

## Default acceptance limits

The bundled [policy](../src/a4_detector/reference_policy.json) preserves the
source application's A4-related engineering limits:

| Check | Limit |
| --- | --- |
| Short image dimension | At least 720 px |
| Shortest detected paper edge | At least 180 px |
| Paper area / image area | 0.04 to 0.55 |
| Paper margin from image boundary | At least 0.015 of the short image dimension |
| Visible physical corners | Four, with no reconstructed corner |
| Candidate score | At least 0.60 |
| Corner uncertainty diagnostic | At most 8 px |
| Boundary residual diagnostic | At most 12 px |
| Pose residual / shortest paper edge | At most 0.012 |
| Maximum tilt across assumed focal lengths | At most 12 degrees |

These are tuned acceptance limits, not universal camera-independent error
bounds. There is no per-request configuration interface in version 0.2.0.
If you change the policy in a fork, version that change and evaluate the
effect on both accepted and rejected captures.

## Comparing a photo set

Run `a4-detect-folder ./photos --output-dir ./a4-results` to apply the same
`detect_a4()` decision to every supported image in a folder. Open
`a4-results/annotated/` to inspect the corners and `a4-results/summary.csv`
to compare decisions, pixel coordinates, rejection reasons, and run times.
The `results/` subfolder contains each complete JSON result. Use
`--recursive` to include nested folders. A rejected image stays in the report
with its reason; an image that cannot be decoded gets an error entry.

When comparing with another detector, use the same oriented source image and
coordinate convention. The corners returned here are ordered top-left,
top-right, bottom-right, bottom-left in image coordinates. Do not compare
pixels from separately resized images without mapping them back to the same
image dimensions.

## Choosing an API

Use `detect_a4()` when you need the original application's A4 acceptance
behavior. It does not apply foot segmentation, landmark, blur, motion, or
end-to-end measurement checks from the source application.

Use `detect_a4_paper()` if your system already owns its validation policy.
Its `success` means the core selected a candidate. Inspect
`visible_corner_count`, `three_corner_reconstruction`, and the boundary
diagnostics before relying on that candidate.

Neither API can distinguish physical A4 from a differently sized sheet with
the same proportions using the image alone.
