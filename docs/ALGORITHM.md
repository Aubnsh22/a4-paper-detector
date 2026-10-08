# Detection algorithm

## Input and output

The input is an 8-bit BGR image. The core detector returns an
`A4DetectionResult` containing a candidate quadrilateral, score, orientation,
boundary diagnostics, and optional debug images. The application API adds a
separate decision about whether that quadrilateral is an acceptable reference.

## Candidate generation

1. Convert the image to grayscale, apply Gaussian blur, and extract Canny
   edges with intensity-dependent thresholds. Morphological closing connects
   nearby edge fragments.
2. Build paper-region masks and additional light/dark foreground masks using
   Otsu thresholding. Search contours for convex quadrilateral candidates.
3. Run a softer edge pass to recover paper boundaries that merge with pale
   or patterned backgrounds.
4. Refine candidate corners using observed image edges. The core also retains
   a three-corner reconstruction path for experimentation; the application
   API rejects that result as a measurement reference.

The implementation is in
[`a4_detection.py`](../src/a4_detector/a4_detection.py). It is classical image
processing; there is no learned A4 model.

## Candidate checks and ranking

Candidates are checked for visibility, convexity, side geometry, homography
conditioning, paper-mask support, edge continuity, brightness, and uniformity.
Each candidate is rectified to an A4-shaped preview for appearance checks.

Ranking combines area, convexity, boundary evidence, brightness, uniformity,
homography stability, aspect-ratio priors, border distance, internal lines,
and the fraction of paper-like interior pixels. Candidates with incompatible
geometry or insufficient boundary support are rejected. Similar competing
candidates can cause an ambiguous result.

The returned score lies between 0 and 1. It is not a calibrated probability.
Rectification to an A4-shaped destination also does not prove that the object
is physically A4.

## Boundary recovery

If the initial candidate fails reference validation,
[`product_a4.py`](../src/a4_detector/product_a4.py) searches for isolated
low-saturation, bright regions at several thresholds. If this does not recover
the sheet, it also tests blue/red colour contrast to distinguish cool paper
from warm pale tiles.

A replacement must have strong observed edge support, acceptable geometry,
and a near-overhead pose. No replacement is made when the original reference
already passes its checks. Competing plausible sheets remain ambiguous.

## Reference acceptance

[`reference.py`](../src/a4_detector/reference.py) checks resolution, four
physical corners, frame margins, paper size, boundary diagnostics, and pose
residuals. The public `detect_a4()` function also applies the original
application's final A4 tilt limit.

Pose is evaluated over four assumed focal lengths: 0.55, 0.85, 1.2, and 1.6
times the larger image dimension. The largest inferred tilt and residual are
used. This is a sensitivity check under a pinhole model, not camera
calibration or a measured phone angle. Lens distortion is assumed zero.

## Source extraction

Version 0.1.0 extracts the Python A4 path from the Foot Measurement application:

| Original component | Standalone component |
| --- | --- |
| `src/a4_detection.py` | `a4_detector.a4_detection`, same core algorithm |
| `src/product_a4.py` | `a4_detector.product_a4`, adjusted validation import |
| A4 functions in `src/capture_assessment.py` | `a4_detector.reference` |
| A4 pose helpers in `src/camera_geometry.py` | `a4_detector.camera_geometry` |
| A4 coordinates and homography in `src/geometry.py` | `a4_detector.geometry` |
| A4 entries from `config/capture_policy.json` | Packaged `reference_policy.json` |

The public wrapper, JSON output, CLI, packaging, and integration tests are
specific to this repository. The original core retains its
`FOOT_A4_RESIDUAL_POLICY` environment setting for reproducibility. Leave it
unset for the documented baseline; `strict_v1` changes core candidate
filtering. The application's reference checks still apply in `detect_a4()`.

Synthetic regression tests were retained. A test requiring a private real
photograph was excluded. The package does not include the source
application's photos, dataset, foot segmentation models, capture UI, or
measurement results. Passing the included tests does not establish accuracy
across real cameras or users.
