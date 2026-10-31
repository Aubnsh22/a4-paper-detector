# A4 Paper Detector

This repository contains the A4 detection and reference-checking code used by the foot-measurement project. It can be run on one photo or on a folder of photos. Folder mode saves an annotated copy of every photo and a report for comparing results.

The detector uses OpenCV and NumPy on the CPU. It does not download a model or send images over the network.

## Install

Requires Python 3.11 or newer.

```bash
python -m pip install "git+https://github.com/Aubnsh22/a4-paper-detector.git@v0.2.0"
```

The package installs `opencv-python-headless`. Avoid installing a second OpenCV wheel into the same environment.

## Compare a folder of photos

Put the test photos in a folder, then run:

```bash
a4-detect-folder ./photos --output-dir ./a4-results
```

The output contains:

- `annotated/`: photos with the accepted A4 boundary in green or a rejection label in red.
- `results/`: the full JSON result for each photo, including corners and any rejection messages.
- `summary.csv`: one row per input photo for a quick spreadsheet comparison.
- `summary.json`: the same summary with overall accepted and rejected counts.

Images in subfolders are included with `--recursive`:

```bash
a4-detect-folder ./photos --recursive --output-dir ./a4-results
```

You can compare the green corners with another detector's output and compare its decision with the `valid` column. Rejected photos are included in the report; an unreadable photo has an error in its row. Supported formats are JPEG, PNG, TIFF, BMP, and WebP.

## Run on one photo

```bash
a4-detect ./photo.jpg --output result.json --overlay overlay.png
```

The command exits with `0` when the reference is accepted, `1` when it is rejected, and `2` for an input or output error.

## Use from Python

```python
import cv2
from a4_detector import detect_a4

image = cv2.imread("photo.jpg")  # uint8 BGR image
if image is None:
    raise ValueError("Could not read photo.jpg")

result = detect_a4(image)
print("Accepted:", result.valid)
print("Corners:", result.corners)  # TL, TR, BR, BL in input-image pixels, or None
print("Reasons:", result.reasons)
print("Messages:", result.messages)
```

Orient the image according to its EXIF data before detection. If the image is resized or cropped first, returned coordinates refer to that transformed image. `confidence` is a ranking score, not a probability.

## What is included

`detect_a4()` runs the core paper detector, the project's boundary-recovery path, and the A4 reference checks. The checks cover the visible paper corners, image and paper size, boundary quality, and camera-pose diagnostics. It does not detect a foot or validate a final foot measurement.

For the lower-level corner candidate without the project's acceptance decision:

```python
from a4_detector import detect_a4_paper

detection = detect_a4_paper(image)
print(detection.success, detection.corners, detection.message)
```

Core detection success does not mean that the sheet passed the reference checks. See [Integration](docs/INTEGRATION.md) for the result fields and [Algorithm](docs/ALGORITHM.md) for the processing steps and source components.

The paper must be a flat A4 sheet (210 × 297 mm). Images alone cannot distinguish A4 from another sheet with the same proportions. Shadows, glare, blur, occlusion, pale or patterned surfaces, and strong perspective can affect the result. The checks are engineering limits, not a guarantee of accuracy for every camera or scene. Evaluate the detector on the photos you intend to use.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

See [Contributing](CONTRIBUTING.md) for the project conventions.

## License

[MIT](LICENSE). OpenCV and NumPy retain their respective licenses.
