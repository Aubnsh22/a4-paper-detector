"""Run from the repository root after installing the package."""
import json
from pathlib import Path

import cv2
import numpy as np

from a4_detector import detect_a4

image_path = Path(__file__).parent / "images/synthetic_a4.png"
image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
result = detect_a4(image)
print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
