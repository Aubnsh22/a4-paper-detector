"""Generate the synthetic sample; no real capture data are used."""
from pathlib import Path

import cv2
import numpy as np

image = np.full((960, 1280, 3), 65, dtype=np.uint8)
corners = np.array([[160, 180], [475, 180], [475, 626], [160, 626]], dtype=np.int32)
cv2.fillConvexPoly(image, corners, (245, 245, 245))
path = Path(__file__).parent / "images/synthetic_a4.png"
path.parent.mkdir(parents=True, exist_ok=True)
ok, data = cv2.imencode(".png", image)
if not ok:
    raise RuntimeError("Could not encode the sample image.")
path.write_bytes(data.tobytes())
