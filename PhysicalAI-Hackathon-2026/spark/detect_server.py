"""Open-vocabulary object detection (OWLv2) as a small HTTP service, run on the Spark in the cosmos venv.

    source /workspace/saai/env.sh
    "$COSMOS" detect_server.py            # loads the model once, then serves on 127.0.0.1:8766

POST /detect   {"image_b64": <png bytes, base64>, "queries": ["a small blue cube", ...],
                "threshold": 0.2, "region": [x0, y0, x1, y1]}
  -> {"detections": [{"label", "score", "box": [x0, y0, x1, y1]}]}   boxes in full-image pixels

The image is cut into overlapping tiles so a 10 mm cube (~18 px in the 1920x1080 table image) is not shrunk away.
The Mac reaches it through an SSH tunnel (ssh -N -L 8766:127.0.0.1:8766 ...).
"""

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

NAME = "google/owlv2-base-patch16-ensemble"
TILE, STRIDE = 640, 480
DEFAULT_REGION = (150, 330, 1920, 1080)  # x0, y0, x1, y1 of the table in the 1920x1080 image, without the arm and wall

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # the Mac runs it on the CPU for offline tests
PORT = int(os.environ.get("DETECT_PORT", "8766"))
lock = threading.Lock()  # one tile batch at a time on the GPU
processor = Owlv2Processor.from_pretrained(NAME)
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
model = Owlv2ForObjectDetection.from_pretrained(NAME, torch_dtype=DTYPE).to(DEVICE).eval()


def detect(rgb: np.ndarray, queries: list[str], threshold: float, region) -> list[dict]:
    x0, y0, x1, y1 = region
    found = []
    for ty in range(y0, max(y1 - 200, y0 + 1), STRIDE):
        for tx in range(x0, max(x1 - 200, x0 + 1), STRIDE):
            crop = rgb[ty:ty + TILE, tx:tx + TILE]
            if crop.shape[0] < 200 or crop.shape[1] < 200:
                continue
            size = max(crop.shape[:2])
            inputs = processor(text=[queries], images=Image.fromarray(crop), return_tensors="pt")
            inputs = {k: (v.to(DEVICE, DTYPE) if v.dtype == torch.float32 else v.to(DEVICE)) for k, v in inputs.items()}
            with lock, torch.inference_mode():
                out = model(**inputs)
            res = processor.post_process_grounded_object_detection(
                out, threshold=threshold, target_sizes=torch.tensor([[size, size]]), text_labels=[queries])[0]
            for box, score, label in zip(res["boxes"], res["scores"], res["text_labels"]):
                b = box.float().tolist()
                found.append({"label": label, "score": float(score), "box": [b[0] + tx, b[1] + ty, b[2] + tx, b[3] + ty]})
    found.sort(key=lambda d: -d["score"])  # the same object turns up in neighbouring tiles: keep the best
    keep = []
    for d in found:
        cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
        if all(k["label"] != d["label"] or np.hypot(cx - (k["box"][0] + k["box"][2]) / 2,
                                                    cy - (k["box"][1] + k["box"][3]) / 2) > 20 for k in keep):
            keep.append(d)
    return keep


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            png = np.frombuffer(base64.b64decode(body["image_b64"]), np.uint8)
            rgb = cv2.cvtColor(cv2.imdecode(png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            out = {"detections": detect(rgb, list(body["queries"]), float(body.get("threshold", 0.2)),
                                        [int(v) for v in body.get("region", DEFAULT_REGION)])}
            code = 200
        except Exception as e:  # noqa: BLE001
            out, code = {"error": repr(e)}, 500
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # health check
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


if __name__ == "__main__":
    print(f"[detect_server] ready on 127.0.0.1:{PORT} ({DEVICE})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
