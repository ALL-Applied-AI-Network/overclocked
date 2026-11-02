"""Held/empty verifier service (C-RADIOv2-B + logistic head) on the Spark, 127.0.0.1:8767.
POST /held {"image_b64": png of the 240x240 fingertip crop (RGB)}  ->  {"p_held": float, "ms": float}"""
import base64, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, torch
from transformers import AutoModel
dev = "cuda"
model = AutoModel.from_pretrained("nvidia/C-RADIOv2-B", trust_remote_code=True).eval().to(dev)
H = np.load("head_cradio.npz"); mu, sd, w, b = H["mu"], H["sd"], H["w"], float(H["b"][0])
lock = threading.Lock()
@torch.no_grad()
def p_held(png: bytes) -> float:
    im = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy((cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(dev)
    with lock, torch.autocast("cuda", dtype=torch.float16):
        f = model(x)[0].float().flatten().cpu().numpy()
    z = float(((f - mu) / sd) @ w + b); return 1.0 / (1.0 + np.exp(-z))
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"]))); t = time.time()
            out = {"p_held": p_held(base64.b64decode(body["image_b64"])), "ms": 0.0}; out["ms"] = (time.time() - t) * 1000; code = 200
        except Exception as e:  # noqa: BLE001
            out, code = {"error": repr(e)}, 500
        d = json.dumps(out).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
if __name__ == "__main__":
    p_held(cv2.imencode(".png", np.zeros((240, 240, 3), np.uint8))[1].tobytes())  # warm up
    print("[verifier] ready on 127.0.0.1:8767", flush=True); ThreadingHTTPServer(("127.0.0.1", 8767), Handler).serve_forever()
