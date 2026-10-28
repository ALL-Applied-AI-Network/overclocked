"""Cosmos3-Nano as a small HTTP service, run on the Spark in the cosmos venv.

    source /workspace/saai/env.sh
    "$COSMOS" cosmos_server.py            # loads the model once, then serves on 127.0.0.1:8765

POST /understand   {"image_b64": <png/jpg bytes, base64>, "instruction": "..."}
  -> {"objects": [{"category", "colour"}], "pick_colour": "red" | null, "raw": "<model reply>"}
POST /ask          {"image_b64": <png/jpg bytes, base64>, "prompt": "...", "max_new_tokens": 300}
  -> {"raw": "<model reply>", "json": <the reply parsed as JSON, or null>}

The Mac reaches it through an SSH tunnel (ssh -N -L 8765:127.0.0.1:8765 ...), never directly.
"""

import base64
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.environ.get("FANUC_SIM", "/workspace/saai/fanuc_er4ia_sim"))
from perceive import ask, canonicalise, load_model, parse_json  # noqa: E402

COLOURS = ("yellow", "blue", "red")

PROMPT = """You are the vision system of a robot arm looking down at a table.
Task given to the robot: "{instruction}"

List every small object on the table. Reply with JSON only:
{{"objects": [{{"category": "cube|cap|tube|other", "colour": "yellow|blue|red|other"}}],
  "pick_colour": "yellow|blue|red|null"}}
"pick_colour" is the colour of the cube the task asks the robot to pick up, or null if the task
does not single out a colour (a colour word describing a bin or container is not a cube colour)."""

lock = threading.Lock()  # one generation at a time on the GPU
model, processor = load_model()


def understand(image: bytes, instruction: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(image)
        f.flush()
        with lock:
            raw = ask(model, processor, f.name, PROMPT.format(instruction=instruction), 500)
    parsed = canonicalise(parse_json(raw) or {})
    pick = parsed.get("pick_colour")
    return {
        "objects": [o for o in parsed.get("objects", []) if isinstance(o, dict)],
        "pick_colour": pick if pick in COLOURS else None,
        "raw": raw,
    }


def ask_free(image: bytes, prompt: str, max_new_tokens: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(image)
        f.flush()
        with lock:
            raw = ask(model, processor, f.name, prompt, max_new_tokens)
    return {"raw": raw, "json": parse_json(raw)}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            image = base64.b64decode(body["image_b64"])
            if self.path == "/ask":
                out = ask_free(image, str(body["prompt"]), int(body.get("max_new_tokens", 300)))
            else:
                out = understand(image, str(body.get("instruction", "")))
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
    print("[cosmos_server] ready on 127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
