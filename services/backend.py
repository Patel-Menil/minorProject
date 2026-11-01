"""
backend.py
----------
Container entry point of the "backend" service (port 8000, Docker network only).
Standard library only (no model code, no data access).

    GET  /api/health                 backend up AND model service ready (calls model /health)
    POST /api/sepsis/predict-input   validates the form payload against the model's input schema
                                     (feature names, 6 bins, numbers, 0/1 fields, training ranges),
                                     then forwards it to the model service
    GET  /api/*                      forwarded unchanged to the model service

Errors from the model service are passed through with their status; if the model service
cannot be reached the answer is 503 with a clear message. Logs contain method, path and
status only - never request bodies, query strings, feature values or identifiers.
"""

import json
import math
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

MODEL_URL = os.environ.get("MODEL_URL", "http://model:8001").rstrip("/")
TIMEOUT = float(os.environ.get("MODEL_TIMEOUT", "120"))
_schema, _schema_lock = None, threading.Lock()


class Fail(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def call_model(method: str, path: str, body: bytes = None) -> tuple[int, bytes]:
    req = urllib.request.Request(MODEL_URL + path, data=body, method=method,
                                 headers={"Content-Type": "application/json"} if body is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        raise Fail(503, f"Model service unavailable at {MODEL_URL} ({getattr(e, 'reason', e)})")


def schema() -> dict:
    global _schema
    with _schema_lock:
        if _schema is None:
            status, body = call_model("GET", "/api/sepsis/input-schema")
            if status != 200:
                raise Fail(status, json.loads(body or b"{}").get("error", "model input schema unavailable"))
            _schema = json.loads(body)
        return _schema


def validate(payload) -> None:
    """Same rules as src/interactive_inference.py; the model service validates again."""
    if not isinstance(payload, dict):
        raise Fail(400, "Invalid input: JSON object expected")
    inputs = payload.get("inputs", payload)
    if not isinstance(inputs, dict) or not isinstance(inputs.get("tabular"), dict) or not isinstance(inputs.get("timeseries"), dict):
        raise Fail(400, "Invalid input: payload needs 'tabular' and 'timeseries' objects")
    s, errors = schema(), []
    thr = payload.get("threshold", 0.5)
    if not isinstance(thr, (int, float)) or isinstance(thr, bool) or not 0 < thr < 1:
        errors.append("threshold must be a number between 0 and 1")

    def check(spec, v, where):
        if v is None:
            if spec["kind"] != "lab":
                errors.append(f"{where}: value required")
            return
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errors.append(f"{where}: must be a finite number")
        elif spec["kind"] == "binary" and v not in (0, 1):
            errors.append(f"{where}: must be 0 or 1")
        elif not spec["min"] <= v <= spec["max"]:
            errors.append(f"{where}: {v:g} is outside the training range {spec['min']:g} to {spec['max']:g}")

    names = {sp["name"] for sp in s["tabular"]}, {sp["name"] for sp in s["timeseries"]}
    extra = sorted(set(inputs["tabular"]) - names[0]) + sorted(set(inputs["timeseries"]) - names[1])
    if extra:
        errors.append(f"unknown features: {extra[:5]}")
    for sp in s["tabular"]:
        if sp["name"] not in inputs["tabular"]:
            errors.append(f"{sp['name']}: missing")
        else:
            check(sp, inputs["tabular"][sp["name"]], sp["name"])
    for sp in s["timeseries"]:
        vals = inputs["timeseries"].get(sp["name"])
        if not isinstance(vals, list) or len(vals) != len(s["bins"]):
            errors.append(f"{sp['name']}: needs {len(s['bins'])} values")
            continue
        for b, v in enumerate(vals):
            check(sp, v, f"{sp['name']} [{s['bins'][b]}]")
    if errors:
        raise Fail(400, "Invalid input: " + "; ".join(errors[:8]) + (f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""))


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, json.dumps({"error": message}).encode())

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                status, body = call_model("GET", "/health")
                model = json.loads(body or b"{}")
                if status != 200:
                    raise Fail(503, f"Model service not ready: {model.get('error', status)}")
                self._send(200, json.dumps({"status": "ok", "backend": "ready", "model": model}).encode())
                return
            if not path.startswith("/api/"):
                raise Fail(404, "Unknown endpoint")
            status, body = call_model("GET", self.path)
            self._send(status, body)
        except Fail as e:
            self._error(e.status, str(e))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                raise Fail(413, "Request too large")
            raw = self.rfile.read(n)
            if path != "/api/sepsis/predict-input":
                raise Fail(404, "Unknown endpoint")
            try:
                payload = json.loads(raw or b"null")
            except ValueError:
                raise Fail(400, "Invalid input: body is not valid JSON")
            validate(payload)
            status, body = call_model("POST", path, json.dumps(payload).encode())
            self._send(status, body)
        except Fail as e:
            self._error(e.status, str(e))

    def log_message(self, fmt, *args) -> None:
        line = fmt % args
        print(f"  [backend] {line.split('?', 1)[0]}", flush=True)  # no query strings / bodies


def main() -> None:
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    print(f" backend: listening on 0.0.0.0:{port}, model service {MODEL_URL}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
