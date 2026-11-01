"""
test_docker_services.py
-----------------------
The backend container's API (services/backend.py) against an in-process model service
(demo/server.py with the REAL mortality + sepsis checkpoint): health, forwarding, validation
and the "model service unavailable" error. Skipped without the local checkpoint / data.

Usage
-----
    python -m pytest tests/test_docker_services.py -v
"""

import importlib
import json
import pathlib
import socket
import sys
import threading
import urllib.error
import urllib.request

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from http.server import ThreadingHTTPServer

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (PROJECT_ROOT / "results" / "minor_project_sepsis" / "final_model_manifest.json").exists()
    or not (PROJECT_ROOT / "processed" / "sepsis_labels.parquet").exists(),
    reason="needs the trained mortality + sepsis checkpoint and the local eICU data (git-ignored)")


def start(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def request(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def stack():
    from demo import server
    from src.interactive_inference import InteractiveModel
    server._interactive = InteractiveModel()
    model_httpd, model_url = start(server.Handler)
    import services.backend as backend
    backend = importlib.reload(backend)
    backend.MODEL_URL = model_url
    backend_httpd, backend_url = start(backend.Handler)
    yield {"backend": backend, "url": backend_url, "model": server._interactive}
    backend_httpd.shutdown(); model_httpd.shutdown()
    server._interactive = None


def baseline(m):
    d = m.defaults()
    return {"tabular": dict(d["tabular"]), "timeseries": {k: list(v) for k, v in d["timeseries"].items()}}


def test_health_reports_model_readiness(stack):
    status, body = request(stack["url"] + "/api/health")
    assert status == 200 and body["backend"] == "ready"
    m = body["model"]
    assert m["ready"] and m["checkpoint_sha256_valid"] and m["architecture_loaded"] and m["inference_context_ready"]
    assert m["targets"] == ["hospital_expired", "sepsis_after_24h"]


def test_prediction_is_forwarded_to_the_real_model(stack):
    p = baseline(stack["model"])
    p["tabular"]["apachescore"] = 130
    status, body = request(stack["url"] + "/api/sepsis/predict-input", {"inputs": p, "threshold": 0.5})
    assert status == 200 and body["source"] == "trained_real_data_checkpoint"
    assert body["targets"] == stack["model"].predict(p)["targets"]  # same numbers as the model itself
    status, schema = request(stack["url"] + "/api/sepsis/input-schema")  # GET routes are forwarded too
    assert status == 200 and len(schema["tabular"]) == 28


def test_backend_rejects_invalid_payloads_before_the_model(stack):
    for bad in ({"inputs": {"tabular": {}, "timeseries": {}}},  # every feature missing
                {"inputs": {**baseline(stack["model"]), "tabular": {**baseline(stack["model"])["tabular"], "age": 999}}},
                {"inputs": baseline(stack["model"]), "threshold": 2},
                "a string"):
        status, body = request(stack["url"] + "/api/sepsis/predict-input", bad)
        assert status == 400 and body["error"].startswith("Invalid input")


def test_model_service_unavailable_gives_503(stack):
    with socket.socket() as s:  # a port with nothing listening
        s.bind(("127.0.0.1", 0)); dead = s.getsockname()[1]
    backend = stack["backend"]
    saved, backend.MODEL_URL = backend.MODEL_URL, f"http://127.0.0.1:{dead}"
    try:
        status, body = request(stack["url"] + "/api/health")
        assert status == 503 and "unavailable" in body["error"]
        status, body = request(stack["url"] + "/api/final/status")
        assert status == 503 and "Model service unavailable" in body["error"]
    finally:
        backend.MODEL_URL = saved
