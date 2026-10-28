"""
server.py
---------
Local HTTP API for the frontend demo (Python standard library only, binds to
127.0.0.1). Every prediction is computed by the saved demo checkpoint at
request time; if the checkpoint or data is missing the API returns an error,
never placeholder numbers.

    GET /api/health
    GET /api/overview                      project, dataset card, split, model
    GET /api/patients?split=test
    GET /api/predict?patient_id=90000507
    GET /api/predict?new_patient=7&hospital=synth_hospital_06
    GET /api/federation                    per-round FedAvg log of the demo training
    GET /api/evaluation                    synthetic demo metrics + recorded eICU results (kept separate)
    GET /api/graph                         demo-model patient graph of a synthetic test hospital

If frontend-playground/dist/ exists (npm run build), it is served at /.
"""

import json
import mimetypes
import pathlib
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from demo import SYNTHETIC_NOTICE
from demo import pipeline as dp

PROJECT_ROOT = dp.PROJECT_ROOT
DIST_DIR = PROJECT_ROOT / "frontend-playground" / "dist"
HISTORICAL = PROJECT_ROOT / "results" / "corrected" / "benchmark_results.json"

_lock = threading.Lock()
_model = None


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def model():
    global _model
    if _model is None:
        try:
            _model = dp.load_model()
        except (FileNotFoundError, ValueError) as e:
            raise ApiError(503, f"{e} (then restart the server)")
    return _model


def read_json(path: pathlib.Path, hint: str):
    if not path.exists():
        raise ApiError(503, f"{path.relative_to(PROJECT_ROOT)} not found. {hint}")
    with open(path) as f:
        return json.load(f)


def overview() -> dict:
    card = read_json(dp.paths()["card"], "Run: python scripts/run_demo.py generate")
    m = model()
    return {
        "project": "Federated Multimodal Graph Learning for Early In-Hospital and ICU Mortality Prediction",
        "purpose": ("Predict in-hospital and ICU mortality from the first 24 hours of an ICU stay, using tabular and "
                    "time-series data, a patient-similarity graph (GraphSAGE) and federated training (FedAvg) across hospitals."),
        "notice": SYNTHETIC_NOTICE,
        "data_source": "synthetic",
        "dataset": card,
        "split": {k: [h.name for h in v] for k, v in dp.demo_split().items()},
        "model": m.checkpoint_info,
        "pipeline": [
            {"step": "Tabular encoder (MLP)", "shape": "67 -> 64", "module": "src/encoders.py TabularEncoder"},
            {"step": "Time-series encoder (2-layer LSTM)", "shape": "6 bins x 74 -> 64", "module": "src/encoders.py TimeSeriesLSTM"},
            {"step": "Multimodal fusion", "shape": "64 + 64 -> 128", "module": "src/fusion.py MultimodalFusion"},
            {"step": "k-NN patient graph (k=5, cosine, within a batch of one hospital)", "shape": "batch -> edges",
             "module": "src/graph_builder.py build_knn_graph"},
            {"step": "GraphSAGE (2 layers)", "shape": "128 -> 64", "module": "src/gnn.py PatientGNN"},
            {"step": "Two prediction heads", "shape": "64 -> 1 + 1", "module": "src/models.py MultiTaskPredictionHeads"},
        ],
        "scope_limits": [
            "No clinical text encoder: the implemented model uses tabular + time-series inputs only.",
            "Targets are in-hospital and ICU mortality; there is no sepsis prediction head.",
            "No centralized all-data baseline; baselines are FedAvg without GNN and local-only training.",
            "Federation is simulated in one process; no secure aggregation or differential privacy.",
        ],
    }


def evaluation() -> dict:
    demo = {}
    for name in ("demo_fedavg_gnn", "demo_fedavg_nognn"):
        path = dp.RESULTS_DIR / f"{name}_summary.json"
        if path.exists():
            with open(path) as f:
                s = json.load(f)
            demo[name] = {"test_metrics": s["test_metrics"], "untrained_val_metrics": s["untrained_val_metrics"],
                          "selected_round": s["selected_round"], "split": s["split"]["test"]}
    if not demo:
        raise ApiError(503, "No demo results yet. Run: python scripts/run_demo.py all")
    historical = None
    if HISTORICAL.exists():
        with open(HISTORICAL) as f:
            h = json.load(f)
        historical = {"source": str(HISTORICAL.relative_to(PROJECT_ROOT)), "protocol": h.get("protocol"),
                      "summary": h.get("summary")}
    return {
        "synthetic_demo": {"data_source": "synthetic", "notice": SYNTHETIC_NOTICE,
                           "description": "Computed now from the demo models' predictions on synthetic test hospitals.",
                           "runs": demo},
        "historical_eicu": {"description": ("Previously recorded corrected eICU experiment (3 seeds, 31 test hospitals). "
                                            "Preserved file; not re-run in this demo, and its checkpoints are not in the repository."),
                            "data": historical},
    }


def predict(q: dict) -> dict:
    get = lambda k: q.get(k, [None])[0]
    threshold = float(get("threshold") or dp.THRESHOLD)
    try:
        with _lock:
            if get("new_patient") is not None:
                return dp.predict(model(), new_patient_seed=int(get("new_patient")), hospital=get("hospital"), threshold=threshold)
            pid = get("patient_id")
            if pid is None:
                raise ApiError(400, "Give patient_id or new_patient")
            return dp.predict(model(), patient_id=int(str(pid).removeprefix("SYN-")), threshold=threshold)
    except KeyError as e:
        raise ApiError(404, str(e).strip("'\""))
    except ValueError as e:
        raise ApiError(400, str(e))


def route(path: str, q: dict):
    if path == "/api/health":
        return {"status": "ok", "data_source": "synthetic", "checkpoint": dp.CHECKPOINT.exists(),
                "dataset": dp.paths()["card"].exists()}
    if path == "/api/overview":
        return overview()
    if path == "/api/patients":
        return {"data_source": "synthetic", "patients": dp.list_patients(which=q.get("split", ["test"])[0])}
    if path == "/api/predict":
        return predict(q)
    if path == "/api/federation":
        s = read_json(dp.RESULTS_DIR / "demo_fedavg_gnn_summary.json", "Run: python scripts/run_demo.py train")
        return {"data_source": "synthetic", "notice": SYNTHETIC_NOTICE, "config": s["config"], "split": s["split"],
                "selected_round": s["selected_round"], "history": s["history"]}
    if path == "/api/evaluation":
        return evaluation()
    if path == "/api/graph":
        return read_json(dp.RESULTS_DIR / "demo_graph.json", "Run: python scripts/run_demo.py train")
    raise ApiError(404, f"Unknown endpoint {path}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            try:
                body, status = route(url.path, parse_qs(url.query)), 200
            except ApiError as e:
                body, status = {"error": str(e)}, e.status
            except Exception as e:  # report, do not invent a result
                traceback.print_exc()
                body, status = {"error": f"{type(e).__name__}: {e}"}, 500
            self._send(status, json.dumps(body).encode(), "application/json")
            return
        self._static(url.path)

    def do_HEAD(self) -> None:  # liveness probes
        self.send_response(200)
        self.end_headers()

    def _static(self, path: str) -> None:
        if not DIST_DIR.is_dir():
            msg = ("Frontend not built. Either run 'npm run build' in frontend-playground/ and restart this server, "
                   "or run 'npm run dev' there and open http://localhost:5173 (it proxies /api to this server).")
            self._send(200, msg.encode(), "text/plain; charset=utf-8")
            return
        target = (DIST_DIR / path.lstrip("/")).resolve()
        if DIST_DIR.resolve() not in target.parents or not target.is_file():
            target = DIST_DIR / "index.html"  # single-page app
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(200, target.read_bytes(), ctype)

    def log_message(self, fmt, *args) -> None:
        print(f"  [api] {self.address_string()} {fmt % args}", flush=True)


def serve(port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"\n {SYNTHETIC_NOTICE}")
    print(f" Demo API on http://127.0.0.1:{port}/api/health")
    print(f" Frontend : http://127.0.0.1:{port}/" if DIST_DIR.is_dir()
          else " Frontend : not built - run 'npm run dev' in frontend-playground/ and open http://localhost:5173")
    print(" Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n stopped")
    finally:
        server.server_close()
