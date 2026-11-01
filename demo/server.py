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
    GET /api/final/status                  FINAL real-data model: manifest + held-out test evaluation
    GET /api/final/predict?test_index=0    FINAL model prediction for a masked held-out test patient
    GET /api/sepsis/status, /api/sepsis/predict   the separate mortality + sepsis model (results/minor_project_sepsis/)
    GET  /api/sepsis/input-schema          form fields (training-split defaults / bounds), example, model info
    POST /api/sepsis/predict-input         form values -> the verified mortality + sepsis checkpoint
                                           (src/interactive_inference.py); invalid input -> 400, no fallback model

The /api/final/* endpoints use only results/minor_project_final/ (scripts/final_model.py).
If that model does not exist they return an error; they never use the demo model.

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
_final = None
FINAL_ROOT = PROJECT_ROOT / "results" / "minor_project_final"  # set by serve(final_root=...)


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


_real_models = {}  # "final" (mortality + ICU mortality) / "sepsis" (mortality + sepsis), loaded on first use


def real_root(kind: str) -> pathlib.Path:
    from scripts.final_model import SEPSIS_ROOT
    return FINAL_ROOT if kind == "final" else SEPSIS_ROOT


def final_model(kind: str = "final"):
    """A real-data model (scripts/final_model.py); errors are passed through, never replaced."""
    if kind not in _real_models:
        from scripts.final_model import FinalModel, FinalModelError
        try:
            _real_models[kind] = FinalModel(real_root(kind))
        except (FinalModelError, FileNotFoundError, ValueError) as e:
            raise ApiError(503, str(e))
    return _real_models[kind]


MANIFEST_KEYS = ("title", "task_id", "targets", "data_dir", "split", "data_counts", "task", "config", "preprocessing",
                 "primary_model", "models", "written_at", "label_definition", "label_counts_24h_cohort",
                 "removed_tabular_features", "tabular_features_used", "limitations")


def final_status(kind: str = "final") -> dict:
    m = final_model(kind)
    return {"data_source": m.manifest["data_source"], "synthetic": m.manifest["synthetic"],
            "manifest": {k: m.manifest[k] for k in MANIFEST_KEYS if k in m.manifest},
            "task_id": m.task, "second_key": m.spec["second"], "second_label": m.spec["second_label"],
            "test_patients": m.n_test}


def final_predict(q: dict, kind: str = "final") -> dict:
    from scripts.final_model import FinalModelError
    try:
        with _lock:
            return final_model(kind).predict(int(q.get("test_index", ["0"])[0]), float(q.get("threshold", ["0.5"])[0]))
    except (FinalModelError, ValueError) as e:
        raise ApiError(400, str(e))


_interactive = None
SEPSIS_ROOT_OVERRIDE = None  # tests may point the interactive model at another folder


def interactive_model():
    """Verified mortality + sepsis checkpoint for form inputs; errors are passed through, never replaced."""
    global _interactive
    if _interactive is None:
        from src.interactive_inference import SEPSIS_ROOT, InteractiveModel
        try:
            _interactive = InteractiveModel(SEPSIS_ROOT_OVERRIDE or SEPSIS_ROOT)
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise ApiError(503, f"Mortality + sepsis checkpoint not available: {e}")
    return _interactive


def predict_input(body: dict) -> dict:
    from src.interactive_inference import InputError
    m = interactive_model()
    try:
        with _lock:
            return m.predict(body.get("inputs", body), float(body.get("threshold", 0.5)))
    except InputError as e:
        raise ApiError(400, f"Invalid input: {e}")
    except (TypeError, ValueError) as e:
        raise ApiError(400, f"Invalid input: {e}")


def route_post(path: str, body) -> dict:
    if path == "/api/sepsis/predict-input":
        if not isinstance(body, dict):
            raise ApiError(400, "Invalid input: JSON object expected")
        return predict_input(body)
    raise ApiError(404, f"Unknown endpoint {path}")


def model_health() -> dict:
    """Readiness of real-model inference: checkpoint found, SHA-256 valid, architecture and graph context loaded."""
    m = interactive_model()  # raises ApiError(503) with the reason if anything is missing
    return {"status": "ready", "checkpoint_found": m.checkpoint.exists(), "checkpoint_sha256_valid": True,
            "checkpoint_sha256_prefix": m.sha[:12], "checkpoint": m.entry["checkpoint"], "checkpoint_round": m.ckpt_meta["round"],
            "architecture_loaded": True, "targets": m.ckpt_meta["targets"],
            "inference_context_ready": int(m.context[0].shape[0]) == 31, "ready": True}


def route(path: str, q: dict):
    if path in ("/health", "/api/model/health"):
        return model_health()
    if path == "/api/sepsis/input-schema":
        return interactive_model().schema()
    for kind in ("final", "sepsis"):
        if path == f"/api/{kind}/status":
            return final_status(kind)
        if path == f"/api/{kind}/predict":
            return final_predict(q, kind)
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
        if url.path.startswith("/api/") or url.path == "/health":
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

    def do_POST(self) -> None:
        url = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                raise ApiError(413, "Request too large")
            try:
                body = json.loads(self.rfile.read(n) or b"null")
            except ValueError:
                raise ApiError(400, "Invalid input: body is not valid JSON")
            body, status = route_post(url.path, body), 200
        except ApiError as e:
            body, status = {"error": str(e)}, e.status
        except Exception as e:  # report, do not invent a result
            traceback.print_exc()
            body, status = {"error": f"{type(e).__name__}: {e}"}, 500
        self._send(status, json.dumps(body).encode(), "application/json")

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
        # method, path and status only: no query strings or request bodies in the logs
        line = fmt % args
        print(f"  [api] {self.address_string()} {line.split('?', 1)[0] + ('...' if '?' in line else '')}", flush=True)


def serve(port: int = 8765, final_root: pathlib.Path = None, host: str = "127.0.0.1") -> None:
    global FINAL_ROOT
    if final_root is not None:
        FINAL_ROOT = pathlib.Path(final_root)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"\n Synthetic demo mode: {SYNTHETIC_NOTICE}")
    manifest = FINAL_ROOT / "final_model_manifest.json"
    print(f" Final model mode   : {manifest if manifest.exists() else f'NOT AVAILABLE ({manifest} missing)'}")
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
