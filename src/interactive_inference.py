"""
interactive_inference.py
------------------------
Runs the trained mortality + sepsis checkpoint on a HYPOTHETICAL patient typed into
the frontend form.

Preprocessing (identical to training; checked against the loader in tests):
  tabular (28 features, checkpoint order) : z-score with the cohort statistics of one fixed
        reference TRAINING hospital, exactly as src.datasets z-scores each hospital's own patients
  labs (20 time-series features)          : raw value in the unit recorded in eICU lab.csv ->
        (value - global mean) / global SD from processed/timeseries_stats.json, which is how the
        stored Parquet values were produced (mean of the results in each 4-hour bin); an empty
        field = "not measured" = 0, as in the stored data
  infusion / medication / treatment (54)  : 0/1 "recorded in this 4-hour bin", as stored
  window                                   : 6 bins of 4 h (0-24 h), sequence length 6

Graph context: the model builds its k-NN graph inside a batch of 32 patients of one hospital.
The hypothetical patient is placed in a batch with the first 31 training patients (loader order)
of the same reference training hospital, so GraphSAGE sees a real batch of the kind it was
trained on. Only training-split data is used; no reference record leaves the backend.

Form defaults and validation bounds are computed from training-split cohort patients only:
default = median; allowed range = 0.1st-99.9th percentile of the recorded values (the range the
model was trained on); 0/1 features accept 0 or 1. The cache (interactive_reference.json) holds
only these aggregates, no patient rows.
"""

import hashlib
import json
import pathlib
from typing import Optional

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SEPSIS_ROOT = PROJECT_ROOT / "results" / "minor_project_sepsis"
REFERENCE_FILE = "interactive_reference.json"
N_BINS = 6
CONTEXT = 31
BIN_LABELS = [f"{4 * b}-{4 * b + 4} h" for b in range(N_BINS)]

QUICK_TABULAR = ["age", "gender_Male", "BMI", "acutephysiologyscore", "apachescore", "verbal", "motor", "eyes",
                 "pao2", "fio2", "creatinine", "diabetes", "cirrhosis", "metastaticcancer", "immunosuppression"]
QUICK_SERIES = ["lab_bun", "lab_creatinine", "lab_bicarbonate", "lab_wbc x 1000", "lab_platelets x 1000", "lab_sodium",
                "lab_potassium", "lab_glucose", "lab_hgb", "infusion_norepinephrine (mcg/min)",
                "treatment_infectious diseases", "treatment_pulmonary"]
# Meanings documented in tabular_data_handoff.md (the only documented units for tabular fields)
TABULAR_DOCS = {
    "age": "years; ages > 89 are recorded as 90",
    "gender_Male": "1 = male, 0 = female / unknown",
    "BMI": "kg/m² (weight / height²)",
    "hospitaladmitoffset": "minutes from hospital admission to ICU admission (negative = before)",
    "unitvisitnumber": "ICU visit number in this hospital stay",
    "acutephysiologyscore": "APACHE IV acute physiology score",
    "apachescore": "APACHE IV score",
    "predictedicumortality": "APACHE IV predicted ICU mortality",
    "predictediculos": "APACHE IV predicted ICU length of stay",
    "verbal": "GCS verbal component (day 1)", "motor": "GCS motor component (day 1)", "eyes": "GCS eyes component (day 1)",
    "pao2": "day-1 PaO2 (APACHE variable)", "fio2": "day-1 FiO2 (APACHE variable)", "creatinine": "day-1 creatinine (APACHE variable)",
}
# Hypothetical demonstration scenario (not a patient record): an older patient with a high APACHE score,
# rising BUN / creatinine, falling bicarbonate and platelets, high WBC and vasopressor support from 8 h.
EXAMPLE = {
    "tabular": {"age": 78, "gender_Male": 1, "BMI": 24.5, "acutephysiologyscore": 88, "apachescore": 104, "verbal": 3,
                "motor": 5, "eyes": 3, "pao2": 72, "fio2": 60, "creatinine": 2.4, "diabetes": 1, "immunosuppression": 1},
    "timeseries": {
        "lab_bun": [38, 42, 47, 51, 56, 60], "lab_creatinine": [2.1, 2.3, 2.5, 2.7, 2.9, 3.1],
        "lab_bicarbonate": [21, 20, 19, 18, 17, 16], "lab_wbc x 1000": [15.5, 16.8, 17.9, 18.6, 19.4, 20.1],
        "lab_platelets x 1000": [180, 165, 150, 138, 126, 115], "lab_sodium": [136, 135, 135, 134, 134, 133],
        "lab_potassium": [4.6, 4.8, 5.0, 5.1, 5.2, 5.3], "lab_glucose": [165, 172, 180, 188, 176, 170],
        "lab_hgb": [10.8, 10.5, 10.2, 10.0, 9.8, 9.6],
        "infusion_norepinephrine (mcg/min)": [0, 0, 1, 1, 1, 1],
        "treatment_infectious diseases": [1, 1, 1, 1, 1, 1], "treatment_pulmonary": [0, 1, 1, 1, 1, 1],
    },
}


class InputError(ValueError):
    """Invalid form payload; .errors lists every problem."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:8]) + (f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""))
        self.errors = errors


def portable(rel: str) -> str:
    """Manifest paths were written on Windows (backslashes); make them work on Linux (Docker) too."""
    return str(rel).replace("\\", "/")


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _label(name: str) -> str:
    for p in ("lab_", "infusion_", "med_", "treatment_"):
        if name.startswith(p):
            return f"{p[:-1]}: {name[len(p):]}"
    return name


# ── reference (training split only) ─────────────────────────────────────────
def build_reference(data_dir: pathlib.Path, tabular_features: list[str], checkpoint_sha: str) -> dict:
    import pandas as pd

    data_dir = pathlib.Path(data_dir)
    labels = pd.read_parquet(data_dir / "sepsis_labels.parquet")
    cohort = labels[(labels["split"] == "train") & labels["label"].notna() & (labels["unitdischargeoffset"] >= 24 * 60)]
    by_h = cohort.groupby("hospital")["patientunitstayid"].apply(set)
    ref_hospital = cohort["hospital"].value_counts().index[0]  # largest training hospital (deterministic)
    ts_cols = json.loads((data_dir / "canonical_timeseries_features.json").read_text())
    stats = json.loads((data_dir / "timeseries_stats.json").read_text())

    tab_parts, ts_parts, units = [], [], {}
    for h, pids in by_h.items():
        hd = data_dir / "hospitals" / h
        t = pd.read_parquet(hd / "tabular" / "tabular_features.parquet")
        t = t[t["patientunitstayid"].isin(pids)]
        for c in tabular_features:
            if c not in t:
                t[c] = 0.0
        tab_parts.append(t[tabular_features].astype(float).fillna(0.0))
        s = pd.read_parquet(hd / "timeseries" / "timeseries_features.parquet")
        s = s[s["patientunitstayid"].isin(pids) & (s["hour_bin"] < N_BINS)]
        for c in ts_cols:
            if c not in s:
                s[c] = 0.0
        ts_parts.append(s[ts_cols].astype(float).fillna(0.0))
        lab_csv = hd / "timeseries" / "lab.csv"
        if lab_csv.exists():
            u = pd.read_csv(lab_csv, usecols=["labname", "labmeasurenamesystem"]).dropna()
            for (name, unit), n in u.groupby([u["labname"].str.lower(), "labmeasurenamesystem"]).size().items():
                units.setdefault(name, {}).setdefault(unit, 0)
                units[name][unit] += int(n)
    tab, ts = pd.concat(tab_parts), pd.concat(ts_parts)

    def spec(name, values, modality, lab=False):
        v = values.to_numpy(dtype=float)
        binary = bool(np.isin(v, [0.0, 1.0]).all())
        s = {"name": name, "label": _label(name), "modality": modality, "binary": binary,
             "quick": name in (QUICK_TABULAR if modality == "tabular" else QUICK_SERIES)}
        if lab:
            base = name[len("lab_"):]
            measured = v[v != 0.0] * stats[base]["std"] + stats[base]["mean"]  # stored z -> raw unit
            s.update({"kind": "lab", "unit": max(units.get(base, {"": 0}), key=units.get(base, {"": 0}).get) or None,
                      "default": round(float(np.median(measured)), 3),
                      "min": round(float(np.quantile(measured, 0.001)), 3), "max": round(float(np.quantile(measured, 0.999)), 3),
                      "measured_share": round(float((v != 0).mean()), 3), "empty_means": "not measured in this bin"})
        elif binary:
            s.update({"kind": "binary", "default": float(np.median(v)), "min": 0.0, "max": 1.0})
        else:
            s.update({"kind": "continuous", "default": round(float(np.median(v)), 3),
                      "min": round(float(np.quantile(v, 0.001)), 3), "max": round(float(np.quantile(v, 0.999)), 3)})
        if modality == "tabular" and name in TABULAR_DOCS:
            s["doc"] = TABULAR_DOCS[name]
        if modality == "timeseries" and not lab and binary:
            s["doc"] = "1 = recorded in this 4-hour bin, 0 = not recorded"
        return s

    return {
        "checkpoint_sha256": checkpoint_sha,
        "source": "training-split cohort patients only (145 training hospitals); aggregates, no patient rows",
        "reference_hospital": ref_hospital,
        "context_patients": CONTEXT,
        "n_training_patients": int(len(tab)),
        "bins": BIN_LABELS,
        "tabular": [spec(c, tab[c], "tabular") for c in tabular_features],
        "timeseries": [spec(c, ts[c], "timeseries", lab=c.startswith("lab_")) for c in ts_cols],
        "rules": {"default": "median of training-split values", "bounds": "0.1st-99.9th percentile of training-split values",
                  "binary": "0 or 1", "labs": "raw unit from lab.csv, converted with timeseries_stats.json (global mean / SD)"},
    }


# ── model ───────────────────────────────────────────────────────────────────
class InteractiveModel:
    """Verified sepsis checkpoint + training-split reference context for form-based inference."""

    def __init__(self, root: pathlib.Path = SEPSIS_ROOT, data_dir: Optional[pathlib.Path] = None):
        import torch
        from scripts.benchmark_baselines import load_run
        from src.sepsis_task import load_sepsis_hospital_dataset
        from torch.utils.data import DataLoader

        root = pathlib.Path(root)
        mpath = root / "final_model_manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"{mpath} not found: train with python scripts/final_model.py train --task mortality_sepsis")
        self.manifest = json.loads(mpath.read_text())
        if self.manifest.get("task_id") != "mortality_sepsis":
            raise ValueError(f"{mpath} is not a mortality + sepsis model")
        entry = self.manifest["models"][self.manifest["primary_model"]]
        ckpt_path = PROJECT_ROOT / portable(entry["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint {entry['checkpoint']} is missing")
        self.sha = _sha256(ckpt_path)
        if self.sha != entry["sha256"]:
            raise ValueError("checkpoint SHA-256 does not match the manifest")
        self.model, ckpt, _ = load_run(ckpt_path.parent, torch.device("cpu"))
        if ckpt.get("task") != "mortality_sepsis" or ckpt.get("synthetic_demo"):
            raise ValueError("checkpoint is not the real-data mortality + sepsis model")
        self.model.eval()
        self.ckpt_meta = {k: ckpt[k] for k in ("round", "tabular_dim", "timeseries_dim", "gnn_type", "knn_k", "use_gnn", "targets")}
        self.entry, self.checkpoint = entry, ckpt_path
        data_dir = pathlib.Path(data_dir or PROJECT_ROOT / portable(self.manifest["data_dir"]))

        ref_path = root / REFERENCE_FILE
        ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
        if ref is None or ref.get("checkpoint_sha256") != self.sha:
            ref = build_reference(data_dir, ckpt["tabular_features"], self.sha)
            try:
                ref_path.write_text(json.dumps(ref, indent=2))
            except OSError:  # read-only mount (Docker): keep the aggregates in memory
                pass
        self.reference = ref
        self.tab_names = [s["name"] for s in ref["tabular"]]
        self.ts_names = [s["name"] for s in ref["timeseries"]]
        if self.tab_names != ckpt["tabular_features"] or len(self.ts_names) != ckpt["timeseries_dim"]:
            raise ValueError("form feature order does not match the checkpoint")

        ds = load_sepsis_hospital_dataset(data_dir / "hospitals" / ref["reference_hospital"], data_dir / "timeseries_stats.json",
                                          str(data_dir / "sepsis_labels.parquet"))
        if ds.timeseries.feature_names != self.ts_names:
            raise ValueError("time-series feature order differs from the loader")
        tab_x, ts_x, lens, _, _ = next(iter(DataLoader(ds, batch_size=CONTEXT, shuffle=False)))
        self.context = (tab_x, ts_x, lens)  # stays on the backend
        self.tab_mean, self.tab_std = ds.tabular.mean.astype(np.float32), ds.tabular.std.astype(np.float32)
        self.ts_mean, self.ts_std = ds.timeseries._col_mean, ds.timeseries._col_std  # 0 / 1: no extra loader scaling
        stats = json.loads((data_dir / "timeseries_stats.json").read_text())
        self.lab_stats = {s["name"]: stats[s["name"][4:]] for s in ref["timeseries"] if s["kind"] == "lab"}

    # ── payload -> model tensors ────────────────────────────────────────────
    def defaults(self) -> dict:
        return {"tabular": {s["name"]: s["default"] for s in self.reference["tabular"]},
                "timeseries": {s["name"]: [s["default"]] * N_BINS for s in self.reference["timeseries"]}}

    def validate(self, payload: dict) -> tuple[np.ndarray, list[list[Optional[float]]]]:
        errors = []
        if not isinstance(payload, dict):
            raise InputError(["payload must be a JSON object"])
        tab_in, ts_in = payload.get("tabular"), payload.get("timeseries")
        if not isinstance(tab_in, dict) or not isinstance(ts_in, dict):
            raise InputError(["payload needs 'tabular' and 'timeseries' objects"])
        unknown = sorted(set(tab_in) - set(self.tab_names)) + sorted(set(ts_in) - set(self.ts_names))
        if unknown:
            errors.append(f"unknown features: {unknown[:5]}")

        def check(spec, raw, where):
            if raw is None or (isinstance(raw, str) and raw.strip() == ""):
                if spec["kind"] == "lab":
                    return None  # not measured
                errors.append(f"{where}: value required")
                return 0.0
            if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                errors.append(f"{where}: not a number")
                return 0.0
            try:
                v = float(raw)
            except ValueError:
                errors.append(f"{where}: '{raw}' is not a number")
                return 0.0
            if not np.isfinite(v):
                errors.append(f"{where}: must be finite")
            elif spec["kind"] == "binary" and v not in (0.0, 1.0):
                errors.append(f"{where}: must be 0 or 1")
            elif not spec["min"] <= v <= spec["max"]:
                errors.append(f"{where}: {v:g} is outside the training range {spec['min']:g} to {spec['max']:g}")
            return v

        tab = np.zeros(len(self.tab_names), dtype=np.float32)
        for i, spec in enumerate(self.reference["tabular"]):
            if spec["name"] not in tab_in:
                errors.append(f"{spec['name']}: missing")
                continue
            tab[i] = check(spec, tab_in[spec["name"]], spec["name"]) or 0.0
        series = []
        for spec in self.reference["timeseries"]:
            vals = ts_in.get(spec["name"])
            if not isinstance(vals, list) or len(vals) != N_BINS:
                errors.append(f"{spec['name']}: needs a list of {N_BINS} values (one per 4-hour bin)")
                series.append([0.0] * N_BINS)
                continue
            series.append([check(spec, v, f"{spec['name']} [{BIN_LABELS[b]}]") for b, v in enumerate(vals)])
        if errors:
            raise InputError(errors)
        return tab, series

    def encode(self, tab_raw: np.ndarray, series: list) -> tuple:
        """Raw form values -> the tensors the loader would produce for a patient of the reference hospital."""
        import torch
        tab = (tab_raw - self.tab_mean) / self.tab_std
        ts = np.zeros((N_BINS, len(self.ts_names)), dtype=np.float32)
        for j, (spec, vals) in enumerate(zip(self.reference["timeseries"], series)):
            for b, v in enumerate(vals):
                if spec["kind"] == "lab":
                    st = self.lab_stats[spec["name"]]
                    ts[b, j] = 0.0 if v is None else (v - st["mean"]) / st["std"]
                else:
                    ts[b, j] = v
        ts = (ts - self.ts_mean) / self.ts_std
        return (torch.tensor(tab, dtype=torch.float32)[None], torch.tensor(ts, dtype=torch.float32)[None],
                torch.tensor([N_BINS]))

    # ── inference ───────────────────────────────────────────────────────────
    def predict(self, payload: dict, threshold: float = 0.5) -> dict:
        import torch
        if not 0.0 < float(threshold) < 1.0:
            raise InputError(["threshold must be between 0 and 1"])
        tab_raw, series = self.validate(payload)
        u_tab, u_ts, u_len = self.encode(tab_raw, series)
        c_tab, c_ts, c_len = self.context
        seen = {}
        hooks = [self.model.fusion.register_forward_hook(lambda m, i, o: seen.update(fused=o)),
                 self.model.gnn.register_forward_hook(lambda m, i, o: seen.update(gnn_in=i[0]))]
        try:
            with torch.no_grad():  # the hypothetical patient is row 0 of a batch with 31 reference training patients
                out = self.model(torch.cat([u_tab, c_tab]), torch.cat([u_ts, c_ts]), torch.cat([u_len, c_len]))
        finally:
            for h in hooks:
                h.remove()
        p_mort = float(torch.sigmoid(out["mortality_logits"][0, 0]))
        p_sep = float(torch.sigmoid(out["icu_logits"][0, 0]))
        ei = out["edge_index"]
        vthr = self.entry["evaluation"].get("validation_chosen_threshold", {})
        base = self.defaults()
        changed = sum(float(v) != base["tabular"][k] for k, v in payload["tabular"].items() if v not in (None, ""))
        changed += sum((v is None or v == "") or float(v) != base["timeseries"][k][b]
                       for k, vals in payload["timeseries"].items() for b, v in enumerate(vals))
        target = lambda p, key: {"probability": p, "predicted_class": int(p >= threshold), "threshold": float(threshold),
                                 "validation_chosen_threshold": (vthr.get(key) or {}).get("threshold")}
        return {
            "source": "trained_real_data_checkpoint",
            "model": "FedAvg + GraphSAGE",
            "checkpoint": self.entry["checkpoint"], "checkpoint_round": self.ckpt_meta["round"],
            "targets": {"hospital_expired": target(p_mort, "mort"), "sepsis_after_24h": target(p_sep, "sepsis")},
            "inference": {
                "checkpoint_verified": True, "checkpoint_sha256_prefix": self.sha[:12],
                "fusion_executed": "fused" in seen and tuple(seen["fused"].shape) == (CONTEXT + 1, 128),
                "graphsage_input_is_fused": "gnn_in" in seen and torch.equal(seen["gnn_in"], seen["fused"]),
                "graph_constructed": ei is not None and ei.shape[1] > 0,
                "graph_edges": 0 if ei is None else int(ei.shape[1]),
                "neighbours_of_input": 0 if ei is None else int((ei[1] == 0).sum()),
                "batch": f"1 hypothetical input + {CONTEXT} training-split patients of a reference training hospital",
                "tabular_input_shape": list(u_tab.shape), "timeseries_input_shape": list(u_ts.shape),
                "values_differing_from_baseline": int(changed),
            },
            "embedding_sizes": {"tabular": int(out["tab_emb"].shape[1]), "timeseries": int(out["ts_emb"].shape[1]),
                                "fused": int(out["fused_emb"].shape[1]), "graph": int(out["graph_emb"].shape[1])},
            "embeddings": {k: [round(float(v), 4) for v in out[f"{k}_emb"][0].tolist()] for k in ("tab", "ts", "fused", "graph")},
            "notice": "Hypothetical input; predictions of a research model trained on eICU data. Not medical advice.",
        }

    def schema(self) -> dict:
        e = self.entry
        return {
            "bins": BIN_LABELS, "tabular": self.reference["tabular"], "timeseries": self.reference["timeseries"],
            "example": EXAMPLE, "rules": self.reference["rules"],
            "reference": {"source": self.reference["source"], "context_patients": CONTEXT,
                          "n_training_patients": self.reference["n_training_patients"],
                          "method": "hypothetical patient scaled like a patient of a fixed training hospital and placed in a "
                                    "batch with 31 of its training patients for the k-NN graph"},
            "model": {
                "status": "Trained checkpoint verified", "checkpoint": e["checkpoint"], "sha256_prefix": self.sha[:12],
                "architecture": e["architecture"], "targets": self.ckpt_meta["targets"],
                "rounds_completed": e["training"]["rounds"], "rounds_configured": e["training"].get("rounds_configured"),
                "selected_round": e["training"]["selected_round"], "best_val_loss": e["training"].get("best_val_loss"),
                "seed": (e["training"].get("args") or {}).get("seed"),
                "clients_per_round": (e["training"].get("args") or {}).get("clients_per_round"),
                "test_patients": e["evaluation"]["n_patients"], "test_hospitals": e["evaluation"]["test_hospitals"],
                "test_metrics": e["evaluation"]["metrics"],
                "validation_chosen_threshold": e["evaluation"].get("validation_chosen_threshold"),
                "label_definition": self.manifest.get("label_definition"), "limitations": self.manifest.get("limitations"),
                "pos_weight": self.manifest.get("pos_weight"),
            },
        }
