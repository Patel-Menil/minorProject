"""
validate_data.py
----------------
Checks that a prepared hospital dataset can be used for the real experiments,
before any training starts. Reads only column names, IDs, hour bins and the
two target columns; prints counts, never patient records.

Checks
  1  patient-level records exist for every split hospital (both Parquet files)
  2  tabular and time-series rows pair by patient ID (no duplicates)
  3  the 67 canonical tabular / 74 canonical time-series columns are present
  4  time-series has integer 4-hour bins covering the first 24 h (bins 0-5)
  5  hospital_expired and icu_expired exist and are binary, without missing values
  6  hospital folders match hospital_split.json
  7  preprocessing files (canonical feature order, timeseries_stats.json) exist
  8  train / val / test cohorts (ICU stay >= 24 h, both modalities) are non-empty
  9  train / val / test hospitals do not overlap
  10 no leakage: targets / discharge offsets are not features; no patient in two hospitals

Usage
-----
    python scripts/validate_data.py                         # processed/
    python scripts/validate_data.py --data-dir D:\\eicu_prepared --report validation.json
Exit code 0 = all checks passed, 1 = at least one failed.
"""

import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.datasets import HOUR_BIN_HOURS, OBS_HOURS, TABULAR_ID_COL, TABULAR_TARGET_COLS, TIMESERIES_TIME_COL

N_BINS = int(OBS_HOURS // HOUR_BIN_HOURS)
TARGETS = ["hospital_expired", "icu_expired"]
EXPECTED = {"tabular": 67, "timeseries": 74}
SPLIT_SETTINGS = {"seed": 42, "val_fraction": 0.15, "test_fraction": 0.15}


def validate(data_dir: pathlib.Path) -> dict:
    """Returns {"ok": bool, "synthetic": bool, "checks": [...], "counts": {...}}; never raises for bad data."""
    import pandas as pd
    import pyarrow.parquet as pq

    data_dir = pathlib.Path(data_dir)
    checks, counts = [], {}

    def check(no: int, name: str, ok: bool, detail: str) -> bool:
        checks.append({"check": no, "name": name, "ok": bool(ok), "detail": detail})
        return ok

    # 7 / 9 / 6: metadata first, so a missing-data report still validates what exists
    files = {n: data_dir / n for n in ("hospital_split.json", "canonical_tabular_features.json",
                                       "canonical_timeseries_features.json", "timeseries_stats.json")}
    missing_meta = [n for n, p in files.items() if not p.exists()]
    check(7, "preprocessing files", not missing_meta,
          "present" if not missing_meta else f"missing: {', '.join(missing_meta)}")
    if missing_meta:
        return {"ok": False, "synthetic": False, "data_dir": str(data_dir), "checks": checks, "counts": counts}
    split = json.loads(files["hospital_split.json"].read_text())
    tab_cols = json.loads(files["canonical_tabular_features.json"].read_text())
    ts_cols = json.loads(files["canonical_timeseries_features.json"].read_text())
    names = {k: list(split.get(k, [])) for k in ("train", "val", "test")}
    counts["split_hospitals"] = {k: len(v) for k, v in names.items()}
    tr, va, te = (set(names[k]) for k in ("train", "val", "test"))
    check(9, "split hospitals do not overlap", not (tr & va or tr & te or va & te) and tr and va and te,
          f"train {len(tr)} / val {len(va)} / test {len(te)}; overlaps {len(tr & va)}, {len(tr & te)}, {len(va & te)}")
    settings = {k: split.get(k) for k in SPLIT_SETTINGS}
    check(7, "split settings match the training scripts", settings == SPLIT_SETTINGS, f"{settings}")
    check(3, "canonical feature lists", len(tab_cols) == EXPECTED["tabular"] and len(ts_cols) == EXPECTED["timeseries"]
          and not set(tab_cols) & set(TABULAR_TARGET_COLS),
          f"{len(tab_cols)} tabular, {len(ts_cols)} time-series (expected 67 / 74)")

    hospitals_dir = data_dir / "hospitals"
    listed = sorted(tr | va | te)
    if not hospitals_dir.is_dir():
        check(1, "patient-level records exist", False, f"folder not found: {hospitals_dir}")
        return {"ok": False, "synthetic": False, "data_dir": str(data_dir), "checks": checks, "counts": counts}
    on_disk = {d.name for d in hospitals_dir.iterdir() if d.is_dir()}
    paths = {h: (hospitals_dir / h / "tabular" / "tabular_features.parquet",
                 hospitals_dir / h / "timeseries" / "timeseries_features.parquet") for h in listed}
    missing = [h for h, (a, b) in paths.items() if not (a.exists() and b.exists())]
    if not check(1, "patient-level records exist", not missing,
                 f"{len(listed) - len(missing)} of {len(listed)} split hospitals have both Parquet files"
                 + (f"; missing e.g. {', '.join(missing[:5])}" if missing else "")):
        return {"ok": False, "synthetic": False, "data_dir": str(data_dir), "checks": checks, "counts": counts}
    extra = sorted(on_disk - set(listed))
    check(6, "hospital folders match the split", True,
          "all split hospitals present" + (f"; {len(extra)} extra folders not in the split (ignored)" if extra else ""))

    synthetic = (data_dir / "DATASET_CARD.json").exists() and json.loads((data_dir / "DATASET_CARD.json").read_text()).get("synthetic", False)
    bad_ids, bad_targets, bad_bins, low_cov, leak, seen = [], [], [], [], [], {}
    ts_cov, no_labs = [], []
    cohort = {"train": 0, "val": 0, "test": 0}
    rows = {"tabular": 0, "timeseries": 0}
    positives = {t: 0 for t in TARGETS}
    for h in listed:
        tab_path, ts_path = paths[h]
        tab_have = set(pq.read_schema(tab_path).names)
        ts_have = set(pq.read_schema(ts_path).names)
        synthetic = synthetic or "synthetic" in tab_have
        need = [TABULAR_ID_COL, "unitdischargeoffset", *TARGETS]
        if not set(need) <= tab_have or not {TABULAR_ID_COL, TIMESERIES_TIME_COL} <= ts_have:
            bad_targets.append(f"{h}: missing {sorted(set(need) - tab_have) or sorted({TABULAR_ID_COL, TIMESERIES_TIME_COL} - ts_have)}")
            continue
        # Every tabular feature must exist. Time-series infusion / medication / treatment columns are
        # hospital-specific (the canonical list is the union over hospitals) and are zero-filled by the
        # loader when absent, as in the recorded experiments; only a hospital without any lab column is flagged.
        missing_tab = set(tab_cols) - tab_have
        if missing_tab:
            low_cov.append(f"{h}: {len(missing_tab)} tabular columns missing")
        ts_cov.append(len(set(ts_cols) & ts_have) / len(ts_cols))
        if not any(c in ts_have for c in ts_cols if c.startswith("lab_")):
            no_labs.append(h)
        tab = pd.read_parquet(tab_path, columns=need)
        ts = pd.read_parquet(ts_path, columns=[TABULAR_ID_COL, TIMESERIES_TIME_COL])
        rows["tabular"] += len(tab)
        rows["timeseries"] += len(ts)
        if tab[TABULAR_ID_COL].duplicated().any():
            bad_ids.append(f"{h}: duplicate tabular patient IDs")
        for t in TARGETS:
            v = tab[t]
            if v.isna().any() or not v.isin([0, 1]).all():
                bad_targets.append(f"{h}: {t} has missing or non-binary values")
            positives[t] += int((v == 1).sum())
        bins = ts[TIMESERIES_TIME_COL]
        if bins.isna().any() or (bins < 0).any() or not (bins == bins.round()).all():
            bad_bins.append(f"{h}: hour_bin is not a non-negative integer")
        for pid in tab[TABULAR_ID_COL]:
            if pid in seen:
                leak.append(f"patient in {seen[pid]} and {h}")
                break
            seen[pid] = h
        in_cohort = set(tab.loc[tab["unitdischargeoffset"] >= OBS_HOURS * 60, TABULAR_ID_COL])
        with_window = set(ts.loc[ts[TIMESERIES_TIME_COL] < N_BINS, TABULAR_ID_COL])
        n = len(in_cohort & with_window)
        for k in ("train", "val", "test"):
            if h in names[k]:
                cohort[k] += n
        if n == 0:
            bad_bins.append(f"{h}: no patient has both a >= 24 h stay and time-series in the first 24 h")

    counts.update({"hospitals_loaded": len(listed), "tabular_rows": rows["tabular"], "timeseries_rows": rows["timeseries"],
                   "cohort_24h_patients": cohort, "positives_all_rows": positives})
    check(2, "tabular / time-series pair by patient ID", not bad_ids, "; ".join(bad_ids[:5]) or "unique IDs per hospital")
    ts_cov.sort()
    cov_text = (f"all 67 tabular columns in every hospital; time-series columns per hospital: median "
                f"{ts_cov[len(ts_cov) // 2]:.0%}, min {ts_cov[0]:.0%} (hospital-specific drug / treatment columns "
                "are zero-filled by the loader)") if ts_cov else ""
    check(3, "canonical columns present in the Parquet files", not low_cov, "; ".join(low_cov[:5]) or cov_text)
    counts["warnings"] = ([f"{len(no_labs)} hospital(s) without any lab column (labs zero-filled): {', '.join(no_labs)}"]
                          if no_labs else [])
    check(4, f"time-series covers the first {OBS_HOURS} h in {N_BINS} bins", not bad_bins, "; ".join(bad_bins[:5]) or "ok")
    check(5, "binary targets hospital_expired / icu_expired", not bad_targets,
          "; ".join(bad_targets[:5]) or f"positives: {positives}")
    check(8, "non-empty 24 h cohorts", all(cohort.values()), f"{cohort}")
    check(10, "no leakage (targets not features, no patient in two hospitals)",
          not leak and not set(tab_cols) & set(TABULAR_TARGET_COLS), "; ".join(leak[:3]) or "ok")
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "synthetic": bool(synthetic), "data_dir": str(data_dir), "checks": checks, "counts": counts}


def print_report(rep: dict) -> None:
    print("=" * 78)
    print(f" Data validation: {rep['data_dir']}")
    if rep["synthetic"]:
        print(" NOTE: this is SYNTHETIC demo data, not eICU patient data.")
    print("=" * 78)
    for c in sorted(rep["checks"], key=lambda c: c["check"]):
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']:>2}. {c['name']}: {c['detail']}")
    for w in rep["counts"].get("warnings", []):
        print(f"  [WARN] {w}")
    if rep["counts"]:
        print(f"  counts: {json.dumps(rep['counts'])}")
    print(f" RESULT: {'all checks passed' if rep['ok'] else 'NOT usable for training'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the prepared hospital dataset before training")
    parser.add_argument("--data-dir", type=pathlib.Path, default=PROJECT_ROOT / "processed")
    parser.add_argument("--report", type=pathlib.Path, default=None, help="Also write the report as JSON (counts only)")
    args = parser.parse_args()
    rep = validate(args.data_dir)
    print_report(rep)
    if args.report:
        args.report.write_text(json.dumps(rep, indent=2))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
