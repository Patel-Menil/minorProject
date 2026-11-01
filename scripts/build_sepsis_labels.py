"""
build_sepsis_labels.py
----------------------
Builds the "sepsis documented after the first 24 h" label (rule in src/sepsis_task.py)
from every split hospital's diagnosis.csv and unitdischargeoffset.

Writes (both git-ignored, local only):
    processed/sepsis_labels.parquet                 one row per ICU stay of a split hospital
    results/minor_project_sepsis/label_summary.json counts and the qualifying phrases (no patient rows)

Usage
-----
    python scripts/build_sepsis_labels.py [--data-dir processed]
"""

import argparse
import collections
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.sepsis_task import CUTOFF_MIN, SEPSIS_SEGMENTS, is_qualifying

OUT_DIR = PROJECT_ROOT / "results" / "minor_project_sepsis"


def build(data_dir: pathlib.Path) -> tuple[pd.DataFrame, dict]:
    split = json.loads((data_dir / "hospital_split.json").read_text())
    role = {h: k for k in ("train", "val", "test") for h in split[k]}
    rows, phrases, missing = [], collections.Counter(), []
    for h in sorted(role):
        hdir = data_dir / "hospitals" / h
        stays = pd.read_parquet(hdir / "tabular" / "tabular_features.parquet", columns=["patientunitstayid", "unitdischargeoffset"])
        diag_path = hdir / "tabular" / "diagnosis.csv"
        if not diag_path.exists():
            missing.append(h)
            diag = pd.DataFrame(columns=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
        else:
            diag = pd.read_csv(diag_path, usecols=["patientunitstayid", "diagnosisoffset", "diagnosisstring"])
        has_diag = set(diag["patientunitstayid"])
        q = diag[diag["diagnosisstring"].map(is_qualifying)]
        phrases.update(q["diagnosisstring"])
        first = q.groupby("patientunitstayid")["diagnosisoffset"].min()
        # first qualifying diagnosis inside the ICU stay (after admission and before discharge)
        df = stays.copy()
        df["hospital"], df["split"] = h, role[h]
        df["has_diagnosis"] = df["patientunitstayid"].isin(has_diag)
        df["first_sepsis_offset"] = df["patientunitstayid"].map(first)
        prior = df["first_sepsis_offset"] <= CUTOFF_MIN
        later = (df["first_sepsis_offset"] > CUTOFF_MIN) & (df["first_sepsis_offset"] <= df["unitdischargeoffset"])
        df["status"] = "negative"
        df.loc[later, "status"] = "positive"
        df.loc[prior, "status"] = "excluded_prior_sepsis"
        df.loc[~df["has_diagnosis"], "status"] = "excluded_no_diagnosis"
        df["label"] = df["status"].map({"positive": 1.0, "negative": 0.0})
        rows.append(df)
    labels = pd.concat(rows, ignore_index=True)
    cohort = labels[labels["unitdischargeoffset"] >= CUTOFF_MIN]  # the 24 h cohort used for training
    summary = {
        "rule": {"qualifying_segments": sorted(SEPSIS_SEGMENTS), "cutoff_minutes": CUTOFF_MIN,
                 "offset_reference": "diagnosisoffset = minutes from ICU admission",
                 "positive": "first qualifying diagnosis after the cutoff and no later than ICU discharge",
                 "excluded": "qualifying diagnosis at or before the cutoff, or no diagnosis rows",
                 "limitation": "documented diagnosis time, not a validated sepsis onset (proxy label)"},
        "hospitals_missing_diagnosis_csv": missing,
        "all_stays": labels["status"].value_counts().to_dict(),
        "cohort_24h_by_split": {k: g["status"].value_counts().to_dict() for k, g in cohort.groupby("split")},
        "qualifying_phrases": dict(phrases.most_common()),
    }
    return labels, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the incident-sepsis proxy label")
    parser.add_argument("--data-dir", type=pathlib.Path, default=PROJECT_ROOT / "processed")
    args = parser.parse_args()
    labels, summary = build(args.data_dir)
    out = args.data_dir / "sepsis_labels.parquet"
    labels.to_parquet(out, index=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "label_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"labels: {out} ({len(labels)} stays)")
    for k, v in summary["cohort_24h_by_split"].items():
        pos, neg = v.get("positive", 0), v.get("negative", 0)
        print(f"  {k:<5} 24 h cohort: {pos} positive / {neg} negative ({pos / max(pos + neg, 1):.2%}), "
              f"excluded prior sepsis {v.get('excluded_prior_sepsis', 0)}, no diagnosis {v.get('excluded_no_diagnosis', 0)}")
    print(f"  qualifying phrases: {len(summary['qualifying_phrases'])}; hospitals without diagnosis.csv: {summary['hospitals_missing_diagnosis_csv']}")


if __name__ == "__main__":
    main()
