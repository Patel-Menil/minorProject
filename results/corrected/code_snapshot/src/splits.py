"""
splits.py
---------
Fixed hospital-level train / validation / test split.

- train : hospitals used by FedAvg clients and to train every baseline
- val   : hospitals used only to pick the best checkpoint (round)
- test  : hospitals used only once, for the final benchmark

The split is saved to JSON the first time and re-read afterwards, so
train_federated.py and benchmark_baselines.py always use the same hospitals.
"""

import json
import pathlib
import random


def has_both_modalities(hospital_dir: pathlib.Path) -> bool:
    return (hospital_dir / "tabular" / "tabular_features.parquet").exists() and (
        hospital_dir / "timeseries" / "timeseries_features.parquet"
    ).exists()


def load_or_create_split(
    hospitals_dir: str | pathlib.Path,
    split_path: str | pathlib.Path,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, list[pathlib.Path]]:
    """Returns {"train": [...], "val": [...], "test": [...]} as hospital directories."""
    hospitals_dir, split_path = pathlib.Path(hospitals_dir), pathlib.Path(split_path)
    settings = {"seed": seed, "val_fraction": val_fraction, "test_fraction": test_fraction}

    if split_path.exists():
        with open(split_path, "r") as f:
            split = json.load(f)
        saved = {k: split[k] for k in settings}
        if saved != settings:
            raise ValueError(
                f"{split_path} was made with {saved}, not {settings}. "
                "Use the same settings or delete the file to make a new split."
            )
    else:
        names = sorted(d.name for d in hospitals_dir.iterdir() if d.is_dir() and has_both_modalities(d))
        random.Random(seed).shuffle(names)
        n_test = round(len(names) * test_fraction)
        n_val = round(len(names) * val_fraction)
        split = {
            **settings,
            "test": sorted(names[:n_test]),
            "val": sorted(names[n_test:n_test + n_val]),
            "train": sorted(names[n_test + n_val:]),
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)

    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    if train & val or train & test or val & test:
        raise ValueError(f"Hospital split {split_path} has overlapping hospitals")
    if not (train and val and test):
        raise ValueError(f"Hospital split {split_path} has an empty train, val or test set")

    out = {k: [hospitals_dir / n for n in split[k]] for k in ("train", "val", "test")}
    missing = [d.name for dirs in out.values() for d in dirs if not has_both_modalities(d)]
    if missing:
        raise FileNotFoundError(f"Split lists hospitals without both Parquet files: {missing}")
    return out
