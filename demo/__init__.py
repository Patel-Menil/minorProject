"""
Synthetic demonstration workflow (viva / evaluation demo).

Everything in this package works WITHOUT the eICU data. It writes a small,
deterministic, clearly labelled synthetic dataset in the same on-disk format
as the prepared eICU hospitals, then runs the project's real data loaders,
model, FedAvg code and evaluation on it. Nothing here is a clinical result.
"""

SYNTHETIC_NOTICE = (
    "SYNTHETIC DATA - academic demonstration only. These patients are generated "
    "by a random process, not eICU records, and model outputs are not clinically meaningful."
)
