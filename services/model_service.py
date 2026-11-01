"""
model_service.py
----------------
Container entry point of the "model" service (port 8001, Docker network only).

Reuses demo/server.py (all model routes: the real-data mortality + sepsis inference,
the final-model tabs and the synthetic demo). Before listening it loads and verifies
the trained mortality + sepsis checkpoint (SHA-256 against its manifest, architecture,
training-split graph context), so GET /health only answers 200 once real inference is
possible. Nothing is trained here. Checkpoints and data come from read-only mounts.
"""

import os
import pathlib
import sys
import warnings

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from demo import server  # noqa: E402


def main() -> None:
    port = int(os.environ.get("MODEL_PORT", "8001"))
    try:
        info = server.model_health()  # loads + verifies the checkpoint now, not on the first request
        print(f" model: checkpoint {info['checkpoint']} (round {info['checkpoint_round']}, "
              f"SHA-256 {info['checkpoint_sha256_prefix']}...) verified; targets {info['targets']}", flush=True)
    except server.ApiError as e:
        # keep serving so /health reports the reason (503); predictions return the same error
        print(f" model: NOT READY - {e}", flush=True)
    server.serve(port=port, host=os.environ.get("MODEL_HOST", "0.0.0.0"))


if __name__ == "__main__":
    main()
