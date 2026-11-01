"""
test_multimodal_fusion.py
-------------------------
Data-free checks that the model is genuinely multimodal: both encoders run,
MultimodalFusion combines each patient's own two embeddings, the fused vector is
what the graph builder and GraphSAGE receive, both heads predict, and one training
step updates parameters in BOTH encoders (gradients flow through the fusion module).
Uses the mortality + sepsis input sizes (28 tabular, 6 bins x 74 time-series).

Usage
-----
    python -m pytest tests/test_multimodal_fusion.py -v
"""

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.models import FederatedPatientModel, MultiTaskLoss

TAB, TS, BINS, N = 28, 74, 6, 16


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N, TAB, generator=g), torch.randn(N, BINS, TS, generator=g),
            torch.randint(1, BINS + 1, (N,), generator=g),
            torch.stack([torch.randint(0, 2, (N,), generator=g).float()] * 2 + [torch.zeros(N)] * 2, dim=1))


def test_fusion_combines_each_patients_two_embeddings_and_feeds_graphsage():
    torch.manual_seed(0)
    model = FederatedPatientModel(TAB, TS).eval()
    seen = {}
    model.fusion.register_forward_hook(lambda m, inp, out: seen.update(fusion_in=inp, fusion_out=out))
    model.gnn.register_forward_hook(lambda m, inp, out: seen.update(gnn_in=inp[0]))
    tab, ts, lens, _ = batch()
    with torch.no_grad():
        out = model(tab, ts, lens)
    tab_in, ts_in = seen["fusion_in"]
    assert torch.equal(tab_in, out["tab_emb"]) and torch.equal(ts_in, out["ts_emb"])      # both encoders' outputs
    assert out["tab_emb"].shape == (N, 64) and out["ts_emb"].shape == (N, 64) and out["fused_emb"].shape == (N, 128)
    assert torch.equal(seen["gnn_in"], out["fused_emb"])                                  # GraphSAGE gets the fused vector
    assert out["graph_emb"].shape == (N, 64) and out["edge_index"].shape == (2, N * 5)    # k-NN graph on fused embeddings
    assert out["mortality_logits"].shape == (N, 1) and out["icu_logits"].shape == (N, 1)  # both heads
    # row i of the fusion is computed from patient i's own embeddings only
    with torch.no_grad():
        alone = model.fusion(out["tab_emb"], out["ts_emb"])
        swapped = model.fusion(out["tab_emb"], out["ts_emb"].roll(1, 0))
    assert torch.allclose(alone, out["fused_emb"], atol=1e-6)
    assert not torch.allclose(swapped, out["fused_emb"])  # changing one modality changes the fused vector


def test_one_training_step_updates_both_encoders_through_fusion():
    torch.manual_seed(0)
    model = FederatedPatientModel(TAB, TS).train()
    tab, ts, lens, y = batch(1)
    loss, _, _ = MultiTaskLoss(pos_weight_mortality=3.0, pos_weight_icu=50.0)(*map(model(tab, ts, lens).get, ("mortality_logits", "icu_logits")), y)
    loss.backward()
    for part in ("tabular_encoder", "timeseries_encoder", "fusion", "gnn", "heads"):
        grads = [p.grad for p in getattr(model, part).parameters()]
        assert all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads), part
    before = {k: v.clone() for k, v in model.state_dict().items()}
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    for prefix in ("tabular_encoder.", "timeseries_encoder.", "fusion."):
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items() if k.startswith(prefix)), prefix
