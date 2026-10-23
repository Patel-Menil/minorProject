"""
test_fusion_graph.py
--------------------
Smoke test for Steps 3 and 4.
Loads a batch of data, runs encoders, fuses embeddings, and builds a graph.
"""

import pathlib
import sys

# Add project root to path so we can import src
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.datasets import load_hospital_dataset
from src.encoders import TabularEncoder, TimeSeriesLSTM
from src.fusion import MultimodalFusion
from src.graph_builder import build_knn_graph, build_threshold_graph

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITAL_DIR = PROCESSED_DIR / "hospitals" / "hospital_102"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"

def main():
    print("=" * 60)
    print(" Smoke Test: Fusion (Step 3) & Graph Builder (Step 4)")
    print("=" * 60)

    # 1. Load Data
    ds = load_hospital_dataset(HOSPITAL_DIR, STATS_PATH)
    features, seqs, lens, _, pids = next(iter(DataLoader(ds, batch_size=16, shuffle=False)))

    # 2. Encoders (Step 2)
    tab_enc = TabularEncoder(input_dim=ds.tabular_dim, embedding_dim=64)
    ts_enc = TimeSeriesLSTM(input_dim=ds.timeseries_dim, hidden_dim=64, embedding_dim=64)
    
    with torch.no_grad():
        tab_emb = tab_enc(features)
        ts_emb = ts_enc(seqs, lens)
        
    print(f"Tabular Embeddings : {tab_emb.shape}")
    print(f"Temporal Embeddings: {ts_emb.shape}")

    # 3. Fusion (Step 3)
    fusion = MultimodalFusion(tabular_dim=64, timeseries_dim=64, fused_dim=128)
    with torch.no_grad():
        fused_emb = fusion(tab_emb, ts_emb)
    
    print(f"Fused Embeddings   : {fused_emb.shape}  <-- Success (Step 3)")

    # 4. Graph Construction (Step 4)
    print("\n--- Graph Construction (Step 4) ---")
    
    # KNN Graph
    knn_edge_index, knn_edge_weight = build_knn_graph(fused_emb, k=3, metric="cosine")
    print(f"KNN Graph (k=3)")
    print(f"  Edges      : {knn_edge_index.shape[1]}")
    print(f"  Edge Index : {knn_edge_index.shape}")
    
    # Threshold Graph
    thresh_edge_index, thresh_edge_weight = build_threshold_graph(fused_emb, threshold=0.7)
    print(f"Threshold Graph (t=0.7)")
    print(f"  Edges      : {thresh_edge_index.shape[1]}")
    print(f"  Edge Index : {thresh_edge_index.shape}")

    print("\n✅ Fusion and Graph construction works perfectly!")
    
if __name__ == "__main__":
    main()
