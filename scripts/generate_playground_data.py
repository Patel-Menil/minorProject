import pathlib
import sys
import json
import torch
from torch.utils.data import DataLoader

# Add project root to path so we can import src
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from src.datasets import EICUTabularDataset, EICUTimeSeriesDataset, timeseries_collate_fn
from src.encoders import TabularEncoder, TimeSeriesLSTM
from src.fusion import MultimodalFusion
from src.graph_builder import build_knn_graph

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
HOSPITAL_DIR = PROCESSED_DIR / "hospitals" / "hospital_102"
STATS_PATH = PROCESSED_DIR / "timeseries_stats.json"
OUTPUT_FILE = PROJECT_ROOT / "frontend-playground" / "public" / "graph_data.json"

def main():
    print("Generating playground data from hospital_102...")
    
    # 1. Load Data
    tab_ds = EICUTabularDataset(HOSPITAL_DIR, fit_scaler=True)
    ts_ds = EICUTimeSeriesDataset(HOSPITAL_DIR, STATS_PATH, max_seq_len=48, tabular_dataset=tab_ds)
    
    # We will just load all patients from this hospital (batch_size = len)
    # Hospital_102 has 31 patients. Let's load them all.
    tab_loader = DataLoader(tab_ds, batch_size=64, shuffle=False)
    ts_loader = DataLoader(ts_ds, batch_size=64, shuffle=False, collate_fn=timeseries_collate_fn)
    
    features, _, _ = next(iter(tab_loader))
    seqs, lens, targets, pids = next(iter(ts_loader))

    # 2. Encoders
    tab_enc = TabularEncoder(input_dim=tab_ds.input_dim, embedding_dim=64)
    ts_enc = TimeSeriesLSTM(input_dim=ts_ds.input_dim, hidden_dim=64, embedding_dim=64)
    
    with torch.no_grad():
        tab_emb = tab_enc(features)
        ts_emb = ts_enc(seqs, lens)
        
    # 3. Fusion
    fusion = MultimodalFusion(tabular_dim=64, timeseries_dim=64, fused_dim=128)
    with torch.no_grad():
        fused_emb = fusion(tab_emb, ts_emb)
        
    # 4. Graph Construction (k-NN)
    edge_index, edge_weights = build_knn_graph(fused_emb, k=3, metric="cosine")
    
    # --- Prepare JSON Data for Frontend ---
    nodes = []
    for i, pid in enumerate(pids):
        # Determine mortality (targets[i][0] is hospital_expired)
        expired = bool(targets[i][0].item()) if targets is not None else False
        
        nodes.append({
            "id": f"{pid}",
            "name": f"Patient {pid}",
            "val": 10, # Size of node
            "color": "#ef4444" if expired else "#10b981", # Red if expired, Green if survived
            "expired": expired
        })
        
    links = []
    edges_np = edge_index.numpy()
    weights_np = edge_weights.numpy()
    
    for j in range(edges_np.shape[1]):
        source_idx = edges_np[0, j]
        target_idx = edges_np[1, j]
        
        # Undirected graph, avoid duplicates (source < target)
        if source_idx < target_idx:
            links.append({
                "source": f"{pids[source_idx]}",
                "target": f"{pids[target_idx]}",
                "weight": float(weights_np[j])
            })
            
    graph_data = {
        "nodes": nodes,
        "links": links
    }
    
    # Ensure public dir exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(graph_data, f, indent=2)
        
    print(f"Graph data successfully written to {OUTPUT_FILE}")
    print(f"Total Nodes: {len(nodes)}, Total Links: {len(links)}")

if __name__ == "__main__":
    main()
