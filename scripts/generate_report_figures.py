"""
generate_report_figures.py
--------------------------
Generates the charts and diagrams for the corrected B.Tech Project Report.

Figures 4-6 are drawn only from saved experiment outputs in results/corrected/
and from the playground graph built by the trained model; if those files are
missing the script stops instead of drawing made-up numbers. Output goes to
report_assets/corrected/ so the original report figures are kept.
"""

import json
import os
import pathlib
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.splits import has_both_modalities
ASSETS_DIR = PROJECT_ROOT / "report_assets" / "corrected"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results" / "corrected"
GRAPH_FILE = PROJECT_ROOT / "frontend-playground" / "public" / "graph_data.json"
MODEL_LABELS = {
    "fedavg_gnn": "FedAvg + GNN\n(proposed)",
    "fedavg_nognn": "FedAvg No-GNN\n(ablation)",
    "local_61": "Local-only\nhosp. 61",
    "local_396": "Local-only\nhosp. 396",
    "local_183": "Local-only\nhosp. 183",
}


def load_json(path: pathlib.Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the experiments first (scripts/run_experiments.py)")
    with open(path) as f:
        return json.load(f)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
})

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Federated Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure_1():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axis('off')

    # Central Server Box (top)
    server_box = patches.FancyBboxPatch((3.0, 4.4), 4.0, 1.4, boxstyle="round,pad=0.15",
                                        ec="#1e3a8a", fc="#dbeafe", lw=2)
    ax.add_patch(server_box)
    ax.text(5.0, 5.3, "Central Federated Server", ha="center", va="center", weight="bold", fontsize=13, color="#1e3a8a")
    ax.text(5.0, 4.8, "• Parameter Aggregator (FedAvg)\n• Model Distribution & Validation\n• No raw patient rows received", 
            ha="center", va="center", fontsize=9.5, color="#1e293b")

    # 3 example training-split hospital clients
    hospitals = [
        {"x": 0.5, "name": "Hospital Client 1\n(eICU hospital 61)"},
        {"x": 3.75, "name": "Hospital Client 2\n(eICU hospital 396)"},
        {"x": 7.0, "name": "Hospital Client 3\n(eICU hospital 183)"},
    ]

    for h in hospitals:
        hx = h["x"]
        # Outer Hospital Box
        h_box = patches.FancyBboxPatch((hx, 0.4), 2.5, 3.2, boxstyle="round,pad=0.1",
                                       ec="#047857", fc="#f0fdf4", lw=1.8)
        ax.add_patch(h_box)
        ax.text(hx + 1.25, 3.3, h["name"], ha="center", va="center", weight="bold", fontsize=10.5, color="#065f46")
        ax.text(hx + 1.25, 2.9, "Local data (first 24 h)", ha="center", va="center", style="italic", fontsize=8.5, color="#047857")

        # Internal Modules
        y_start = 2.45
        modules = [
            ("Tabular MLP + Time-Series LSTM", "#e0f2fe", "#0284c7"),
            ("Multimodal Fusion (128D)", "#fef3c7", "#d97706"),
            ("k-NN Graph within each Batch", "#ede9fe", "#7c3aed"),
            ("PatientGNN + Multi-Task Heads", "#fee2e2", "#dc2626"),
        ]
        for name, bg, border in modules:
            m_box = patches.FancyBboxPatch((hx + 0.15, y_start - 0.35), 2.2, 0.38,
                                           boxstyle="round,pad=0.04", ec=border, fc=bg, lw=1)
            ax.add_patch(m_box)
            ax.text(hx + 1.25, y_start - 0.16, name, ha="center", va="center", fontsize=8, weight="medium", color="#1e293b")
            y_start -= 0.5

        # Connection arrows between Hospital and Server
        ax.annotate("", xy=(hx + 1.25, 4.4), xytext=(hx + 1.25, 3.6),
                    arrowprops=dict(arrowstyle="<->", color="#2563eb", lw=1.8, mutation_scale=15))
        ax.text(hx + 1.25, 4.0, "Model weights only\n(raw data stays local)", ha="center", va="center",
                fontsize=7.5, color="#1d4ed8", backgroundcolor="#ffffff")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    plt.tight_layout()
    out_path = ASSETS_DIR / "fig1_system_architecture.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Data Flow Diagram (DFD)
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure_2():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')

    stages = [
        ("Raw eICU Tables\n(CSV / SQL)", 0.5, 2.5, "#f1f5f9", "#475569"),
        ("Hospital Split\n(145 train / 31 val\n/ 31 test)", 2.4, 2.5, "#e0f2fe", "#0284c7"),
        ("Features, first 24 h\n(Tabular 67D,\nTS 6 x 74D)", 4.3, 2.5, "#fef3c7", "#d97706"),
        ("Modality Encoders\n(MLP &\n2-Layer LSTM)", 6.2, 2.5, "#dcfce7", "#16a34a"),
        ("Fusion & GNN\n(k-NN Graph +\nSAGE/GAT/GCN)", 8.1, 2.5, "#ede9fe", "#7c3aed"),
    ]

    for label, x, y, bg, border in stages:
        box = patches.FancyBboxPatch((x, y - 0.6), 1.5, 1.2, boxstyle="round,pad=0.08",
                                     ec=border, fc=bg, lw=1.5)
        ax.add_patch(box)
        ax.text(x + 0.75, y, label, ha="center", va="center", fontsize=8.5, weight="bold", color="#1e293b")

    for i in range(len(stages) - 1):
        x1 = stages[i][1] + 1.5
        x2 = stages[i + 1][1]
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.5, mutation_scale=12))

    # Output Heads
    out_y1 = 3.6
    out_y2 = 1.4
    h1 = patches.FancyBboxPatch((8.1, out_y1 - 0.4), 1.5, 0.8, boxstyle="round,pad=0.05", ec="#dc2626", fc="#fee2e2", lw=1.5)
    h2 = patches.FancyBboxPatch((8.1, out_y2 - 0.4), 1.5, 0.8, boxstyle="round,pad=0.05", ec="#ea580c", fc="#ffedd5", lw=1.5)
    ax.add_patch(h1)
    ax.add_patch(h2)
    ax.text(8.85, out_y1, "Task 1:\nIn-Hospital Mortality", ha="center", va="center", fontsize=8, weight="bold", color="#991b1b")
    ax.text(8.85, out_y2, "Task 2:\nICU Mortality", ha="center", va="center", fontsize=8, weight="bold", color="#9a3412")

    ax.annotate("", xy=(8.85, out_y1 - 0.4), xytext=(8.85, 3.1),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5, mutation_scale=12))
    ax.annotate("", xy=(8.85, out_y2 + 0.4), xytext=(8.85, 1.9),
                arrowprops=dict(arrowstyle="->", color="#ea580c", lw=1.5, mutation_scale=12))

    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.5, 4.5)
    plt.tight_layout()
    out_path = ASSETS_DIR / "fig2_data_flow_diagram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Hospital Patient Count Distribution
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure_3():
    # Same 207 hospitals as the experiments (both modalities present)
    h_dirs = sorted([d for d in (PROJECT_ROOT / "processed" / "hospitals").iterdir()
                     if d.is_dir() and has_both_modalities(d)])
    counts = []
    for d in h_dirs:
        df = pd.read_parquet(d / "tabular" / "tabular_features.parquet")
        counts.append(len(df))

    counts = sorted(counts, reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Rank-frequency curve
    ax1.plot(range(1, len(counts) + 1), counts, color="#0284c7", lw=2)
    ax1.fill_between(range(1, len(counts) + 1), counts, color="#e0f2fe", alpha=0.6)
    ax1.set_title("Hospital Patient Cohort Scale (Sorted)")
    ax1.set_xlabel("Hospital Rank")
    ax1.set_ylabel("Patient Count (Log Scale)")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Histogram
    ax2.hist(counts, bins=25, color="#10b981", edgecolor="#047857", alpha=0.7)
    ax2.axvline(np.median(counts), color="#dc2626", linestyle="--", lw=1.8, label=f"Median: {int(np.median(counts))} pts")
    ax2.axvline(np.mean(counts), color="#d97706", linestyle="-.", lw=1.8, label=f"Mean: {int(np.mean(counts))} pts")
    ax2.set_title("Distribution of Patients Across 207 Hospitals")
    ax2.set_xlabel("Number of Patients")
    ax2.set_ylabel("Frequency (Hospitals)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig3_hospital_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Federated Training Convergence Curves
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure_4():
    """Validation loss and AUROC per round (mean ± SD over seeds) for the two FedAvg models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, color in [("fedavg_gnn", "#dc2626"), ("fedavg_nognn", "#2563eb")]:
        runs = [load_json(p) for p in sorted((RESULTS_DIR / name).glob("seed*/training_history.json"))]
        if not runs:
            raise FileNotFoundError(f"No training histories for {name} in {RESULTS_DIR}")
        rounds = [r["round"] for r in runs[0]]
        label = MODEL_LABELS[name].replace("\n", " ")
        for ax, key in [(ax1, "val_loss"), (ax2, "mort_auroc")]:
            vals = np.array([[r[key] for r in run] for run in runs])
            mean = vals.mean(0)
            sd = vals.std(0, ddof=1) if len(runs) > 1 else np.zeros(len(rounds))
            ax.plot(rounds, mean, color=color, lw=2, label=f"{label} (n={len(runs)} seeds)")
            ax.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=0.15)

    ax1.set_title("Validation Loss per Communication Round")
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Validation Loss (31 val hospitals)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    ax2.set_title("Validation In-Hospital Mortality AUROC")
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("AUROC (31 val hospitals)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig4_training_convergence.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Comparative Baseline Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure_5():
    """Test-hospital results (mean ± SD over seeds) from benchmark_results.json."""
    summary = load_json(RESULTS_DIR / "benchmark_results.json")["summary"]
    models = [m for m in MODEL_LABELS if m in summary]
    metrics = [("mort_auroc", "In-hospital mortality AUROC", "#3b82f6"),
               ("icu_auroc", "ICU mortality AUROC", "#10b981"),
               ("mort_auprc", "In-hospital mortality AUPRC", "#f59e0b")]

    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (key, label, color) in enumerate(metrics):
        means = [summary[m][key]["mean"] for m in models]
        sds = [summary[m][key]["sd"] or 0 for m in models]
        bars = ax.bar(x + (i - 1) * width, means, width, yerr=sds, capsize=3, label=label, color=color, edgecolor="#334155")
        for rect, v in zip(bars, means):
            ax.annotate(f"{v:.3f}", xy=(rect.get_x() + rect.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    n_seeds = summary[models[0]]["n_seeds"]
    ax.set_ylabel("Score on test hospitals")
    ax.set_title(f"Held-out Test Hospitals (31): mean ± SD over {n_seeds} seeds")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig5_baseline_comparisons.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Patient Graph Topology
# ─────────────────────────────────────────────────────────────────────────────
def generate_figure_6():
    """The real patient graph built by the trained model (generate_playground_data.py output)."""
    import networkx as nx

    data = load_json(GRAPH_FILE)
    if data["meta"]["source"] != "model":
        raise ValueError(f"{GRAPH_FILE} is not real model output; run scripts/generate_playground_data.py")
    g = nx.Graph()
    g.add_nodes_from(n["id"] for n in data["nodes"])
    g.add_edges_from((l["source"], l["target"]) for l in data["links"])
    pos = nx.spring_layout(g, seed=0)
    nodes = data["nodes"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axis("off")
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#94a3b8", width=0.6, alpha=0.6)
    sc = ax.scatter([pos[n["id"]][0] for n in nodes], [pos[n["id"]][1] for n in nodes],
                    c=[n["pred_mortality"] for n in nodes], cmap="RdYlGn_r", vmin=0, vmax=1, s=45,
                    edgecolors=["#7f1d1d" if n["true_hospital_expired"] else "#334155" for n in nodes],
                    linewidths=[2.2 if n["true_hospital_expired"] else 0.6 for n in nodes], zorder=2)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Predicted in-hospital mortality")
    ax.scatter([], [], s=45, facecolors="white", edgecolors="#7f1d1d", linewidths=2.2, label="Died in hospital (recorded)")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    m = data["meta"]
    ax.set_title(f"Patient Graph, Test {m['hospital'].replace('_', ' ')} ({m['patients']} patients, k={m['knn_k']})")

    plt.tight_layout()
    out_path = ASSETS_DIR / "fig6_patient_graph_topology.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    print("All 6 publication figures generated successfully!")
