"""
generate_final_report_figures.py
--------------------------------
Figures for the final B.Tech report (report_assets/final_report/). Diagrams describe the
implemented mortality + sepsis model (two modalities, no text encoder). Charts are drawn only
from saved artifacts: training_history.json of both sepsis-task runs, the manifests, the label
summary and report_assets/final_report/test_curves_aggregate.json (aggregate ROC / PR points).

Usage
-----
    python scripts/generate_final_report_figures.py
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "report_assets" / "final_report"
SEP = ROOT / "results" / "minor_project_sepsis"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10})

C = {"tab": "#dbeafe", "ts": "#dcfce7", "fuse": "#fef3c7", "graph": "#ede9fe", "head": "#fee2e2", "fed": "#f1f5f9",
     "edge": "#334155"}


def box(ax, x, y, w, h, text, color, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=color, ec=C["edge"], lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, lw=1.2, color=C["edge"], linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", fontsize=8, color="#475569")


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = canvas(9.6, 5.2, (0, 13), (0, 7))
    box(ax, 0.2, 4.9, 2.4, 1.1, "Tabular features\n28 values / stay", C["tab"])
    box(ax, 0.2, 1.7, 2.4, 1.1, "Time-series features\n6 bins x 74 (first 24 h)", C["ts"])
    box(ax, 3.3, 4.9, 2.3, 1.1, "Tabular MLP\n28-128-64-64", C["tab"], bold=True)
    box(ax, 3.3, 1.7, 2.3, 1.1, "2-layer LSTM\nhidden 64 -> 64", C["ts"], bold=True)
    box(ax, 6.3, 3.25, 2.0, 1.3, "Multimodal\nFusion\n64 + 64 -> 128", C["fuse"], bold=True)
    box(ax, 8.9, 3.25, 1.8, 1.3, "k-NN graph\n(cosine, k = 5)\nwithin batch", C["graph"])
    box(ax, 11.2, 3.25, 1.6, 1.3, "GraphSAGE\n2 layers\n128 -> 64", C["graph"], bold=True)
    box(ax, 10.2, 5.4, 2.6, 0.9, "Head 1: hospital_expired", C["head"])
    box(ax, 10.2, 1.5, 2.6, 0.9, "Head 2: sepsis_after_24h", C["head"])
    arrow(ax, 2.6, 5.45, 3.3, 5.45); arrow(ax, 2.6, 2.25, 3.3, 2.25)
    arrow(ax, 5.6, 5.45, 6.6, 4.55, "64-D"); arrow(ax, 5.6, 2.25, 6.6, 3.25, "64-D")
    arrow(ax, 8.3, 3.9, 8.9, 3.9, "128-D"); arrow(ax, 10.7, 3.9, 11.2, 3.9)
    arrow(ax, 12.0, 4.55, 11.6, 5.4); arrow(ax, 12.0, 3.25, 11.6, 2.4)
    ax.add_patch(FancyBboxPatch((0.05, 0.15), 12.9, 6.7, boxstyle="round,pad=0.02,rounding_size=0.1", fc="none", ec="#94a3b8", lw=1, ls="--"))
    ax.text(6.5, 0.45, "One hospital client: the same model is trained locally at every sampled hospital and averaged by FedAvg",
            ha="center", fontsize=8.5, style="italic", color="#475569")
    save(fig, "fig_architecture.png")


def encoders():
    fig, ax = canvas(9.6, 3.9, (0, 13), (0, 5.2))
    ax.text(3.0, 4.85, "(a) Tabular encoder", ha="center", weight="bold")
    for i, (t, c) in enumerate([("Input\n28", C["tab"]), ("Linear 128\nBatchNorm\nReLU\nDropout", "#eff6ff"),
                                ("Linear 64\nBatchNorm\nReLU\nDropout", "#eff6ff"), ("Linear 64\nembedding", C["tab"])]):
        box(ax, 0.3 + i * 1.55, 2.0, 1.3, 1.9, t, c, fs=8.5)
        if i:
            arrow(ax, 0.3 + i * 1.55 - 0.25, 2.95, 0.3 + i * 1.55, 2.95)
    ax.text(9.6, 4.85, "(b) Time-series encoder", ha="center", weight="bold")
    for b in range(6):
        box(ax, 6.6 + b * 1.0, 0.35, 0.85, 0.9, f"bin {b}\n{4 * b}-{4 * b + 4} h", C["ts"], fs=7.5)
        arrow(ax, 7.02 + b * 1.0, 1.25, 7.02 + b * 1.0, 1.95)
    box(ax, 6.6, 1.95, 5.85, 0.9, "2-layer LSTM, hidden 64\n(packed sequences, true length <= 6)", "#f0fdf4", fs=8.5)
    arrow(ax, 9.5, 2.85, 9.5, 3.45)
    ax.text(9.7, 3.1, "last valid hidden state", fontsize=7.5, color="#475569")
    box(ax, 8.6, 3.45, 1.8, 0.8, "Linear 64\nembedding", C["ts"], fs=8.5)
    save(fig, "fig_encoders.png")


def fusion_graph():
    import numpy as np
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.set_xlim(0, 7); a1.set_ylim(0, 5); a1.axis("off"); a1.set_title("(a) Multimodal fusion", fontsize=10, weight="bold")
    box(a1, 0.2, 3.3, 1.7, 0.9, "tab emb\n64", C["tab"]); box(a1, 0.2, 0.8, 1.7, 0.9, "ts emb\n64", C["ts"])
    box(a1, 2.5, 2.05, 1.2, 0.9, "concat\n128", "#fffbeb")
    box(a1, 4.2, 1.55, 2.5, 1.9, "Linear 128->128\nBN, ReLU, Dropout\nLinear 128->128\nBN, ReLU", C["fuse"], fs=8.5)
    arrow(a1, 1.9, 3.75, 2.5, 2.8); arrow(a1, 1.9, 1.25, 2.5, 2.2); arrow(a1, 3.7, 2.5, 4.2, 2.5)
    a1.text(5.45, 0.9, "fused patient vector h (128-D)", ha="center", fontsize=8.5)
    rng = np.random.default_rng(3)  # schematic only: random points, not patients
    pts = rng.normal(size=(14, 2)); a2.set_title("(b) k-NN graph within one batch (schematic)", fontsize=10, weight="bold")
    sim = pts @ pts.T / (np.linalg.norm(pts, axis=1)[:, None] * np.linalg.norm(pts, axis=1)[None])
    np.fill_diagonal(sim, -9)
    for i in range(len(pts)):
        for j in np.argsort(-sim[i])[:3]:
            a2.annotate("", xy=pts[i], xytext=pts[j], arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=0.8))
    a2.scatter(pts[:, 0], pts[:, 1], s=90, c="#a78bfa", edgecolors=C["edge"], zorder=3)
    a2.scatter(pts[0, 0], pts[0, 1], s=140, c="#f59e0b", edgecolors=C["edge"], zorder=4)
    a2.set_xticks([]); a2.set_yticks([])
    a2.text(0.02, 0.02, "each patient receives edges from its k most similar\npatients (cosine of fused vectors); k = 3 drawn, k = 5 used",
            transform=a2.transAxes, fontsize=7.5, color="#475569")
    save(fig, "fig_fusion_graph.png")


def fedavg():
    fig, ax = canvas(9.6, 4.4, (0, 13), (0, 6.2))
    box(ax, 3.6, 4.6, 5.8, 1.4, "Server (simulated in the same process)\nglobal model w(t), FedAvg aggregation,\ncheckpoint = lowest validation loss", C["fed"], bold=True)
    for i, h in enumerate(["Hospital A", "Hospital B", "Hospital C", "...", "Hospital J"]):
        x = 0.4 + i * 2.55
        box(ax, x, 0.6, 2.1, 1.6, "..." if h == "..." else f"{h}\nlocal data stays here\n1 epoch, <= 30 steps",
            C["tab"] if h != "..." else "white", fs=8)
        if h != "...":
            arrow(ax, 6.1, 4.6, x + 0.75, 2.2, style="-|>")
            arrow(ax, x + 1.35, 2.2, 6.9, 4.6, style="-|>", ls="--")
    ax.text(1.6, 3.6, "w(t) sent to 10 sampled\ntraining hospitals", fontsize=8.5, color="#475569")
    ax.text(9.1, 3.6, "local weights w_k and example\ncounts n_k returned (dashed)", fontsize=8.5, color="#475569")
    ax.text(6.5, 0.1, "w(t+1) = sum_k (n_k / n) w_k ; 50 rounds; all clients run in one process", ha="center", fontsize=8.5, style="italic")
    save(fig, "fig_fedavg.png")


def sepsis_timeline():
    fig, ax = canvas(9.6, 3.6, (-3, 100), (0, 5))
    ax.plot([0, 96], [4.3, 4.3], color=C["edge"], lw=1.5)
    for t in (0, 24, 48, 72, 96):
        ax.plot([t, t], [4.2, 4.4], color=C["edge"]); ax.text(t, 4.55, f"{t} h", ha="center", fontsize=8.5)
    ax.axvspan(0, 24, ymin=0.05, ymax=0.82, color="#dbeafe", alpha=0.6)
    ax.text(12, 3.75, "observation window\n(model inputs: 6 x 4-h bins)", ha="center", fontsize=8.5)
    ax.axvline(24, ymin=0.05, ymax=0.82, color="#dc2626", lw=1.5, ls="--"); ax.text(25, 3.85, "cutoff 1,440 min", color="#dc2626", fontsize=8.5)
    rows = [("Excluded: sepsis documented by 24 h", 14, 70, "#94a3b8"), ("Positive: first sepsis diagnosis after 24 h, before ICU discharge", 50, 80, "#dc2626"),
            ("Negative: diagnoses recorded, no qualifying sepsis diagnosis", None, 66, "#16a34a"), ("Excluded: no diagnosis rows at all", None, 58, "#94a3b8")]
    for i, (label, dx, stay, col) in enumerate(rows):
        y = 2.9 - i * 0.72
        ax.plot([0, stay], [y, y], color=col, lw=3, solid_capstyle="round", alpha=0.85)
        ax.plot(stay, y, marker="|", ms=12, color=col)
        if dx is not None:
            ax.plot(dx, y, marker="D", ms=7, color="white", mec=col, mew=2)
        ax.text(stay + 1.5, y, label, va="center", fontsize=8)
    ax.text(0, 0.05, "bar = ICU stay (| = discharge), diamond = first qualifying sepsis / septic-shock diagnosis (diagnosisoffset)", fontsize=7.8, color="#475569")
    save(fig, "fig_sepsis_label.png")


def cohort_flow():
    lab = json.loads((SEP / "label_summary.json").read_text())
    c = lab["cohort_24h_by_split"]
    tot = lambda key: sum(v.get(key, 0) for v in c.values())
    cohort24 = sum(sum(v.values()) for v in c.values())
    pos, neg, prior, nodx = tot("positive"), tot("negative"), tot("excluded_prior_sepsis"), tot("excluded_no_diagnosis")
    fig, ax = canvas(9.0, 4.8, (0, 12), (0, 7))
    box(ax, 3.5, 5.9, 5, 0.85, f"All ICU stays in 207 split hospitals: 200,826", C["fed"], bold=True)
    box(ax, 3.5, 4.35, 5, 0.85, f"Stays of at least 24 h (24 h cohort): {cohort24:,}", C["tab"])
    box(ax, 8.9, 4.35, 2.9, 0.85, f"excluded: stay < 24 h\n{200826 - cohort24:,}", "white", fs=8.5)
    box(ax, 3.5, 2.8, 5, 0.85, f"Diagnosis rows present: {cohort24 - nodx:,}", C["tab"])
    box(ax, 8.9, 2.8, 2.9, 0.85, f"excluded: no diagnosis rows\n{nodx:,}", "white", fs=8.5)
    box(ax, 3.5, 1.25, 5, 0.85, f"Sepsis cohort: {pos + neg:,}\n{pos:,} positive / {neg:,} negative", C["head"], bold=True)
    box(ax, 8.9, 1.25, 2.9, 0.85, f"excluded: sepsis by 24 h\n{prior:,}", "white", fs=8.5)
    for y in (5.9, 4.35, 2.8):
        arrow(ax, 6, y, 6, y - 0.7)
    for y in (4.78, 3.23, 1.68):
        arrow(ax, 8.5, y, 8.9, y)
    ax.text(6, 0.45, "per split (train / val / test): " + " / ".join(f"{c[k].get('positive', 0) + c[k].get('negative', 0):,}" for k in ("train", "val", "test")),
            ha="center", fontsize=8.5)
    save(fig, "fig_cohort_flow.png")


def dfd():
    fig, ax = canvas(9.6, 4.6, (0, 13), (0, 6.4))
    box(ax, 0.2, 4.6, 2.6, 1.1, "eICU source tables\n(per hospital, local)", "white")
    box(ax, 3.5, 4.6, 2.8, 1.1, "Prepared Parquet\ntabular + time-series", C["tab"])
    box(ax, 7.0, 4.6, 2.6, 1.1, "Label builder\n(diagnosis.csv)", C["head"])
    box(ax, 10.2, 4.6, 2.6, 1.1, "sepsis_labels.parquet", C["head"])
    box(ax, 3.5, 2.5, 2.8, 1.2, "Loader: pairing by\npatient ID, 24 h window,\nleak-free columns", C["ts"])
    box(ax, 7.0, 2.5, 2.6, 1.2, "Federated training\n(train hospitals)\nval: checkpoint", C["graph"])
    box(ax, 10.2, 2.5, 2.6, 1.2, "Checkpoint +\nmanifest (SHA-256)", C["fed"])
    box(ax, 7.0, 0.4, 2.6, 1.2, "Test evaluation\n(31 test hospitals)", C["fuse"])
    box(ax, 10.2, 0.4, 2.6, 1.2, "Inference API +\nReact frontend", C["fuse"])
    for a in [(2.8, 5.15, 3.5, 5.15), (6.3, 5.15, 7.0, 5.15), (9.6, 5.15, 10.2, 5.15), (4.9, 4.6, 4.9, 3.7), (11.5, 4.6, 8.9, 3.7),
              (6.3, 3.1, 7.0, 3.1), (9.6, 3.1, 10.2, 3.1), (11.5, 2.5, 11.5, 1.6), (11.0, 2.5, 8.6, 1.6), (4.9, 2.5, 7.0, 1.0)]:
        arrow(ax, *a)
    save(fig, "fig_dfd.png")


def deployment():
    fig, ax = canvas(9.6, 3.4, (0, 13), (0, 4.6))
    box(ax, 0.2, 1.6, 2.0, 1.3, "Browser\n127.0.0.1:<port>", "white")
    box(ax, 3.0, 1.6, 2.6, 1.3, "frontend\nnginx + built React\n(port 80)", C["fuse"], bold=True)
    box(ax, 6.4, 1.6, 2.6, 1.3, "backend\nvalidation + proxy\n(port 8000, internal)", C["tab"], bold=True)
    box(ax, 9.8, 1.6, 3.0, 1.3, "model\ncheckpoint inference\n(port 8001, internal)", C["graph"], bold=True)
    for x1, x2 in ((2.2, 3.0), (5.6, 6.4), (9.0, 9.8)):
        arrow(ax, x1, 2.25, x2, 2.25)
    box(ax, 9.8, 0.1, 3.0, 0.9, "read-only mounts: processed/,\nresults/, demo_artifacts/", "white", fs=8)
    arrow(ax, 11.3, 1.0, 11.3, 1.6)
    ax.text(6.5, 3.6, "run-demo.ps1: free port 8765-8795 -> .env.runtime -> docker compose up -> health checks", ha="center", fontsize=8.5, style="italic")
    save(fig, "fig_docker.png")


def training_curves():
    hist = {n: json.loads((SEP / n / "seed42" / "training_history.json").read_text()) for n in ("fedavg_gnn", "fedavg_nognn")}
    sel = {n: json.loads((SEP / "final_model_manifest.json").read_text())["models"][n]["training"]["selected_round"] for n in hist}
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.0))
    for key, ax, title in (("val_loss", axes[0], "Validation loss"), ("mort_auroc", axes[1], "Validation AUROC: mortality"),
                           ("icu_auroc", axes[2], "Validation AUROC: sepsis")):
        for n, col in (("fedavg_gnn", "#2563eb"), ("fedavg_nognn", "#f59e0b")):
            r = [h["round"] for h in hist[n]]; v = [h[key] for h in hist[n]]
            ax.plot(r, v, color=col, lw=1.4, label="FedAvg + GraphSAGE" if n == "fedavg_gnn" else "FedAvg, No-GNN")
            ax.axvline(sel[n], color=col, ls=":", lw=1)
        ax.set_title(title, fontsize=9.5); ax.set_xlabel("communication round"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_training_curves.png")


def test_curves():
    d = json.loads((OUT / "test_curves_aggregate.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.9))
    for head, title, j in (("mort", "in-hospital mortality", 0), ("second", "sepsis after 24 h", 2)):
        for n, col, lab in (("fedavg_gnn", "#2563eb", "GNN"), ("fedavg_nognn", "#f59e0b", "No-GNN")):
            c = d[n][head]
            axes[j].plot(c["fpr"], c["tpr"], color=col, lw=1.3, label=f"{lab} {c['auroc']:.3f}")
            axes[j + 1].plot(c["recall"], c["precision"], color=col, lw=1.3, label=f"{lab} {c['auprc']:.3f}")
        axes[j].plot([0, 1], [0, 1], color="#94a3b8", ls="--", lw=0.8)
        axes[j + 1].axhline(d["fedavg_gnn"][head]["prevalence"], color="#94a3b8", ls="--", lw=0.8)
        axes[j].set_title(f"ROC: {title}", fontsize=9); axes[j].set_xlabel("false positive rate"); axes[j].set_ylabel("true positive rate")
        axes[j + 1].set_title(f"PR: {title}", fontsize=9); axes[j + 1].set_xlabel("recall"); axes[j + 1].set_ylabel("precision")
        for a in (axes[j], axes[j + 1]):
            a.legend(fontsize=7, loc="best"); a.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig_test_curves.png")


def metric_bars():
    m = json.loads((SEP / "final_model_manifest.json").read_text())["models"]
    import numpy as np
    labels = ["Mortality\nAUROC", "Mortality\nAUPRC", "Sepsis\nAUROC", "Sepsis\nAUPRC"]
    vals = {n: [m[n]["evaluation"]["metrics"][t][k] for t, k in (("mort", "auroc"), ("mort", "auprc"), ("sepsis", "auroc"), ("sepsis", "auprc"))] for n in m}
    x = np.arange(4); fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, (n, col, lab) in enumerate((("fedavg_gnn", "#2563eb", "FedAvg + GraphSAGE"), ("fedavg_nognn", "#f59e0b", "FedAvg, No-GNN"))):
        b = ax.bar(x + (i - 0.5) * 0.36, vals[n], 0.36, color=col, label=lab)
        ax.bar_label(b, fmt="%.3f", fontsize=7.5, padding=2)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Held-out test hospitals (14,910 stays)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig_metric_bars.png")


def flowchart():
    from matplotlib.patches import Polygon
    fig, ax = canvas(7.2, 10.6, (0, 10), (0, 15.2))
    nodes = [("term", "Start"), ("box", "Validate data (10 checks)"), ("dec", "All checks\npassed?"),
             ("box", "Build sepsis labels; pos_weight\nfrom training labels only"),
             ("box", "Initialise global model w(0); t = 1"), ("box", "Sample 10 of 145 training hospitals"),
             ("box", "Local training at each client\n(1 epoch, at most 30 steps) -> w_k, n_k"),
             ("box", "FedAvg: w(t) = sum_k (n_k / n) w_k"), ("box", "Evaluate w(t) on 31 validation hospitals"),
             ("dec", "L_val < best?"), ("dec", "t < 50?"), ("box", "Score the saved checkpoint once\non 31 test hospitals"),
             ("term", "Write manifest (SHA-256, metrics)\nEnd")]
    cx, bw, bh, dw, dh = 4.2, 4.6, 0.72, 2.6, 0.92
    ys = [14.6 - i * 1.12 for i in range(len(nodes))]
    half = []
    for (kind, text), y in zip(nodes, ys):
        if kind == "dec":
            ax.add_patch(Polygon([(cx - dw / 2, y), (cx, y + dh / 2), (cx + dw / 2, y), (cx, y - dh / 2)], closed=True,
                                 fc=C["fuse"], ec=C["edge"], lw=1.1))
            ax.text(cx, y, text, ha="center", va="center", fontsize=9)
            half.append(dh / 2)
        else:
            col = C["fed"] if kind == "term" else (C["graph"] if "FedAvg" in text or "Local" in text else C["tab"])
            box(ax, cx - bw / 2, y - bh / 2, bw, bh, text, col, fs=9, bold=kind == "term")
            half.append(bh / 2 + 0.02)
    for i in range(len(nodes) - 1):
        arrow(ax, cx, ys[i] - half[i], cx, ys[i + 1] + half[i + 1])
    for i, word in ((2, "yes"), (9, "no"), (10, "no")):
        ax.text(cx + 0.12, ys[i] - half[i] - 0.22, word, fontsize=8, color="#475569")
    # failed data check
    box(ax, 7.1, ys[2] - 0.36, 2.7, 0.72, "Stop: report the\nfailed check", "white", fs=8.5)
    arrow(ax, cx + dw / 2, ys[2], 7.1, ys[2])
    ax.text(6.0, ys[2] + 0.1, "no", fontsize=8, color="#475569")
    # new best round
    box(ax, 7.1, ys[9] - 0.36, 2.7, 0.72, "Save w(t) as\nthe checkpoint", C["head"], fs=8.5)
    arrow(ax, cx + dw / 2, ys[9], 7.1, ys[9])
    ax.text(6.0, ys[9] + 0.1, "yes", fontsize=8, color="#475569")
    arrow(ax, 8.45, ys[9] - 0.36, cx + dw / 4, ys[10] + dh / 4)
    # next round
    ax.plot([cx - dw / 2, 0.7, 0.7], [ys[10], ys[10], ys[5]], color=C["edge"], lw=1.2)
    arrow(ax, 0.7, ys[5], cx - bw / 2, ys[5])
    ax.text(0.85, (ys[10] + ys[5]) / 2, "yes: t = t + 1", fontsize=8, color="#475569", rotation=90, va="center")
    save(fig, "fig_flowchart.png")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (architecture, encoders, fusion_graph, fedavg, sepsis_timeline, cohort_flow, dfd, deployment, training_curves, test_curves, metric_bars, flowchart):
        f()
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
