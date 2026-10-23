"""
generate_report.py
------------------
Generates a faculty-ready .docx progress report for Steps 1-4.
"""
import pathlib
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "Progress_Report_Steps1_to_4.docx"
SCREENSHOT_DIR = pathlib.Path("/home/dhruv/.gemini/antigravity-ide/brain/db057c23-1b86-4556-af62-c920c1897c00")

def set_cell_shading(cell, color):
    from docx.oxml.ns import qn
    from lxml import etree
    shading = etree.SubElement(cell._element.get_or_add_tcPr(), qn('w:shd'))
    shading.set(qn('w:fill'), color)
    shading.set(qn('w:val'), 'clear')

def add_styled_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    
    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, "2563EB")
    
    # Data rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
            if r_idx % 2 == 1:
                set_cell_shading(cell, "EFF6FF")
    
    return table

def main():
    doc = Document()
    
    # --- Styles ---
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    
    # ===== TITLE PAGE =====
    for _ in range(6):
        doc.add_paragraph()
    
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Federated Multimodal Graph Learning Framework")
    run.bold = True
    run.font.size = Pt(26)
    run.font.color.rgb = RGBColor(30, 64, 175)
    
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Mortality & Sepsis Prediction using the eICU Dataset")
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(100, 116, 139)
    
    doc.add_paragraph()
    
    line = doc.add_paragraph()
    line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = line.add_run("Progress Report — Steps 1 through 4")
    run.font.size = Pt(14)
    run.bold = True
    
    doc.add_paragraph()
    
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run("Date: October 7, 2026\nDataset: eICU Collaborative Research Database")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(100, 116, 139)
    
    doc.add_page_break()
    
    # ===== TABLE OF CONTENTS =====
    doc.add_heading('Table of Contents', level=1)
    toc_items = [
        "1. Executive Summary",
        "2. Dataset Overview",
        "3. Step 1 — Data Preparation",
        "4. Step 2 — Modality-Specific Feature Learning",
        "    4.1 Tabular Encoder (MLP)",
        "    4.2 Time-Series Encoder (LSTM)",
        "    4.3 Text Encoder (MiniBERT)",
        "    4.4 Federated Pre-training",
        "5. Step 3 — Multimodal Fusion",
        "6. Step 4 — Intra-Hospital Patient Graph Construction",
        "7. Interactive Visualization Playground",
        "8. Codebase Structure",
        "9. Summary & Next Steps",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
    
    doc.add_page_break()
    
    # ===== 1. EXECUTIVE SUMMARY =====
    doc.add_heading('1. Executive Summary', level=1)
    doc.add_paragraph(
        'This report documents the implementation progress of the first four steps of our '
        'Federated Multimodal Graph Learning Framework for predicting in-hospital mortality '
        'and sepsis. The framework treats each hospital as an independent federated learning '
        'client, ensuring that raw patient data never leaves the hospital boundary.'
    )
    doc.add_paragraph('We have successfully completed:')
    items = [
        'Step 1 — Data Preparation (partitioning the eICU dataset by hospital)',
        'Step 2 — Modality-Specific Feature Learning (Tabular MLP, Time-Series LSTM, and Text MiniBERT encoders)',
        'Step 3 — Multimodal Fusion (concatenation + projection network)',
        'Step 4 — Intra-Hospital Patient Graph Construction (k-NN and threshold-based graphs)',
    ]
    for item in items:
        doc.add_paragraph(item, style='List Bullet')
    
    doc.add_paragraph(
        'Additionally, a federated pre-training script for the Tabular Autoencoder using FedAvg '
        'has been implemented and executed, and a React-based interactive visualization playground '
        'has been built to inspect the patient graphs.'
    )
    
    # ===== 2. DATASET OVERVIEW =====
    doc.add_heading('2. Dataset Overview', level=1)
    add_styled_table(doc,
        ['Metric', 'Value'],
        [
            ['Source', 'eICU Collaborative Research Database'],
            ['Total Hospitals', '208'],
            ['Total Patients', '200,859'],
            ['Tabular Features per Patient', '72 (demographics, vitals, diagnoses, medications)'],
            ['Time-Series Features per Timestep', '51 (labs, infusions, treatments)'],
            ['Time-Series Resolution', 'Hourly bins (padded/truncated to 48 hours)'],
            ['Text Features', 'Lab notes, infusion names, medication names'],
            ['Prediction Targets', 'Hospital mortality, ICU mortality'],
        ]
    )
    doc.add_paragraph()
    doc.add_paragraph(
        'Data is partitioned by hospitalid so that each hospital directory contains independent '
        'tabular and timeseries parquet files. Each hospital performs all preprocessing locally, '
        'ensuring no cross-hospital data sharing occurs at the data level.'
    )
    
    # ===== 3. STEP 1 =====
    doc.add_heading('3. Step 1 — Data Preparation', level=1)
    doc.add_paragraph('Module: src/datasets.py')
    doc.add_paragraph('Classes: EICUTabularDataset, EICUTimeSeriesDataset')
    
    doc.add_heading('What It Does', level=2)
    steps = [
        'Loads the .parquet files for a single hospital.',
        'Separates input features from target labels (mortality flags).',
        'Applies z-score normalization using globally computed statistics.',
        'Pads/truncates time-series sequences to a fixed window of 48 hours.',
        'Extracts unstructured text from free-text columns (lab_other_names, infusion_other_names, med_other_names).',
    ]
    for s in steps:
        doc.add_paragraph(s, style='List Number')
    
    doc.add_heading('Sample Output (Hospital 102)', level=2)
    add_styled_table(doc,
        ['Property', 'Value'],
        [
            ['Patients', '31'],
            ['Tabular input dimension', '67 features'],
            ['Time-series input dimension', '46 features × 48 timesteps'],
            ['Mortality cases', '3 out of 31 (9.7%)'],
        ]
    )
    
    # ===== 4. STEP 2 =====
    doc.add_heading('4. Step 2 — Modality-Specific Feature Learning', level=1)
    doc.add_paragraph(
        'Three separate neural network encoders process each modality independently, '
        'each producing a 64-dimensional patient embedding.'
    )
    
    doc.add_heading('4.1 Tabular Encoder (MLP)', level=2)
    add_styled_table(doc,
        ['Property', 'Value'],
        [
            ['Architecture', '3-layer MLP (input → 128 → 64 → 64)'],
            ['Normalization', 'BatchNorm after each hidden layer'],
            ['Activation', 'ReLU'],
            ['Regularization', 'Dropout (p=0.3)'],
            ['Input', '67 static clinical features'],
            ['Output', '64D embedding per patient'],
        ]
    )
    
    doc.add_heading('4.2 Time-Series Encoder (LSTM)', level=2)
    add_styled_table(doc,
        ['Property', 'Value'],
        [
            ['Architecture', '2-layer Stacked LSTM + Linear Projection'],
            ['Hidden dimension', '64'],
            ['Sequence handling', 'pack_padded_sequence for variable-length inputs'],
            ['Output extraction', 'Last valid hidden state'],
            ['Input', '(48, 46) — 48 hours × 46 features'],
            ['Output', '64D embedding per patient'],
        ]
    )
    
    doc.add_heading('4.3 Text Encoder (MiniBERT)', level=2)
    add_styled_table(doc,
        ['Property', 'Value'],
        [
            ['Base model', 'prajjwal1/bert-mini (HuggingFace)'],
            ['Parameters', '~11.3M (pre-trained)'],
            ['Tokenizer', 'BertTokenizer'],
            ['Max sequence length', '512 tokens'],
            ['Projection', 'Linear(256→128) → BN → ReLU → Linear(128→64)'],
            ['Input', 'Concatenated clinical text strings'],
            ['Output', '64D embedding per patient'],
        ]
    )
    
    doc.add_heading('Smoke Test Results (Hospital 102)', level=2)
    p = doc.add_paragraph()
    run = p.add_run(
        '[PASS] All smoke tests passed!\n'
        '   Tabular  : (67,) → (64,)\n'
        '   LSTM     : (48, 46) → (64,)\n'
        '   Text     : (variable length str) → (64,)'
    )
    run.font.name = 'Consolas'
    run.font.size = Pt(10)
    
    doc.add_heading('4.4 Federated Pre-training (Tabular Autoencoder)', level=2)
    add_styled_table(doc,
        ['Parameter', 'Value'],
        [
            ['Communication rounds', '15'],
            ['Local epochs per round', '3'],
            ['Client fraction per round', '15% (~31 hospitals)'],
            ['Batch size', '32'],
            ['Learning rate', '1e-3'],
            ['Aggregation strategy', 'FedAvg (weighted by patient count)'],
        ]
    )
    doc.add_paragraph()
    doc.add_paragraph(
        'The resulting pre-trained weights are saved at pretrained_weights/global_tabular_ae.pth.'
    )
    
    # ===== 5. STEP 3 =====
    doc.add_heading('5. Step 3 — Multimodal Fusion', level=1)
    doc.add_paragraph('Module: src/fusion.py')
    doc.add_paragraph('Class: MultimodalFusion')
    doc.add_paragraph(
        'The three 64D modality embeddings are concatenated to form a 192D vector, '
        'which is then projected through a smoothing MLP:'
    )
    p = doc.add_paragraph()
    run = p.add_run(
        'Tabular Emb (64) ─┐\n'
        '                   ├─ Concat (192) → Linear → BN → ReLU → Dropout\n'
        'TimeSeries Emb(64)─┤                → Linear(192→128) → BN → ReLU\n'
        'Text Emb (64) ─────┘\n'
        '                     Output: Unified Patient Embedding (128D)'
    )
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    
    add_styled_table(doc,
        ['Property', 'Value'],
        [
            ['Input', '3 × 64D = 192D concatenated embedding'],
            ['Hidden layer', '192 → 192 (with BatchNorm, ReLU, Dropout)'],
            ['Output', '128D unified patient embedding'],
        ]
    )
    
    # ===== 6. STEP 4 =====
    doc.add_heading('6. Step 4 — Intra-Hospital Patient Graph Construction', level=1)
    doc.add_paragraph('Module: src/graph_builder.py')
    doc.add_paragraph('Functions: build_knn_graph, build_threshold_graph')
    
    doc.add_heading('6.1 k-Nearest Neighbors (k-NN) Graph', level=2)
    steps = [
        'Computes pairwise cosine similarity between all patient embeddings.',
        'Connects each patient to their top-k most similar neighbors.',
        'Produces directed edges (source → target) with similarity scores as edge weights.',
    ]
    for s in steps:
        doc.add_paragraph(s, style='List Bullet')
    
    doc.add_heading('6.2 Threshold-Based Graph', level=2)
    steps = [
        'Computes pairwise cosine similarity.',
        'Creates an edge between patients whose similarity exceeds a threshold (e.g., 0.75).',
        'Produces undirected edges with similarity scores as weights.',
    ]
    for s in steps:
        doc.add_paragraph(s, style='List Bullet')
    
    doc.add_heading('Privacy Enforcement', level=2)
    doc.add_paragraph(
        'Since each batch only contains patients from one hospital (by construction of the '
        'federated data partitioning), cross-hospital edges are mathematically impossible. '
        'No additional filtering is required.'
    )
    
    doc.add_heading('Graph Output (Hospital 102, k=3 KNN)', level=2)
    add_styled_table(doc,
        ['Property', 'Value'],
        [
            ['Nodes (patients)', '31'],
            ['Edges', '49'],
            ['Edge format', 'PyTorch Geometric edge_index (2 × E)'],
            ['Edge weights', 'Cosine similarity scores'],
            ['Survived nodes', '28 (green)'],
            ['High-risk nodes', '3 (red)'],
        ]
    )
    
    # ===== 7. VISUALIZATION =====
    doc.add_heading('7. Interactive Visualization Playground', level=1)
    doc.add_paragraph(
        'To facilitate inspection and demonstration, we built a React + Vite frontend that '
        'renders the patient graph as an interactive force-directed layout.'
    )
    
    features = [
        'Force-directed graph with physics simulation (nodes repel, edges attract)',
        'Color-coded nodes — Green (Stable/Survived), Red (High Risk/Expired)',
        'Click-to-inspect — Selecting a patient reveals clinical text and neighbor count',
        'Real data — Generated from actual hospital_102 patient data through the full pipeline',
    ]
    for f in features:
        doc.add_paragraph(f, style='List Bullet')
    
    # Add screenshots if they exist
    img1 = SCREENSHOT_DIR / "graph_overview.png"
    img2 = SCREENSHOT_DIR / "patient_details.png"
    
    if img1.exists():
        doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(img1), width=Inches(5.5))
        caption = doc.add_paragraph('Figure 1: Patient Graph Overview — Hospital 102')
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.size = Pt(9)
        caption.runs[0].font.color.rgb = RGBColor(100, 116, 139)
    
    if img2.exists():
        doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(img2), width=Inches(5.5))
        caption = doc.add_paragraph('Figure 2: Selected Patient Details — Clinical Text and Modality Tags')
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].font.size = Pt(9)
        caption.runs[0].font.color.rgb = RGBColor(100, 116, 139)
    
    # ===== 8. CODEBASE =====
    doc.add_heading('8. Codebase Structure', level=1)
    p = doc.add_paragraph()
    run = p.add_run(
        'project-root/\n'
        '├── processed/                    # Raw data (208 hospitals)\n'
        '│   ├── hospitals/hospital_*/\n'
        '│   ├── feature_schema.json\n'
        '│   └── timeseries_stats.json\n'
        '├── src/                          # Core modules\n'
        '│   ├── datasets.py               # Step 1\n'
        '│   ├── encoders.py               # Step 2\n'
        '│   ├── fusion.py                 # Step 3\n'
        '│   └── graph_builder.py          # Step 4\n'
        '├── scripts/                      # Executable scripts\n'
        '├── tests/                        # Smoke tests\n'
        '├── pretrained_weights/           # Saved model weights\n'
        '└── frontend-playground/          # React visualization'
    )
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    
    # ===== 9. SUMMARY =====
    doc.add_heading('9. Summary & Next Steps', level=1)
    
    doc.add_heading('Completed (Steps 1–4)', level=2)
    add_styled_table(doc,
        ['Step', 'Component', 'Status'],
        [
            ['1', 'Data Preparation', '✅ Complete'],
            ['2', 'Tabular Encoder (MLP)', '✅ Complete'],
            ['2', 'Time-Series Encoder (LSTM)', '✅ Complete'],
            ['2', 'Text Encoder (MiniBERT)', '✅ Complete'],
            ['2', 'Federated Pre-training (AE + FedAvg)', '✅ Complete'],
            ['3', 'Multimodal Fusion', '✅ Complete'],
            ['4', 'Patient Graph Construction', '✅ Complete'],
        ]
    )
    
    doc.add_paragraph()
    doc.add_heading('Remaining (Steps 5–8)', level=2)
    add_styled_table(doc,
        ['Step', 'Component', 'Status'],
        [
            ['5', 'Graph Neural Network (GraphSAGE / GAT)', '⬜ Not started'],
            ['6', 'Multi-task Prediction Heads', '⬜ Not started'],
            ['7', 'Federated Client Training Loop', '⬜ Not started'],
            ['8', 'Central Server Aggregation (FedAvg)', '⬜ Not started'],
        ]
    )
    
    doc.add_paragraph()
    doc.add_heading('Planned Next Actions', level=2)
    next_items = [
        'Implement a Graph Neural Network module (GraphSAGE or GAT) for graph-enhanced patient embeddings.',
        'Add dual prediction heads for mortality and sepsis classification with a multi-task loss function.',
        'Build the full federated training loop coordinating all 208 hospital clients.',
        'Evaluate using AUROC, AUPRC, Accuracy, Precision, Recall, and F1-score.',
    ]
    for item in next_items:
        doc.add_paragraph(item, style='List Number')
    
    # Save
    doc.save(str(OUTPUT_PATH))
    print(f"Report saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
