"""
final_report_content.py
-----------------------
Text of the final report as a list of blocks, rendered by build_final_report.py.
Block kinds: ("h1", num, title) chapter, ("h2", text), ("h3", text), ("p", text), ("bullets", [..]),
("fig", file, caption, width_in), ("table", caption, header, rows, widths_in), ("eq", file, number), ("pb",).
Citations are written [@key] and numbered by first appearance. **bold** is supported.
Every number is either read from the manifests (passed in `m`) or matches them.
"""


def content(m: dict) -> list:
    G, NG, FG, FNG = m["G"], m["NG"], m["FG"], m["FNG"]
    gm, gs = G["evaluation"]["metrics"]["mort"], G["evaluation"]["metrics"]["sepsis"]
    nm, ns = NG["evaluation"]["metrics"]["mort"], NG["evaluation"]["metrics"]["sepsis"]
    gv, nv = G["evaluation"]["validation_chosen_threshold"], NG["evaluation"]["validation_chosen_threshold"]
    gt, nt = G["training"], NG["training"]
    lc, sc = m["LC"], m["SC"]
    pos_all = sum(lc[k]["positive"] for k in lc)
    neg_all = sum(lc[k]["negative"] for k in lc)
    prior = sum(lc[k]["excluded_prior_sepsis"] for k in lc)
    nodx = sum(lc[k]["excluded_no_diagnosis"] for k in lc)
    pw = m["SEP"]["pos_weight"]["sepsis_after_24h"]
    env = m["SEP"]["environment"]
    f3, f4 = m["f3"], m["f4"]
    fgm, fgi = FG["evaluation"]["metrics"]["mort"], FG["evaluation"]["metrics"]["icu"]
    hs = m["HIST"]["summary"]
    n_test = G["evaluation"]["n_patients"]

    B = []
    h1 = lambda n, t: B.append(("h1", n, t))
    h2 = lambda t: B.append(("h2", t))
    h3 = lambda t: B.append(("h3", t))
    p = lambda t: B.append(("p", t))
    bl = lambda items: B.append(("bullets", items))
    fig = lambda f, c, w=6.2: B.append(("fig", f, c, w))
    tab = lambda c, h, r, w: B.append(("table", c, h, r, w))
    eq = lambda f, n: B.append(("eq", f, n))

    # ── Chapter 1 ───────────────────────────────────────────────────────────
    h1(1, "INTRODUCTION")
    h2("1.1 Background")
    p("Intensive care units (ICUs) treat patients whose condition can change within hours. Two outcomes that matter "
      "for planning care are in-hospital death and the development of sepsis, a life-threatening organ dysfunction "
      "caused by a dysregulated response to infection [@sepsis3]. Bedside severity scores such as APACHE IV estimate "
      "mortality risk from first-day physiology [@zimmerman], and electronic health records (EHRs) now record far more "
      "information than these scores use: demographics, comorbidities, repeated laboratory measurements, infusions, "
      "medications and treatments. Machine-learning models trained on such records can combine many weak signals into "
      "a single risk estimate [@harut], [@rajkomar].")
    p("Two practical obstacles stand in the way. First, ICU data are heterogeneous: some information is static "
      "(recorded once per stay), while other information arrives as an irregular sequence over time. A useful model "
      "must represent both kinds of data and combine them. Second, clinical data are distributed across hospitals "
      "and are governed by data-use agreements, so pooling all records on one server is often not acceptable. "
      "Federated learning addresses the second problem by training a shared model while each participant keeps its "
      "own records and sends only model parameters [@mcmahan], [@rieke].")
    p("This project studies how far a federated, multimodal, graph-based model can go on a large public ICU "
      "database. It uses the eICU Collaborative Research Database, a multi-centre collection of ICU stays from "
      "hospitals across the United States, distributed through PhysioNet under a data-use agreement [@eicu], "
      "[@physionet]. Because eICU records the hospital of every stay, each hospital can be treated as a separate "
      "federated client, which makes the database a natural test bed for hospital-level federated learning.")
    h2("1.2 Motivation")
    p("Patients who look alike tend to follow similar courses. Graph neural networks (GNNs) can exploit this by "
      "connecting similar patients and letting each patient's representation draw on its neighbours [@hamilton], "
      "[@rocheteau]. In a federated setting the graph must respect hospital boundaries: edges may only join patients "
      "of the same hospital, because a hospital cannot see another hospital's patients. Combining modality-specific "
      "encoders, a fusion layer, an intra-hospital patient graph and federated averaging is the central idea of the "
      "original project specification (Overview document) and of this report.")
    h2("1.3 Problem statement")
    p("Given the first 24 hours of an ICU stay, predict (i) whether the patient will die in hospital and (ii) whether "
      "sepsis will be documented later in the stay, using a model that is trained across hospitals without pooling "
      "their raw records. The precise formulation, including the cohort and the label definitions, is given in "
      "Chapter 3.")
    h2("1.4 Objectives")
    bl(["Prepare a patient-aligned, leakage-controlled dataset from the eICU hospitals with a hospital-level "
        "train, validation and test split.",
        "Encode tabular data with a multilayer perceptron and the first-24-hour time series with an LSTM, and combine "
        "the two embeddings with a learned multimodal fusion module (two-modality fusion).",
        "Build a patient-similarity graph inside each hospital batch and refine the fused representations with "
        "GraphSAGE.",
        "Train the model with FedAvg over simulated hospital clients, select the checkpoint on validation hospitals "
        "and evaluate once on held-out test hospitals.",
        "Predict in-hospital mortality and subsequently documented sepsis with two prediction heads, and compare "
        "with a No-GNN ablation.",
        "Provide a reproducible demonstration: an inference API, a web interface for hypothetical inputs and a "
        "containerised deployment."])
    h2("1.5 Scope and what this report does not claim")
    p("The project is an academic research prototype. The federation is simulated in a single process on one "
      "computer; no secure aggregation or differential privacy is implemented. The original specification also "
      "proposed a clinical-text modality encoded with MiniBERT or ClinicalBERT; this was not implemented, so the "
      "trained model is a **two-modality** (tabular + time-series) model. The sepsis target is a proxy based on the "
      "time at which a sepsis diagnosis was documented, not a clinically validated onset time. None of the results "
      "establish clinical usefulness. These limitations are discussed in detail in Chapter 8.")
    h2("1.6 Organization of the report")
    p("Chapter 2 reviews related work. Chapter 3 defines the problem, the dataset and the objectives. Chapter 4 "
      "describes the methodology: data preparation, sepsis-label construction, leakage control and the model. "
      "Chapter 5 gives the algorithms, Chapter 6 the block diagrams and data flow, and Chapter 7 the implementation, "
      "tools, testing, web interface and deployment. Chapter 8 presents the experimental results, compares them with "
      "the original specification and discusses limitations. Chapter 9 concludes and outlines future work.")

    # ── Chapter 2 ───────────────────────────────────────────────────────────
    h1(2, "LITERATURE REVIEW")
    h2("2.1 Severity scores and early warning in the ICU")
    p("Classical ICU risk models are logistic regressions on a fixed set of first-day variables. APACHE IV, developed "
      "on more than 110,000 admissions from 104 ICUs, predicts hospital mortality from acute physiology, age, "
      "chronic health and admission diagnosis, and reported an area under the ROC curve of 0.88 [@zimmerman]. Such "
      "scores are transparent and widely used, but they compress the first day into worst values and cannot use "
      "the temporal pattern of measurements or the many variables outside the score. The eICU database stores APACHE "
      "IV variables and predictions for most stays [@eicu]; in this project the APACHE scores are used as tabular "
      "input features, not as a competing baseline.")
    p("Sepsis is defined by the Sepsis-3 consensus as life-threatening organ dysfunction caused by a dysregulated host "
      "response to infection, with organ dysfunction operationalised as an acute increase in the SOFA score [@sofa] of two "
      "points or more [@sepsis3]. Establishing the exact onset time requires suspected infection (cultures and "
      "antibiotics) and serial organ-function scores, which is why sepsis-prediction studies spend considerable "
      "effort on label construction.")
    h2("2.2 Deep learning on ICU records and time series")
    p("Recurrent neural networks, and in particular the long short-term memory (LSTM) network [@lstm], became the "
      "standard way to model ICU time series because their gated memory can carry information across irregular "
      "sequences. Harutyunyan et al. built public benchmarks for in-hospital mortality, decompensation, length of "
      "stay and phenotyping on MIMIC-III and showed that LSTM models with multitask training are strong baselines "
      "[@harut]. Rajkomar et al. trained deep models on the full EHR of two hospital systems and predicted "
      "in-hospital mortality, readmission and length of stay with high discrimination [@rajkomar]. These studies "
      "motivate the use of an LSTM over the first-day measurements in this project and the use of a shared trunk "
      "with several prediction heads (multitask learning).")
    h2("2.3 Sepsis prediction")
    p("Early sepsis prediction has been studied extensively. Nemati et al. proposed the Artificial Intelligence "
      "Sepsis Expert, which computed 65 hourly features and predicted Sepsis-3 onset several hours in advance in ICU "
      "patients from two hospitals, with external validation on MIMIC-III [@nemati]. The PhysioNet/Computing in "
      "Cardiology Challenge 2019 provided hourly data for more than 60,000 ICU patients from three hospital systems, "
      "labelled sepsis onset with Sepsis-3 criteria and scored entries with a utility metric that rewards early and "
      "penalises late or false alarms [@reyna]. The challenge showed that results obtained on one hospital system "
      "did not always transfer to another, which highlights the value of hospital-level evaluation. In contrast to "
      "these works, the present project does not reconstruct Sepsis-3 onset: the available prepared data do not "
      "contain the culture and antibiotic timing needed, so a documentation-time proxy is used and its limitations "
      "are stated explicitly (Section 4.3).")
    h2("2.4 Multimodal representation learning")
    p("Multimodal learning combines information from different data types. Baltrušaitis et al. organise the field "
      "into representation, translation, alignment, fusion and co-learning, and distinguish early fusion "
      "(concatenating raw inputs), late fusion (combining per-modality predictions) and intermediate fusion of "
      "learned representations [@baltrusaitis]. Intermediate fusion lets each modality use a suitable encoder while "
      "the fused vector can still capture interactions between modalities; it is the approach used here.")
    p("Within intermediate fusion, three designs are common. Concatenation joins the modality embeddings and lets "
      "the following layers learn their interactions. Attention-based fusion computes weights that decide how much "
      "each modality contributes for a given patient, and gated fusion multiplies each embedding by a learned gate "
      "between zero and one. Attention and gating can down-weight a noisy or missing modality, at the cost of more "
      "parameters and data. The original specification of this project allowed all three; the implementation uses "
      "concatenation followed by a learned two-layer fusion network, the simplest of the options, because both of "
      "the implemented modalities are present for every patient in the cohort.")
    p("Clinical free text is a third modality that the original specification intended to use. Transformer language "
      "models such as BERT [@bert] and their clinical adaptations, for example the Bio+Clinical BERT models trained "
      "on MIMIC-III notes [@alsentzer], produce note embeddings that can be fused with structured data. This "
      "requires temporally valid free-text notes; the prepared eICU files used in this project contain no such "
      "notes, which is why the text modality was not implemented (Section 8.7).")
    h2("2.5 Graph neural networks and patient similarity")
    p("Graph neural networks compute node representations by aggregating information from neighbours. The graph "
      "convolutional network (GCN) of Kipf and Welling uses a normalised neighbourhood average [@gcn]; graph "
      "attention networks (GAT) learn attention weights over neighbours [@gat]; GraphSAGE learns separate "
      "transformations of a node and of an aggregate (for example the mean) of its neighbours, and is inductive, so "
      "it can embed nodes not seen during training [@hamilton]. Inductiveness matters here because every hospital "
      "and every test patient is new to the model.")
    p("Rocheteau et al. applied this idea to eICU: an LSTM encoded each patient's time series, patients were linked "
      "by diagnosis similarity, and a GNN refined the representations; the hybrid LSTM-GNN improved length-of-stay "
      "prediction over an LSTM alone [@rocheteau]. Their graph spans the whole centralised dataset. The present "
      "project keeps the graph inside a single hospital (in fact inside a batch of one hospital's patients), as "
      "federated learning requires, and builds edges from the learned fused embeddings rather than from diagnoses, "
      "because diagnoses recorded after the prediction time would leak future information.")
    p("How the graph is built is a design decision in its own right. Some problems come with a given graph, such as a "
      "citation network; for patients, the graph must be constructed. Common choices are to connect patients that "
      "share a diagnosis or other attribute, or to connect each patient to its k nearest neighbours in a feature or "
      "embedding space. A k-nearest-neighbour graph has a fixed number of edges per node, which keeps the cost "
      "linear in the number of patients and avoids very dense neighbourhoods. Transductive methods such as the "
      "original GCN need the whole graph during training [@gcn], whereas inductive methods such as GraphSAGE learn "
      "functions that can be applied to new graphs [@hamilton]; this is why GraphSAGE fits a setting in which every "
      "batch and every hospital forms a new graph.")
    h2("2.6 Federated learning in healthcare")
    p("Federated averaging (FedAvg) trains a global model by repeatedly sending it to a sample of clients, running a "
      "few local optimisation steps on each client's data and averaging the returned weights in proportion to the "
      "amount of local data [@mcmahan]. Surveys identify the main challenges as statistically heterogeneous "
      "(non-IID) clients, systems heterogeneity, communication cost and privacy [@li], [@kairouz]. In medicine, "
      "Sheller et al. showed for brain-tumour segmentation across ten institutions that federated training reached "
      "about 99% of the quality of centrally pooled training [@sheller], and Dayan et al. trained a COVID-19 outcome "
      "model across twenty institutions on five continents [@dayan]. Rieke et al. discuss how federated learning "
      "could support digital health while noting that sending model updates alone does not guarantee privacy "
      "[@rieke]. Formal protections such as secure aggregation [@bonawitz] and differentially private training "
      "[@abadi] are separate mechanisms; neither is implemented in this project.")
    p("Statistical heterogeneity is particularly relevant for hospital data. Hospitals differ in size, case mix, "
      "documentation practice and the set of laboratory tests and drugs they record, so the local data distributions "
      "are not identically distributed. Under such conditions the local models drift apart during local training, "
      "and averaging them can converge more slowly than centralised training [@li], [@kairouz]. Limiting the number "
      "of local steps, sampling a subset of clients per round and weighting clients by their data size are simple "
      "ways to control this drift; all three are used in this project (Section 4.11).")
    h2("2.7 Evaluation under class imbalance")
    p("Mortality and especially sepsis are rare outcomes. The area under the ROC curve (AUROC) measures ranking and is "
      "insensitive to prevalence, so it can look reassuring even when few positive predictions are correct. Saito "
      "and Rehmsmeier showed that precision-recall curves and the area under them (AUPRC) are more informative for "
      "imbalanced problems [@saito]. The Brier score, the mean squared error of predicted probabilities, measures "
      "calibration and accuracy of the probabilities [@brier]. For these reasons this report reports AUROC, AUPRC, "
      "Brier score and threshold-dependent metrics together and does not report plain accuracy, which would be "
      "dominated by the majority class.")
    p("Threshold-dependent metrics also depend on how the threshold is chosen. A threshold tuned on the test data "
      "gives an optimistic estimate, because the same labels are used to choose the operating point and to measure "
      "it. The usual practice, followed in this report, is to fix the threshold on validation data and then apply it "
      "once to the test data, and to report the default threshold of 0.5 alongside it.")
    h2("2.8 Summary and research gap")
    tab("Comparison of related approaches", ["Work", "Data", "Modalities", "Graph", "Federated", "Outcome"],
        [["APACHE IV [@zimmerman]", "104 US ICUs", "Tabular (day-1)", "No", "No", "Mortality"],
         ["Harutyunyan et al. [@harut]", "MIMIC-III", "Time series", "No", "No", "Mortality and others"],
         ["Nemati et al. [@nemati]", "2 hospitals + MIMIC-III", "Tabular + time series", "No", "No", "Sepsis-3 onset"],
         ["Reyna et al. [@reyna]", "3 hospital systems", "Hourly time series", "No", "No", "Sepsis-3 onset"],
         ["Rocheteau et al. [@rocheteau]", "eICU (pooled)", "Time series + diagnoses", "Whole dataset", "No", "Length of stay, mortality"],
         ["Sheller et al. [@sheller]", "10 institutions", "Imaging", "No", "Yes", "Segmentation"],
         ["This project", "eICU, 207 hospitals", "Tabular + time series (fused)", "Within hospital batch", "Simulated FedAvg", "Mortality + sepsis proxy"]],
        [1.55, 1.1, 1.25, 0.95, 0.85, 1.0])
    tab("Contribution of the reviewed works and their use in this project", ["Ref.", "Main contribution", "Use in this project"],
        [["[@zimmerman]", "APACHE IV hospital-mortality model from day-1 data", "APACHE IV scores used as tabular inputs"],
         ["[@sepsis3], [@sofa]", "Sepsis-3 definition; SOFA organ-dysfunction score", "Reference definition; explains why the label is a proxy"],
         ["[@eicu], [@physionet]", "Multi-centre ICU database and its distribution platform", "Data source; hospital identifiers define the clients"],
         ["[@lstm], [@harut]", "LSTM; multitask ICU time-series benchmarks", "Time-series encoder; shared trunk with two heads"],
         ["[@rajkomar]", "Deep learning on full EHR data", "Motivation for learned representations"],
         ["[@nemati], [@reyna]", "Early sepsis prediction; multi-system challenge", "Context for the sepsis task and hospital-level evaluation"],
         ["[@baltrusaitis]", "Taxonomy of multimodal learning and fusion", "Intermediate fusion of two embeddings"],
         ["[@bert], [@alsentzer]", "Transformer language models; clinical BERT", "Planned text encoder (not implemented)"],
         ["[@gcn], [@gat], [@hamilton]", "GCN, GAT and GraphSAGE graph networks", "GraphSAGE used; GCN and GAT coded only"],
         ["[@rocheteau]", "LSTM-GNN on eICU with a diagnosis graph", "Closest prior work; graph kept within hospitals here"],
         ["[@mcmahan], [@li], [@kairouz]", "FedAvg; challenges of federated learning", "Aggregation rule and client sampling"],
         ["[@sheller], [@dayan], [@rieke]", "Federated learning across medical institutions", "Motivation for hospital-level federation"],
         ["[@bonawitz], [@abadi]", "Secure aggregation; differential privacy", "Not implemented; listed as future work"],
         ["[@saito], [@brier]", "PR curves for imbalance; Brier score", "Choice of AUPRC and Brier score"]],
        [1.0, 2.7, 2.57])
    p("Few studies combine all four ingredients: modality-specific encoders with fusion, a patient graph, "
      "hospital-level federated training and evaluation on unseen hospitals. This project implements and evaluates "
      "that combination for two modalities on eICU, while being explicit that the federation is simulated and that "
      "the text modality of the original plan is missing.")

    # ── Chapter 3 ───────────────────────────────────────────────────────────
    h1(3, "PROBLEM DEFINITION AND OBJECTIVES")
    h2("3.1 Formal problem formulation")
    p("Let hospital *k* hold ICU stays *i* with a tabular vector *x_i* (28 values) and a time-series matrix *S_i* of six "
      "4-hour bins with 74 features per bin, all taken from the first 24 hours of the stay. The task is to learn a "
      "single model *f* that outputs two probabilities: *p_i^mort* for in-hospital death (`hospital_expired`) and "
      "*p_i^sep* for sepsis first documented after 24 hours (`sepsis_after_24h`). The model is trained by federated "
      "averaging over the training hospitals; no stay of a validation or test hospital is used for training, and "
      "the test hospitals are used only once, after the checkpoint has been fixed.")
    h2("3.2 Dataset")
    p(f"The data are prepared per-hospital files derived from eICU [@eicu]. The configured split contains 207 "
      f"hospitals with both tabular and time-series files and {m['SEP']['data_counts']['tabular_rows']:,} ICU stays. "
      "Hospitals are assigned to splits at random with a fixed seed (42): 145 training, 31 validation and 31 test "
      "hospitals. Splitting by hospital rather than by patient prevents a hospital's practice patterns from "
      "appearing in both training and testing and mirrors the situation of a new hospital joining the federation.")
    tab("Dataset and cohort summary (verified from the saved manifests)", ["Item", "Training", "Validation", "Test", "Total"],
        [["Hospitals", "145", "31", "31", "207"],
         ["Stays of at least 24 h (24 h cohort)", "93,764", "21,100", "18,014", "132,878"],
         ["Sepsis cohort (after exclusions)", f"{sc['train']:,}", f"{sc['val']:,}", f"{sc['test']:,}", f"{sum(sc.values()):,}"],
         ["Sepsis positives", f"{lc['train']['positive']:,}", f"{lc['val']['positive']:,}", f"{lc['test']['positive']:,}", f"{pos_all:,}"],
         ["Sepsis prevalence", f"{lc['train']['positive'] / sc['train']:.2%}", f"{lc['val']['positive'] / sc['val']:.2%}",
          f"{lc['test']['positive'] / sc['test']:.2%}", f"{pos_all / sum(sc.values()):.2%}"]],
        [2.4, 1.0, 1.0, 1.0, 1.0])
    h2("3.3 Objectives")
    bl(["**O1:** a reproducible, patient-aligned data pipeline with a hospital-level split and an explicit 24-hour "
        "observation window.",
        "**O2:** a two-modality model (MLP + LSTM) with a learned fusion module.",
        "**O3:** an intra-hospital patient-similarity graph and GraphSAGE refinement of the fused embeddings.",
        "**O4:** FedAvg training with validation-based checkpoint selection and a No-GNN ablation.",
        "**O5:** a mortality + sepsis model with a documented, leakage-controlled sepsis label.",
        "**O6:** honest evaluation on held-out hospitals with AUROC, AUPRC, Brier score, precision, recall, F1 and "
        "specificity.",
        "**O7:** a working demonstration: inference API, web interface and containerised deployment."])
    h2("3.4 Requirements")
    p("Functional requirements are the seven objectives above. Non-functional requirements follow from the setting: "
      "training must run on a laptop CPU in reasonable time; every result must be reproducible from a fixed seed and "
      "saved configuration; restricted patient data must never leave the local machine or enter version control; and "
      "the demonstration must fail visibly rather than display substitute predictions when a model is missing.")

    # ── Chapter 4 ───────────────────────────────────────────────────────────
    h1(4, "PROPOSED METHODOLOGY AND WORK DESCRIPTION")
    p("Figure 4.1 gives an overview of the implemented model. Two modality-specific encoders produce 64-dimensional "
      "embeddings that the fusion module combines into a 128-dimensional patient vector. A k-nearest-neighbour graph "
      "is built from these vectors inside each batch of patients from one hospital, GraphSAGE refines them, and two "
      "heads predict the two outcomes. The same model is trained locally at each sampled hospital and combined by "
      "FedAvg. There is no text encoder in the trained model.")
    fig("fig_architecture.png", "Architecture of the implemented two-modality federated GNN (mortality + sepsis model)")
    h2("4.1 Data source and preparation")
    p("Each hospital's data are stored in two files: a tabular file with one row per ICU stay and a time-series file "
      "with one row per stay and 4-hour bin (`hour_bin` counts 4-hour blocks from ICU admission). These per-hospital "
      "feature files were produced in an earlier preprocessing stage from the eICU source tables (documented in the "
      "repository's data hand-off notes); the scripts that created them are not part of the current repository, so "
      "this report describes the files as they are and the processing that the project code applies to them.")
    p("The tabular file contains 67 canonical features: demographics (age, with ages above 89 recorded as 90, sex, "
      "BMI and one-hot ethnicity), the time from hospital to ICU admission, the ICU visit number, APACHE IV scores and "
      "predictions, eight chronic comorbidity flags, the day-1 Glasgow Coma Scale components, PaO2, FiO2 and "
      "creatinine, 18 diagnosis-category features (`diag_*`) and 21 admission-drug features (`admitdrug_*`). The "
      "time-series file contains 74 features per bin: 20 laboratory values (stored as z-scores of the bin mean, "
      "using global means and standard deviations), and 0/1 indicators for 20 infusions, 20 medications and 14 "
      "treatment categories recorded in the bin. The outcome columns `hospital_expired` and `icu_expired` and the "
      "discharge offsets are stored separately from the features.")
    h3("4.1.1 Loading, alignment and the 24-hour window")
    p("The loader (`src/datasets.py`) reads one hospital at a time. It keeps only stays with an ICU length of stay of "
      "at least 1,440 minutes, so that the full 24-hour window is observed and the outcome cannot occur inside the "
      "input window. Time-series rows are restricted to bins 0–5 (0–24 h), sorted, padded to six steps and "
      "accompanied by their true length for sequence packing. Tabular and time-series rows are paired by patient "
      "identifier in a single dataset object; a stay missing either modality is dropped, and every item asserts that "
      "both modalities belong to the same patient. This pairing replaced an earlier implementation that shuffled the "
      "two modalities separately and misaligned patients. Missing canonical columns are filled with zero and "
      "tabular features are z-scored within each hospital using that hospital's own cohort statistics; this "
      "normalisation is label-free and is applied in the same way to training, validation and test hospitals.")
    h3("4.1.2 Hospital-level split and data validation")
    p("The split is stored in `processed/hospital_split.json` and reused by every script. A validation script "
      "checks ten conditions before training: the patient-level files exist for every split hospital, identifiers "
      "are unique, the 67 + 74 canonical columns are available, the time series covers bins 0–5, both targets are "
      "binary without missing values, the folders match the split, the preprocessing files exist, the cohorts are "
      "non-empty, the splits do not overlap and no patient appears in two hospitals. On the local data all checks "
      "passed; one validation hospital has no laboratory columns, which the loader fills with zeros.")
    h2("4.2 Two prediction tasks and two experiments")
    p("Two experiments with different targets were run and must not be confused. The **mortality + ICU-mortality "
      "model** predicts `hospital_expired` and `icu_expired` from all 67 tabular features. The **mortality + sepsis "
      "model**, which corresponds to the original mortality-and-sepsis objective, predicts `hospital_expired` and "
      "`sepsis_after_24h`, uses a restricted feature set (Section 4.4) and a smaller cohort. Unless stated "
      "otherwise, the model in this report is the mortality + sepsis model. ICU mortality is never used as a "
      "substitute for sepsis.")
    h2("4.3 Sepsis label construction")
    p("The sepsis label is built by `scripts/build_sepsis_labels.py` using the rule in `src/sepsis_task.py`. A "
      "diagnosis qualifies when one segment of its pipe-separated `diagnosisstring` path is exactly *sepsis* or "
      "*septic shock*, for example `cardiovascular|shock / hypotension|sepsis|severe`. Strings for SIRS alone "
      "(\"signs and symptoms of sepsis (SIRS)\"), septic arthritis, aseptic meningitis and complications recorded "
      "as \"... due to sepsis\" do not qualify. The time of a diagnosis is `diagnosisoffset`, measured in minutes "
      "from ICU admission; the cutoff is 1,440 minutes.")
    tab("Sepsis label rule", ["Case", "Rule", "Label"],
        [["Prior sepsis", "First qualifying diagnosis at or before 1,440 min", "Excluded"],
         ["Incident sepsis", "First qualifying diagnosis after 1,440 min and no later than ICU discharge", "Positive (1)"],
         ["No sepsis", "Diagnosis rows exist, no qualifying diagnosis during the ICU stay", "Negative (0)"],
         ["No diagnosis data", "No diagnosis rows for the stay", "Excluded"]],
        [1.3, 3.8, 1.2])
    fig("fig_sepsis_label.png", "Construction of the sepsis label relative to the 24-hour prediction cutoff")
    p(f"In the data, {len(m['LAB']['qualifying_phrases'])} distinct qualifying phrases occur. Within the 24-hour "
      f"cohort of 132,878 stays, {nodx:,} stays had no diagnosis rows and {prior:,} had sepsis documented by 24 hours; "
      f"both groups were excluded. The resulting sepsis cohort has {sum(sc.values()):,} stays with {pos_all:,} "
      f"positives (Figure 4.3). The label therefore means *sepsis first documented after the first 24 hours*. It is a "
      "documentation-time proxy: a diagnosis entry can be recorded hours or days after the clinical onset, some "
      "patients with sepsis may never receive a sepsis diagnosis entry, and the definition is not Sepsis-3 "
      "[@sepsis3]. The diagnosis text is used only to construct the target; it is never an input to the model.")
    fig("fig_cohort_flow.png", "Cohort flow from all stays to the sepsis cohort", 5.6)
    h2("4.4 Leakage prevention")
    p("A prediction made at 24 hours must not use information recorded later. Two groups of tabular features "
      "violated this for the sepsis task. The 18 `diag_*` features summarise diagnoses from the whole stay; a check "
      "on five hospitals showed that every patient whose infectious-disease diagnoses appear only after 24 hours "
      "(71 of 71 such patients) is still flagged in `diag_infectious diseases`, so these features would reveal the "
      "label. For the 21 "
      "`admitdrug_*` features, 46.7% of admission-drug entries are recorded more than 1,440 minutes after ICU "
      "admission. Both groups were removed for the sepsis model, leaving 28 tabular features: demographics, "
      "admission offset and visit number, APACHE IV scores and predictions, comorbidity flags, the GCS components and "
      "the day-1 PaO2, FiO2 and creatinine. The 74 time-series features are binned by time and limited to the first "
      "24 hours by the loader. The diagnosis-derived features remain in the earlier mortality + ICU-mortality model, "
      "where a small possible timing leak is acknowledged.")
    h2("4.5 Tabular encoder")
    p("The tabular encoder is a three-layer multilayer perceptron: Linear(28→128), batch normalisation, ReLU and "
      "dropout; Linear(128→64), batch normalisation, ReLU and dropout; and a final Linear(64→64) that produces the "
      "64-dimensional tabular embedding *z_tab* (Equation 4.1). Dropout is 0.2 in training.")
    eq("eq_mlp.png", "4.1")
    h2("4.6 Time-series encoder")
    p("The six bins of 74 features are processed by a two-layer LSTM with 64 hidden units [@lstm]. Sequences are "
      "packed by their true length (at most six) so that padded steps do not influence the result. The last valid "
      "hidden state of the second layer is projected by a linear layer to the 64-dimensional temporal embedding "
      "*z_ts* (Equation 4.2, Figure 4.4).")
    eq("eq_lstm.png", "4.2")
    fig("fig_encoders.png", "Tabular MLP encoder (a) and time-series LSTM encoder over six 4-hour bins (b)")
    h2("4.7 Multimodal fusion")
    p("The `MultimodalFusion` module concatenates the two embeddings into a 128-dimensional vector and passes it "
      "through Linear(128→128), batch normalisation, ReLU, dropout, Linear(128→128), batch normalisation and ReLU "
      "(Equation 4.3). The result *h_i* is the unified patient representation; it is the input to graph "
      "construction and to GraphSAGE. Because the fusion layers are learned jointly with both encoders, gradients "
      "of the loss reach the tabular MLP and the LSTM through the fusion module. A unit test confirms that a "
      "training step updates the parameters of both encoders and of the fusion module, and that GraphSAGE receives "
      "exactly the fused vector.")
    eq("eq_fusion.png", "4.3")
    h2("4.8 Patient-similarity graph")
    p("The graph is built dynamically for every batch of 32 patients from one hospital, using the cosine similarity "
      "of the fused vectors (Equation 4.4). Each patient receives directed edges from its five most similar patients "
      "in the batch (self-edges excluded); edges are not symmetrised. Because a batch never mixes hospitals, no edge "
      "crosses a hospital boundary. The graph is therefore a batch-level graph inside a hospital, not one persistent "
      "graph of all patients of a hospital; this keeps the memory and computation small and matches the way the "
      "model is trained and evaluated (Figure 4.5).")
    eq("eq_knn.png", "4.4")
    fig("fig_fusion_graph.png", "Multimodal fusion (a) and k-NN patient graph within one batch (b, schematic with random points)")
    h2("4.9 GraphSAGE and prediction heads")
    p("Two GraphSAGE layers with mean aggregation [@hamilton] refine the fused vectors (128→64→64). Each layer "
      "combines a transformation of the patient's own vector with a transformation of the mean of its neighbours, "
      "followed by batch normalisation, ReLU and a residual connection (Equation 4.5); dropout is applied between "
      "layers. GraphSAGE is used through PyTorch Geometric [@pyg]. Two separate heads, each Linear(64→32), batch "
      "normalisation, ReLU, dropout and Linear(32→1), output the logits for in-hospital mortality and for sepsis. In "
      "the No-GNN ablation the graph and GraphSAGE are removed and the heads read the fused vector directly; "
      "everything else is identical.")
    eq("eq_sage.png", "4.5")
    h2("4.10 Multitask loss with class weighting")
    p("The loss is the sum of two binary cross-entropy terms with positive-class weights (Equation 4.6). The mortality "
      f"weight is 3.0, as in the earlier experiments. Sepsis is much rarer (1.97% of training stays), and a first "
      f"training run with weight 3.0 produced no positive sepsis predictions, so the final run uses a class weight "
      f"equal to the ratio of negatives to positives in the training hospitals, {pw} (Equation 4.7). Only training "
      "labels enter this ratio.")
    eq("eq_loss.png", "4.6")
    eq("eq_posweight.png", "4.7")
    h2("4.11 Federated training with FedAvg")
    p("Training is simulated in one process (Figure 4.6). In each of 50 communication rounds, ten of the 145 training "
      "hospitals are sampled. Each sampled client copies the global model, trains it on its own patients for one "
      "local epoch with Adam [@adam] (learning rate 0.001, weight decay 1e-5, batch size 32, gradient clipping at "
      "5.0) and at most 30 optimiser steps, and returns its weights and the number of examples it trained on. The "
      "server averages the floating-point parameters weighted by these counts (Equation 4.8) [@mcmahan]; integer "
      "batch-normalisation counters keep the global value. After every round the global model is evaluated on all "
      "31 validation hospitals, and the round with the lowest validation loss is kept as the checkpoint. The test "
      "hospitals are scored only once, after training has finished.")
    eq("eq_fedavg.png", "4.8")
    fig("fig_fedavg.png", "FedAvg workflow across simulated hospital clients")
    p("The per-client step limit keeps very large hospitals from dominating a round and keeps the run time small; a "
      f"consequence is that not every training patient is seen. In the final run the GNN model took "
      f"{gt['optimizer_steps']:,} optimiser steps on {gt['examples_trained']:,} examples, covering "
      f"{gt['unique_patients_trained']:,} unique training patients from {gt['hospitals_sampled']} of the 145 "
      "training hospitals. No client failed.")

    # ── Chapter 5 ───────────────────────────────────────────────────────────
    h1(5, "PROPOSED ALGORITHMS")
    h2("5.1 Algorithm 1: client-side local training")
    tab("Algorithm 1: LocalTrain(w, k)", ["Step", "Operation"],
        [["Input", "global weights w, hospital k, local epochs E = 1, batch size B = 32, step limit T = 30"],
         ["1", "Load hospital k with the task loader (24 h window, cohort rule, patient pairing, leak-free columns)"],
         ["2", "Copy the global model; create Adam (lr 0.001, weight decay 1e-5)"],
         ["3", "for each epoch and each shuffled batch of size >= 2 (BatchNorm needs two samples):"],
         ["3a", "  z_tab = MLP(x), z_ts = LSTM(S), h = Fusion(z_tab, z_ts)"],
         ["3b", "  build the k-NN graph on h (cosine, k = 5); g = GraphSAGE(h, edges)"],
         ["3c", "  compute the weighted multitask loss (Eq. 4.6); back-propagate; clip gradients at 5.0; Adam step"],
         ["3d", "  count examples n_k; stop after T steps"],
         ["Output", "local weights w_k and example count n_k (no patient data leave the client)"]],
        [0.8, 5.5])
    h2("5.2 Algorithm 2: server-side federated averaging")
    tab("Algorithm 2: FederatedTraining", ["Step", "Operation"],
        [["Input", "training hospitals H_train (145), validation hospitals H_val (31), rounds R = 50, clients C = 10, seed 42"],
         ["1", "Initialise the global model w(0); best = +inf"],
         ["2", "for t = 1 .. R:"],
         ["2a", "  sample C hospitals from H_train with a seeded generator"],
         ["2b", "  for each sampled hospital k: (w_k, n_k) = LocalTrain(w(t-1), k)"],
         ["2c", "  w(t) = sum_k (n_k / n) w_k for floating-point tensors (Eq. 4.8)"],
         ["2d", "  L_val = loss of w(t) on all patients of H_val; log AUROC, AUPRC, F1"],
         ["2e", "  if L_val < best: best = L_val; save w(t) as the checkpoint"],
         ["3", "Score the saved checkpoint once on the 31 test hospitals; write the manifest (path, size, SHA-256)"]],
        [0.8, 5.5])
    h2("5.3 Algorithm 3: inference for one hypothetical patient")
    p("The web interface lets a user type a hypothetical patient. Because the model builds its graph inside a batch, "
      "a single patient alone would have no neighbours. The inference algorithm therefore (1) validates the form "
      "values against training-split ranges, (2) scales the tabular values with the statistics of a fixed reference "
      "training hospital and converts laboratory values to z-scores with the global laboratory statistics, exactly "
      "as the training loader does, (3) places the patient in a batch with 31 training patients of the same "
      "reference hospital, (4) runs both encoders, fusion, the k-NN graph, GraphSAGE and both heads under "
      "evaluation mode without gradients, and (5) returns the two probabilities of the hypothetical patient. A test "
      "confirms that this preprocessing reproduces the training loader's tensors exactly for real training "
      "patients.")

    # ── Chapter 6 ───────────────────────────────────────────────────────────
    h1(6, "PROPOSED FLOWCHART / BLOCK DIAGRAM / DFD")
    h2("6.1 System block diagram")
    p("The block diagram of the trained model was shown in Figure 4.1. Its data path is tabular features → MLP and "
      "time series → LSTM, fusion of the two embeddings, the batch-level k-NN graph, GraphSAGE and the two heads. The "
      "same diagram applies to every hospital client.")
    h2("6.2 Data flow diagram")
    p("Figure 6.1 shows how data move through the project, from the local eICU source tables to the prepared files, "
      "the label builder, federated training, test evaluation and the inference service. Raw and derived patient "
      "data stay on the local machine; only aggregate metrics, the model weights and their manifest are produced.")
    fig("fig_dfd.png", "Data flow diagram of the project")
    h2("6.3 Training and evaluation flow")
    p("The experiment flow follows Chapter 5: data validation, label construction, 50 rounds of FedAvg with an "
      "evaluation on the validation hospitals after each round, checkpoint selection by the lowest validation loss, "
      "and a single evaluation on the test hospitals. Figure 6.2 shows this flow as a flowchart, and Figure 6.3 shows "
      "the per-round validation curves that it produced for the two sepsis-task models together with the rounds that "
      "were selected. The validation loss falls quickly in the first ten rounds and then flattens, while the "
      "validation AUROC curves are noisier because each round trains on a different sample of ten hospitals. The "
      "deployment flow of the demonstration is shown in Chapter 7.")
    fig("fig_flowchart.png", "Flowchart of federated training, checkpoint selection and test evaluation", 4.6)
    fig("fig_training_curves.png", "Validation loss and validation AUROC per round for both sepsis-task models (dotted lines: selected checkpoints)")

    # ── Chapter 7 ───────────────────────────────────────────────────────────
    h1(7, "IMPLEMENTATION (TOOLS AND TECHNOLOGY USED)")
    h2("7.1 Software environment")
    tab("Software and hardware used for the final runs", ["Component", "Version / detail"],
        [["Operating system", "Windows 11 (10.0.26200), 64-bit"],
         ["Processor", "AMD Ryzen 7 7445HS (CPU only; CUDA not used)"],
         ["Python", env["python"]], ["PyTorch", env["torch"] + " [@pytorch]"],
         ["PyTorch Geometric", env["torch_geometric"] + " [@pyg]"], ["NumPy / pandas / PyArrow", f"{env['numpy']} / {env['pandas']} / {env['pyarrow']}"],
         ["scikit-learn", env["scikit_learn"] + " [@sklearn]"], ["Frontend", "React 19.3, Vite 8.3.3, react-force-graph-2d 1.29.2"],
         ["Containers", "Docker Engine 29.9.2, Docker Compose 5.6.1, nginx (stable-alpine), python:3.14-slim"]],
        [2.0, 4.3])
    h2("7.2 Code organisation")
    tab("Main modules of the repository", ["Path", "Purpose"],
        [["src/datasets.py", "Per-hospital loading, 24 h window, cohort rule, patient pairing"],
         ["src/encoders.py, src/fusion.py", "Tabular MLP, time-series LSTM, multimodal fusion"],
         ["src/graph_builder.py, src/gnn.py", "Batch k-NN graph; GraphSAGE (GAT and GCN available, not evaluated)"],
         ["src/models.py", "End-to-end model, prediction heads, multitask loss"],
         ["src/sepsis_task.py, scripts/build_sepsis_labels.py", "Sepsis rule, leak-free loader, label building"],
         ["src/splits.py, src/evaluation.py, src/metrics.py", "Hospital split, shared evaluation, metrics"],
         ["scripts/train_federated.py", "FedAvg training (local clients, aggregation, checkpoint selection)"],
         ["scripts/validate_data.py, scripts/final_model.py", "Data checks; train / evaluate / check / predict with manifests"],
         ["src/interactive_inference.py, demo/server.py", "Form-based inference and the HTTP API"],
         ["services/, compose.yaml, run-demo.ps1", "Three-container deployment and launch scripts"],
         ["frontend-playground/", "React web interface"]],
        [2.6, 3.7])
    h2("7.3 Training configuration")
    tab("Training configuration of the final mortality + sepsis runs", ["Setting", "Value"],
        [["Algorithm", "FedAvg, simulated in one process"], ["Seed", "42 (model, sampling, batch order); split seed 42"],
         ["Rounds / clients per round", "50 / 10 of 145 training hospitals"], ["Local training", "1 epoch, at most 30 steps, batch 32"],
         ["Optimiser", "Adam, learning rate 0.001, weight decay 1e-5, gradient clip 5.0"],
         ["Graph", "cosine k-NN, k = 5, within each batch of one hospital"],
         ["GNN", "GraphSAGE, 2 layers (128→64→64), mean aggregation, residual"],
         ["Dropout", "0.2"], ["Loss weights", f"mortality pos_weight 3.0, sepsis pos_weight {pw}"],
         ["Checkpoint", "lowest validation loss over 50 rounds"],
         ["Parameters", f"{G['architecture']['parameters']:,} (GNN), {NG['architecture']['parameters']:,} (No-GNN)"]],
        [2.2, 4.1])
    h2("7.4 Experiment management and reproducibility")
    p("Every training run writes its arguments, git commit, task, removed features and label file to `run_info.json`, "
      "its per-round log to `training_history.json`, and the best checkpoint. A run counts as complete only when its "
      "checkpoint loads and its history is valid; incomplete runs are reported instead of being skipped silently, an "
      "earlier bug in the experiment runner. Recorded results in `results/corrected/` are protected from being "
      "overwritten. After training, `scripts/final_model.py` scores the test hospitals once and writes a manifest "
      "with the checkpoint path, size and SHA-256 hash, the architecture, the configuration actually used, the data "
      "counts and the metrics; inference refuses a checkpoint whose hash does not match. Repeating the earlier "
      "mortality + ICU-mortality run with the same seed reproduced the recorded round-by-round history exactly, "
      "which confirms that the CPU pipeline is deterministic.")
    h2("7.5 Testing")
    p("The repository contains 16 test files (87 test cases in the latest run, all passing). Data-free tests use small "
      "synthetic hospitals and cover patient alignment, the 24-hour window and cohort, target definitions, "
      "the sepsis rule on constructed cases, the leak-free feature list, model shapes, multimodal fusion and gradient "
      "flow into both encoders, FedAvg weighting, the hospital split, metrics, checkpoint save and reload, the "
      "run-completeness rules and the protection of recorded results. Tests that need the local eICU files (for "
      "example the preprocessing parity of the interactive inference with the training loader) run only when those "
      "files are present.")
    h2("7.6 Inference API and web interface")
    p("The React interface (Figure 7.1 and Figure 7.2) has two clearly separated modes. A synthetic-demonstration mode "
      "uses generated data and a separately trained demo model to show the software pipeline (overview, patient "
      "graph, federation log, evaluation). The real-model mode has three tabs: the mortality + ICU-mortality model, "
      "the mortality + sepsis model, and an **Interactive prediction** page. On that page the user edits quick "
      "fields or all 28 tabular and 74 × 6 time-series inputs, presses Predict and receives the two probabilities "
      "computed by the saved checkpoint through `POST /api/sepsis/predict-input`. Defaults are training-split medians, "
      "values outside the 0.1st–99.9th percentile of the training data are rejected, and the page states that the "
      "input is hypothetical. If the checkpoint is missing or invalid the page shows the error instead of a "
      "substitute result. Probabilities are shown with the 0.5 threshold; the validation-chosen threshold is "
      "displayed only as a reference.")
    fig("fig_ui_interactive_results.png", "Interactive prediction page: hypothetical example input and the probabilities returned by the trained checkpoint", 5.3)
    fig("fig_ui_pipeline_metrics.png", "Model pipeline and held-out test metrics shown on the Interactive prediction page", 5.3)
    h2("7.7 Containerised deployment")
    p("For the demonstration, the project runs as three Docker containers (Figure 7.3). The **frontend** container "
      "serves the built React application with nginx and forwards `/api/` requests to the **backend** container, a "
      "small standard-library service that validates form payloads and forwards them. The **model** container loads "
      "and verifies the checkpoint and runs inference; its health check succeeds only after the checkpoint hash, "
      "the architecture and the graph context have been verified. Only the frontend is published, on 127.0.0.1. The "
      "data, checkpoints and results are mounted read-only and are not copied into any image. A PowerShell launcher "
      "selects a free port between 8765 and 8795 without stopping other programs, starts the containers in the "
      "order model → backend → frontend and opens the browser when all three are healthy. These steps, including the "
      "fallback when the preferred port is occupied and the error shown when the model container is stopped, were "
      "tested.")
    fig("fig_docker.png", "Three-container deployment of the demonstration")
    h2("7.8 Deployment verification")
    p("The deployment was checked with the launcher and the running containers. Table 7.4 lists the checks and their "
      "outcomes. In every failure case the user sees an explicit error message; the interface never falls back to "
      "a substitute or synthetic prediction.")
    tab("Checks of the containerised demonstration", ["Check", "Expected behaviour", "Outcome"],
        [["Normal start", "All three containers report healthy; /api/health names the checkpoint", "Passed (healthy after about 29 s)"],
         ["Preferred port 8765 already in use", "Launcher leaves the other program alone and selects the next free port", "Passed (8766, later 8767)"],
         ["Model container stopped", "Prediction request returns an error that the page displays", "Passed"],
         ["Checkpoint file corrupted", "SHA-256 differs from the manifest; the launcher stops with an error", "Passed"],
         ["Checkpoint file missing", "The launcher stops before starting the containers", "Passed"]],
        [1.7, 3.1, 1.47])

    # ── Chapter 8 ───────────────────────────────────────────────────────────
    h1(8, "RESULT DISCUSSION AND ANALYSIS")
    h2("8.1 Experimental setup")
    p(f"Both sepsis-task models (FedAvg + GraphSAGE and the No-GNN ablation) were trained with the configuration of "
      f"Table 7.3, completed all 50 rounds and were scored once on the {n_test:,} sepsis-cohort stays of the 31 test "
      f"hospitals. The GNN checkpoint was selected at round {gt['selected_round']} (validation loss "
      f"{gt['best_val_loss']}), the No-GNN checkpoint at round {nt['selected_round']} (validation loss "
      f"{nt['best_val_loss']}). Test prevalence was {gm['prevalence']:.2%} for in-hospital mortality and "
      f"{gs['prevalence']:.2%} for sepsis. All results come from a single seed (42).")
    h2("8.2 Evaluation metrics")
    p("With TP, FP, TN and FN the counts at a decision threshold, the threshold-dependent metrics are defined in "
      "Equation 8.1, and the Brier score in Equation 8.2 [@brier]. AUROC is the probability that a random positive is "
      "ranked above a random negative; AUPRC (average precision) summarises the precision-recall curve and is the "
      "more informative ranking metric for rare outcomes [@saito]. AUROC and AUPRC are computed with scikit-learn "
      "[@sklearn]; they do not depend on a threshold.")
    eq("eq_metrics.png", "8.1")
    eq("eq_brier.png", "8.2")
    h2("8.3 Results on the held-out test hospitals")
    tab(f"Test results of the mortality + sepsis models ({n_test:,} stays, threshold 0.5; both models trained with FedAvg; Mortality = in-hospital mortality, Sepsis = sepsis first documented after 24 h)",
        ["Model", "Task", "AUROC", "AUPRC", "Brier", "Precision", "Recall", "F1", "Specificity"],
        [["GNN", "Mortality", f3(gm["auroc"]), f3(gm["auprc"]), f3(gm["brier"]), f3(gm["precision"]), f3(gm["recall"]), f3(gm["f1"]), f3(gm["specificity"])],
         ["GNN", "Sepsis", f3(gs["auroc"]), f3(gs["auprc"]), f3(gs["brier"]), f3(gs["precision"]), f3(gs["recall"]), f3(gs["f1"]), f3(gs["specificity"])],
         ["No-GNN", "Mortality", f3(nm["auroc"]), f3(nm["auprc"]), f3(nm["brier"]), f3(nm["precision"]), f3(nm["recall"]), f3(nm["f1"]), f3(nm["specificity"])],
         ["No-GNN", "Sepsis", f3(ns["auroc"]), f3(ns["auprc"]), f3(ns["brier"]), f3(ns["precision"]), f3(ns["recall"]), f3(ns["f1"]), f3(ns["specificity"])]],
        [0.85, 0.75, 0.66, 0.66, 0.55, 0.74, 0.6, 0.55, 0.85])
    p(f"For in-hospital mortality the GNN model reaches an AUROC of {f3(gm['auroc'])} and an AUPRC of "
      f"{f3(gm['auprc'])}, against a prevalence of {gm['prevalence']:.1%}; the No-GNN ablation is slightly lower "
      f"({f3(nm['auroc'])} and {f3(nm['auprc'])}). For sepsis the GNN model reaches an AUROC of {f3(gs['auroc'])}, "
      f"so it ranks patients who later receive a sepsis diagnosis above most others, but its AUPRC is only "
      f"{f3(gs['auprc'])} against a prevalence of {gs['prevalence']:.2%}. At the default threshold of 0.5 neither "
      "model predicts any positive sepsis case: the largest predicted sepsis probability on the test set is about "
      "0.13 for the GNN model and below 0.05 for the No-GNN model, so recall, precision and F1 are zero and the "
      "specificity of 1.000 is trivial. The class weight raised the predicted probabilities compared with the "
      "first run but not enough to cross 0.5.")
    fig("fig_test_curves.png", "ROC and precision-recall curves on the test hospitals (values in the legends: AUROC or AUPRC)")
    h2("8.4 Validation-chosen thresholds")
    p("Because 0.5 is not a meaningful operating point here, the threshold that maximises F1 on the **validation** "
      "hospitals was also computed; the test hospitals were then scored once at that threshold (Table 8.2). Test "
      "labels were not used to choose any threshold.")
    tab("Test metrics at thresholds chosen on the validation hospitals", ["Model", "Task", "Threshold", "Precision", "Recall", "F1", "Specificity"],
        [["FedAvg + GNN", "Mortality", f3(gv["mort"]["threshold"]), f3(gv["mort"]["test"]["precision"]), f3(gv["mort"]["test"]["recall"]), f3(gv["mort"]["test"]["f1"]), f3(gv["mort"]["test"]["specificity"])],
         ["FedAvg + GNN", "Sepsis", f3(gv["sepsis"]["threshold"]), f3(gv["sepsis"]["test"]["precision"]), f3(gv["sepsis"]["test"]["recall"]), f3(gv["sepsis"]["test"]["f1"]), f3(gv["sepsis"]["test"]["specificity"])],
         ["FedAvg, No-GNN", "Mortality", f3(nv["mort"]["threshold"]), f3(nv["mort"]["test"]["precision"]), f3(nv["mort"]["test"]["recall"]), f3(nv["mort"]["test"]["f1"]), f3(nv["mort"]["test"]["specificity"])],
         ["FedAvg, No-GNN", "Sepsis", f3(nv["sepsis"]["threshold"]), f3(nv["sepsis"]["test"]["precision"]), f3(nv["sepsis"]["test"]["recall"]), f3(nv["sepsis"]["test"]["f1"]), f3(nv["sepsis"]["test"]["specificity"])]],
        [1.3, 1.0, 0.9, 0.85, 0.7, 0.6, 0.9])
    p(f"For mortality, the validation-chosen threshold of {f3(gv['mort']['threshold'])} gives the GNN model a recall of "
      f"{f3(gv['mort']['test']['recall'])} at a precision of {f3(gv['mort']['test']['precision'])}. For sepsis, the "
      f"threshold of {f3(gv['sepsis']['threshold'])} gives a precision of {f3(gv['sepsis']['test']['precision'])}, a "
      f"recall of {f3(gv['sepsis']['test']['recall'])} and an F1 of {f3(gv['sepsis']['test']['f1'])}: roughly one "
      "flagged patient in fourteen later received a sepsis diagnosis, and about one in thirteen of those patients "
      "was flagged. Detection of this rare proxy label is therefore weak, even though the ranking (AUROC) is "
      "moderate.")
    h2("8.5 Effect of the patient graph")
    fig("fig_metric_bars.png", "AUROC and AUPRC of the GNN model and the No-GNN ablation on the test hospitals", 5.2)
    p(f"The GraphSAGE model is better than the No-GNN ablation on all four ranking metrics (Figure 8.2): mortality "
      f"AUROC {f3(gm['auroc'])} vs {f3(nm['auroc'])}, mortality AUPRC {f3(gm['auprc'])} vs {f3(nm['auprc'])}, sepsis "
      f"AUROC {f3(gs['auroc'])} vs {f3(ns['auroc'])} and sepsis AUPRC {f3(gs['auprc'])} vs {f3(ns['auprc'])}. The "
      "differences are small and come from a single seed without a significance test, so they indicate a possible "
      "benefit of the graph rather than prove one. The training curves (Figure 6.3) also show that the validation "
      "sepsis AUROC peaked around round 18 and then declined while the combined validation loss kept improving; the "
      "checkpoint, chosen by the combined loss, which is dominated by the more frequent mortality outcome, has a "
      f"validation sepsis AUROC of {G['training']['best_val_metrics']['icu_auroc']:.3f}. Task-specific checkpoint "
      "selection is a natural next step.")
    h2("8.6 Comparison with the earlier mortality + ICU-mortality model")
    p("The earlier model predicts a different second target (ICU mortality), uses all 67 tabular features including "
      "the diagnosis and admission-drug features, and is evaluated on the full 24-hour test cohort of 18,014 stays. "
      "Its results (Table 8.3) are therefore not directly comparable with those of the sepsis model; they are listed "
      "for context only. Its seed-42 run reproduced the recorded corrected experiment exactly, and the recorded "
      f"three-seed results give an in-hospital mortality AUROC of {hs['fedavg_gnn']['mort_auroc']['mean']:.4f} ± "
      f"{hs['fedavg_gnn']['mort_auroc']['sd']:.4f} for FedAvg + GNN and {hs['local_183']['mort_auroc']['mean']:.4f} ± "
      f"{hs['local_183']['mort_auroc']['sd']:.4f} for the best local-only baseline (hospital 183), which shows the "
      "benefit of federation over training on a single hospital.")
    tab("Earlier mortality + ICU-mortality model, seed 42 (different target, features and cohort)",
        ["Model", "Task", "AUROC", "AUPRC", "Brier", "Test stays"],
        [["FedAvg + GraphSAGE", "In-hospital mortality", f3(fgm["auroc"]), f3(fgm["auprc"]), f3(fgm["brier"]), f"{FG['evaluation']['n_patients']:,}"],
         ["FedAvg + GraphSAGE", "ICU mortality", f3(fgi["auroc"]), f3(fgi["auprc"]), f3(fgi["brier"]), f"{FG['evaluation']['n_patients']:,}"],
         ["FedAvg, No-GNN", "In-hospital mortality", f3(FNG["evaluation"]["metrics"]["mort"]["auroc"]), f3(FNG["evaluation"]["metrics"]["mort"]["auprc"]), f3(FNG["evaluation"]["metrics"]["mort"]["brier"]), f"{FNG['evaluation']['n_patients']:,}"],
         ["FedAvg, No-GNN", "ICU mortality", f3(FNG["evaluation"]["metrics"]["icu"]["auroc"]), f3(FNG["evaluation"]["metrics"]["icu"]["auprc"]), f3(FNG["evaluation"]["metrics"]["icu"]["brier"]), f"{FNG['evaluation']['n_patients']:,}"]],
        [1.6, 1.6, 0.7, 0.7, 0.7, 0.9])
    p("The mortality AUROC of the sepsis model (0.804) is somewhat lower than that of the earlier model (0.813). The "
      "two models differ in cohort and in inputs, so the gap cannot be attributed to one cause, but removing the "
      "diagnosis features, which contain post-24-hour information, is expected to cost some apparent performance; "
      "the sepsis model's number is the more conservative estimate.")
    h2("8.7 Comparison with the original specification")
    tab("Implementation status against the Overview specification",
        ["Specified component", "Status", "What was actually done"],
        [["Data preparation (cleaning, normalisation, hospital partitions)", "Implemented and tested", "Prepared files from an earlier stage; loading, cohort, alignment, validation and hospital split implemented"],
         ["Tabular encoder (MLP / TabNet)", "Implemented and tested", "Three-layer MLP"],
         ["Time-series encoder (LSTM / GRU / Transformer)", "Implemented and tested", "Two-layer LSTM over six 4-hour bins"],
         ["Text encoder (MiniBERT / ClinicalBERT)", "Not implemented", "No free-text notes in the prepared data"],
         ["Fusion (concatenation / attention / gated)", "Implemented and tested", "Concatenation followed by a two-layer fusion MLP (two modalities)"],
         ["Intra-hospital patient graph", "Partially implemented", "k-NN graph inside each batch of one hospital, not a full hospital graph"],
         ["GNN (GCN / GraphSAGE / GAT)", "Implemented and tested", "GraphSAGE evaluated; GCN and GAT coded but not evaluated"],
         ["Mortality prediction", "Implemented and tested", "hospital_expired head"],
         ["Sepsis prediction", "Partially implemented", "Documentation-time proxy label; weak detection at default threshold"],
         ["FedAvg", "Implemented and tested", "Simulated in one process over 145 training hospitals"],
         ["Local-only baseline", "Partially implemented", "Evaluated for the mortality + ICU-mortality experiment only"],
         ["Centralised baseline", "Not implemented", "No pooled all-data model was trained"],
         ["Metrics (AUROC, AUPRC, F1, accuracy)", "Partially implemented", "AUROC, AUPRC, F1, precision, recall, specificity, Brier; accuracy not reported"],
         ["Privacy (data stay local)", "Partially implemented", "Simulated separation; no secure aggregation or differential privacy"],
         ["Frontend prediction workflow", "Implemented and tested", "Interactive prediction with the trained checkpoint; Docker deployment"]],
        [2.1, 1.55, 2.62])
    h2("8.8 Limitations")
    bl(["**No text modality.** The model fuses two modalities. A text encoder would need temporally valid clinical "
        "notes, a separate encoder and retraining.",
        "**Proxy sepsis label.** The label uses the documentation time of a sepsis diagnosis, not a validated onset "
        "such as Sepsis-3; documentation can lag the clinical onset and some cases are never documented.",
        "**Weak sepsis detection.** At 0.5 no sepsis case is predicted; at the validation-chosen threshold precision "
        "and recall are below 0.08.",
        "**Simulated federation.** All clients run in one process on one computer; there is no network communication, "
        "secure aggregation or differential privacy, so no privacy guarantee is claimed.",
        "**Batch-level graph.** Edges are built within batches of 32 patients, so a patient's neighbours depend on "
        "the batch composition.",
        "**No centralised baseline and a single seed** for the sepsis experiment; no significance testing.",
        "**Preprocessing provenance.** The scripts that produced the per-hospital feature files are not in the "
        "repository, and tabular features are normalised within each hospital, including test hospitals (label-free).",
        "**No clinical validation.** The system is a research prototype and must not be used for clinical decisions."])

    # ── Chapter 9 ───────────────────────────────────────────────────────────
    h1(9, "CONCLUSION AND FUTURE SCOPE")
    h2("9.1 Conclusion")
    p("This project implemented and evaluated a federated, two-modality graph learning framework on the eICU "
      "database. A tabular MLP and a time-series LSTM are combined by a learned fusion module, a k-NN patient graph is "
      "built inside each hospital batch, GraphSAGE refines the fused representations, and FedAvg trains the model "
      "across 145 simulated hospital clients with validation-based checkpoint selection. A leakage-controlled "
      "sepsis proxy label was constructed, and the final mortality + sepsis model was trained for all 50 rounds and "
      f"evaluated once on {n_test:,} stays from 31 unseen hospitals. It reached an in-hospital mortality AUROC of "
      f"{f3(gm['auroc'])} and a sepsis AUROC of {f3(gs['auroc'])}, both slightly above the No-GNN ablation, but its "
      "detection of the rare sepsis label is weak and it produces no positive sepsis predictions at the default "
      "threshold. The project also delivers a reproducible pipeline with data validation, manifests and tests, and "
      "a containerised demonstration in which hypothetical inputs are processed by the trained checkpoint.")
    p("Measured against the original specification, the core architecture (two encoders, fusion, intra-hospital "
      "graph, GNN, federated averaging and mortality prediction) is implemented; the clinical-text modality and the "
      "centralised baseline are missing, sepsis prediction rests on a proxy label, and the federation is simulated "
      "without formal privacy protection.")
    h2("9.2 Future scope")
    bl(["Add a clinical-text modality with a ClinicalBERT-style encoder once temporally valid notes are available, "
        "and extend the fusion to three modalities.",
        "Build a Sepsis-3-based onset label from cultures, antibiotics and serial organ-function scores.",
        "Train a centralised all-data baseline and repeat all experiments over several seeds with significance tests.",
        "Select checkpoints and thresholds per task, and calibrate the predicted probabilities.",
        "Evaluate a persistent hospital-level graph and other GNN types (GAT, GCN).",
        "Deploy the federation across separate machines and add secure aggregation and differential privacy."])
    return B
