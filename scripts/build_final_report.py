"""
build_final_report.py
---------------------
Builds BTech_Minor_Project_Report_Final.docx following the IIIT Bhopal 2026 project-report format
(A4, Times New Roman, main headings 16 pt, sub-headings 14 pt, text 12 pt, captions 11 pt,
references 10 pt, line spacing 1.15, justified; figure captions below, table captions above;
roman page numbers for the front matter, arabic from Chapter 1).

All numbers are read from the saved manifests / label summary at build time where possible.
Two passes: pass 1 renders with placeholder page numbers, the exported PDF is searched for headings
and captions (--locate), pass 2 fills the Table of Content / List of Figures / List of Tables.

Usage
-----
    python scripts/build_final_report.py --out report.docx [--pages pages.json]
    python scripts/build_final_report.py --locate report.pdf --pages pages.json   # needs PyMuPDF
"""

import argparse
import json
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from final_report_content import content  # noqa: E402

FIG = ROOT / "report_assets" / "final_report"
EQ = FIG / "equations"
SEP = json.loads((ROOT / "results" / "minor_project_sepsis" / "final_model_manifest.json").read_text())
FIN = json.loads((ROOT / "results" / "minor_project_final" / "final_model_manifest.json").read_text())
LAB = json.loads((ROOT / "results" / "minor_project_sepsis" / "label_summary.json").read_text())
HIST = json.loads((ROOT / "results" / "corrected" / "benchmark_results.json").read_text())
G, NG = SEP["models"]["fedavg_gnn"], SEP["models"]["fedavg_nognn"]
FG, FNG = FIN["models"]["fedavg_gnn"], FIN["models"]["fedavg_nognn"]
LC = SEP["label_counts_24h_cohort"]
SEPSIS_COHORT = {k: LC[k]["positive"] + LC[k]["negative"] for k in ("train", "val", "test")}
assert SEPSIS_COHORT["test"] == G["evaluation"]["n_patients"] == SEP["evaluated_cohort_patients"]["test"]
TITLE = "Federated Multimodal Graph Learning Framework for Mortality and Sepsis Prediction using eICU"
LOGO = FIG / "iiitb_logo.jpeg"
STUDENTS = [("Dhruv Savaliya", "24U020052"), ("Menil Patel", "24U020048")]
SUPERVISOR = ("Supriyo Das", "Minor Project Mentor", "Department of Computer Science and Engineering")
TEXT_W = 6.27  # inches between 1-inch margins on A4


def f3(v):
    return f"{v:.3f}"


def f4(v):
    return f"{v:.4f}"


# ── references (IEEE, numbered by first citation) ───────────────────────────
REFS = {
    "zimmerman": 'J. E. Zimmerman, A. A. Kramer, D. S. McNair, and F. M. Malila, "Acute Physiology and Chronic Health Evaluation (APACHE) IV: Hospital mortality assessment for today\'s critically ill patients," Critical Care Medicine, vol. 34, no. 5, pp. 1297–1310, 2006.',
    "sofa": 'J.-L. Vincent et al., "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure," Intensive Care Medicine, vol. 22, no. 7, pp. 707–710, 1996.',
    "sepsis3": 'M. Singer et al., "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)," JAMA, vol. 315, no. 8, pp. 801–810, 2016, doi: 10.1001/jama.2016.0287.',
    "eicu": 'T. J. Pollard, A. E. W. Johnson, J. D. Raffa, L. A. Celi, R. G. Mark, and O. Badawi, "The eICU Collaborative Research Database, a freely available multi-center database for critical care research," Scientific Data, vol. 5, Art. no. 180178, 2018, doi: 10.1038/sdata.2018.178.',
    "physionet": 'A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," Circulation, vol. 101, no. 23, pp. e215–e220, 2000.',
    "mcmahan": 'H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in Proc. 20th Int. Conf. Artificial Intelligence and Statistics (AISTATS), PMLR vol. 54, 2017, pp. 1273–1282.',
    "lstm": 'S. Hochreiter and J. Schmidhuber, "Long short-term memory," Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.',
    "harut": 'H. Harutyunyan, H. Khachatrian, D. C. Kale, G. Ver Steeg, and A. Galstyan, "Multitask learning and benchmarking with clinical time series data," Scientific Data, vol. 6, Art. no. 96, 2019, doi: 10.1038/s41597-019-0103-9.',
    "rajkomar": 'A. Rajkomar et al., "Scalable and accurate deep learning with electronic health records," npj Digital Medicine, vol. 1, Art. no. 18, 2018.',
    "nemati": 'S. Nemati, A. Holder, F. Razmi, M. D. Stanley, G. D. Clifford, and T. G. Buchman, "An interpretable machine learning model for accurate prediction of sepsis in the ICU," Critical Care Medicine, vol. 46, no. 4, pp. 547–553, 2018.',
    "reyna": 'M. A. Reyna et al., "Early prediction of sepsis from clinical data: The PhysioNet/Computing in Cardiology Challenge 2019," Critical Care Medicine, vol. 48, no. 2, pp. 210–217, 2020, doi: 10.1097/CCM.0000000000004145.',
    "baltrusaitis": 'T. Baltrušaitis, C. Ahuja, and L.-P. Morency, "Multimodal machine learning: A survey and taxonomy," IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423–443, 2019.',
    "bert": 'J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. NAACL-HLT, 2019, pp. 4171–4186.',
    "alsentzer": 'E. Alsentzer et al., "Publicly available clinical BERT embeddings," in Proc. 2nd Clinical Natural Language Processing Workshop, 2019, pp. 72–78, doi: 10.18653/v1/W19-1909.',
    "gcn": 'T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in Proc. Int. Conf. Learning Representations (ICLR), 2017.',
    "hamilton": 'W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in Advances in Neural Information Processing Systems 30 (NeurIPS), 2017, pp. 1024–1034.',
    "gat": 'P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "Graph attention networks," in Proc. Int. Conf. Learning Representations (ICLR), 2018.',
    "rocheteau": 'E. Rocheteau, C. Tong, P. Veličković, N. Lane, and P. Liò, "Predicting patient outcomes with graph representation learning," arXiv:2101.03940, 2021.',
    "li": 'T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, "Federated learning: Challenges, methods, and future directions," IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 50–60, 2020.',
    "kairouz": 'P. Kairouz et al., "Advances and open problems in federated learning," Foundations and Trends in Machine Learning, vol. 14, no. 1–2, pp. 1–210, 2021.',
    "rieke": 'N. Rieke et al., "The future of digital health with federated learning," npj Digital Medicine, vol. 3, Art. no. 119, 2020.',
    "sheller": 'M. J. Sheller et al., "Federated learning in medicine: Facilitating multi-institutional collaborations without sharing patient data," Scientific Reports, vol. 10, Art. no. 12598, 2020, doi: 10.1038/s41598-020-69250-1.',
    "dayan": 'I. Dayan et al., "Federated learning for predicting clinical outcomes in patients with COVID-19," Nature Medicine, vol. 27, pp. 1735–1743, 2021.',
    "bonawitz": 'K. Bonawitz et al., "Practical secure aggregation for privacy-preserving machine learning," in Proc. ACM SIGSAC Conf. Computer and Communications Security (CCS), 2017, pp. 1175–1191.',
    "abadi": 'M. Abadi et al., "Deep learning with differential privacy," in Proc. ACM SIGSAC Conf. Computer and Communications Security (CCS), 2016, pp. 308–318.',
    "saito": 'T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," PLoS ONE, vol. 10, no. 3, Art. no. e0118432, 2015.',
    "brier": 'G. W. Brier, "Verification of forecasts expressed in terms of probability," Monthly Weather Review, vol. 78, no. 1, pp. 1–3, 1950.',
    "adam": 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. Int. Conf. Learning Representations (ICLR), 2015.',
    "pytorch": 'A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in Advances in Neural Information Processing Systems 32 (NeurIPS), 2019, pp. 8024–8035.',
    "pyg": 'M. Fey and J. E. Lenssen, "Fast graph representation learning with PyTorch Geometric," in Proc. ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.',
    "sklearn": 'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.',
}

# ── equations (matplotlib mathtext) ──────────────────────────────────────────
EQUATIONS = {
    "eq_mlp": r"$z_{tab}=W_3\,\phi(W_2\,\phi(W_1x_i+b_1)+b_2)+b_3,\qquad \phi(\cdot)=\mathrm{Dropout}(\mathrm{ReLU}(\mathrm{BN}(\cdot)))$",
    "eq_lstm": r"$(h_t^{(l)},c_t^{(l)})=\mathrm{LSTM}^{(l)}(h_t^{(l-1)},h_{t-1}^{(l)},c_{t-1}^{(l)}),\quad h_t^{(0)}=s_t,\ t=1,\ldots,T_i\leq 6;\qquad z_{ts}=W_oh_{T_i}^{(2)}+b_o$",
    "eq_fusion": r"$u_i=[z_{tab}\,\|\,z_{ts}]\in\mathbb{R}^{128},\qquad h_i=\mathrm{ReLU}(\mathrm{BN}(W_{f2}\,\phi(W_{f1}u_i+b_{f1})+b_{f2}))\in\mathbb{R}^{128}$",
    "eq_knn": r"$\mathrm{sim}(i,j)=\dfrac{h_i\cdot h_j}{\|h_i\|\,\|h_j\|},\qquad \mathcal{N}(i)=\underset{j\neq i,\ |\mathcal{N}(i)|=k}{\mathrm{arg\,top}}\ \mathrm{sim}(i,j),\quad k=5$",
    "eq_sage": r"$g_i^{(l+1)}=\mathrm{ReLU}\left(\mathrm{BN}\left(W_s^{(l)}g_i^{(l)}+W_n^{(l)}\dfrac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)}g_j^{(l)}\right)\right)+R^{(l)}g_i^{(l)},\quad g_i^{(0)}=h_i$",
    "eq_loss": r"$\mathcal{L}=\sum_{t\in\{\mathrm{mort},\,\mathrm{sep}\}}-\dfrac{1}{N}\sum_{i=1}^{N}\left[w_t\,y_{i,t}\log\sigma(\ell_{i,t})+(1-y_{i,t})\log(1-\sigma(\ell_{i,t}))\right]$",
    "eq_posweight": r"$w_{\mathrm{sep}}=\dfrac{N^{-}_{\mathrm{train}}}{N^{+}_{\mathrm{train}}}=\dfrac{72{,}457}{1{,}453}\approx 49.87,\qquad w_{\mathrm{mort}}=3.0$",
    "eq_fedavg": r"$w^{(t)}=\sum_{k\in S_t}\dfrac{n_k}{n}\,w_k^{(t)},\qquad n=\sum_{k\in S_t}n_k,\quad |S_t|=10$",
    "eq_metrics": r"$\mathrm{Precision}=\dfrac{TP}{TP+FP},\quad \mathrm{Recall}=\dfrac{TP}{TP+FN},\quad \mathrm{Specificity}=\dfrac{TN}{TN+FP},\quad F_1=\dfrac{2\,\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}$",
    "eq_brier": r"$\mathrm{Brier}=\dfrac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2$",
}


def render_equations():
    EQ.mkdir(parents=True, exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "stix"
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=12)
        fig.savefig(EQ / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=False,
                    facecolor="white")
        plt.close(fig)


# ── low-level docx helpers ───────────────────────────────────────────────────
def set_cell_shading(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def set_table_borders(table, on=True):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if on else "nil")
        if on:
            el.set(qn("w:sz"), "4")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)


def set_col_widths(table, widths):
    table.autofit = False
    tblPr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    grid = table._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w * 1440)))
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Inches(w)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def add_field(run, instr):
    for kind, text in (("begin", None), ("instr", instr), ("separate", None), ("text", "1"), ("end", None)):
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        elif kind == "text":
            el = OxmlElement("w:t")
            el.text = text
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        run._r.append(el)


def page_numbering(section, fmt, start):
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))
    section._sectPr.append(pg)


def set_margins(section, top, right, bottom, left):
    section.page_width, section.page_height = Twips(11906), Twips(16838)
    section.top_margin, section.right_margin = Twips(top), Twips(right)
    section.bottom_margin, section.left_margin = Twips(bottom), Twips(left)
    section.header_distance, section.footer_distance = Twips(708), Twips(560)


def strip_theme_fonts(style):
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rFonts.get(qn(a)) is not None:
            del rFonts.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(a), "Times New Roman")


# ── report builder ───────────────────────────────────────────────────────────
class Report:
    def __init__(self, pages):
        self.pages = pages or {}
        self.doc = Document()
        self.cite = {}
        self.figs, self.tabs, self.toc = [], [], []
        self.chapter = 0
        self.nfig = self.ntab = 0
        self._styles()

    # styles
    def _styles(self):
        st = self.doc.styles
        normal = st["Normal"]
        normal.font.name, normal.font.size = "Times New Roman", Pt(12)
        strip_theme_fonts(normal)
        pf = normal.paragraph_format
        pf.line_spacing, pf.space_after, pf.space_before = 1.15, Pt(6), Pt(0)
        for name, size, before, after in (("Heading 1", 16, 0, 18), ("Heading 2", 14, 12, 6), ("Heading 3", 12, 10, 4)):
            s = st[name]
            s.font.name, s.font.size, s.font.bold, s.font.italic = "Times New Roman", Pt(size), True, False
            s.font.color.rgb = RGBColor(0, 0, 0)
            strip_theme_fonts(s)
            s.paragraph_format.space_before, s.paragraph_format.space_after = Pt(before), Pt(after)
            s.paragraph_format.keep_with_next = True
            s.paragraph_format.line_spacing = 1.15
        st["Heading 1"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        lb = st["List Bullet"]
        lb.font.name, lb.font.size = "Times New Roman", Pt(12)
        lb.paragraph_format.space_after, lb.paragraph_format.line_spacing = Pt(3), 1.15

    # citations: number by first appearance over the whole content
    def number_citations(self, blocks, extra_texts=()):
        def scan(t):
            for key in re.findall(r"\[@(\w+)\]", t):
                if key not in REFS:
                    raise KeyError(f"unknown reference {key}")
                self.cite.setdefault(key, len(self.cite) + 1)

        for t in extra_texts:
            scan(t)
        for b in blocks:
            for part in b[1:]:
                if isinstance(part, str):
                    scan(part)
                elif isinstance(part, list):
                    for item in part:
                        if isinstance(item, list):
                            for c in item:
                                scan(str(c))
                        else:
                            scan(str(item))
        missing = set(REFS) - set(self.cite)
        if missing:
            raise ValueError(f"references never cited: {sorted(missing)}")

    def cites(self, text):
        return re.sub(r"\[@(\w+)\]", lambda m: f"[{self.cite[m.group(1)]}]", text)

    # inline markup: **bold**, *italic with _sub ^sup*, `code`
    def runs(self, par, text, size=None, bold=False):
        text = self.cites(text)
        for tok in re.split(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*?\*)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = par.add_run(tok[2:-2])
                r.bold = True
            elif tok.startswith("`"):
                r = par.add_run(tok[1:-1])
                r.font.name = "Courier New"
                r.font.size = Pt((size or 12) - 1)
                r.bold = bold
                continue
            elif tok.startswith("*"):
                m = re.fullmatch(r"([^_^]+)(?:_([^\^]+))?(?:\^(.+))?", tok[1:-1])
                base, sub, sup = m.groups()
                r = par.add_run(base)
                r.italic = True
                for part, attr in ((sub, "subscript"), (sup, "superscript")):
                    if part:
                        rr = par.add_run(part)
                        rr.italic = True
                        setattr(rr.font, attr, True)
                        if size:
                            rr.font.size = Pt(size)
                if size:
                    r.font.size = Pt(size)
                continue
            else:
                r = par.add_run(tok)
                r.bold = bold
            if size:
                r.font.size = Pt(size)

    def para(self, text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, bold=False, after=None, before=None,
             keep=False, style=None):
        p = self.doc.add_paragraph(style=style)
        p.alignment = align
        if text:
            self.runs(p, text, size=size, bold=bold)
        if after is not None:
            p.paragraph_format.space_after = Pt(after)
        if before is not None:
            p.paragraph_format.space_before = Pt(before)
        if keep:
            p.paragraph_format.keep_with_next = True
        return p

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def front_heading(self, text, before=0):
        p = self.para(text, align=WD_ALIGN_PARAGRAPH.CENTER, size=16, bold=True, after=18, before=before)
        p.paragraph_format.keep_with_next = True
        return p

    def logo(self, width):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(LOGO), width=Inches(width))
        return p

    def pg(self, key):
        return self.pages.get(key, "00")

    # ── front matter ──
    def title_page(self):
        sec = self.doc.sections[0]
        set_margins(sec, 1418, 1418, 1418, 1701)
        C = WD_ALIGN_PARAGRAPH.CENTER
        self.para(TITLE.upper().replace("EICU", "eICU"), align=C, size=18, bold=True, before=24, after=18)
        self.para("B.TECH MINOR PROJECT REPORT", align=C, size=13, bold=True, after=4)
        self.para("Submitted in partial fulfillment for the award of the degree of", align=C, after=4)
        self.para("BACHELOR OF TECHNOLOGY", align=C, size=16, bold=True, after=0)
        self.para("Department of Computer Science and Engineering", align=C, size=13, bold=True, after=14)
        self.para("Submitted to", align=C, after=2)
        self.para("INDIAN INSTITUTE OF INFORMATION TECHNOLOGY", align=C, size=16, bold=True, after=0)
        self.para("BHOPAL (M.P.)", align=C, size=16, bold=True, after=8)
        self.logo(1.75)
        self.para("Submitted by", align=C, size=14, bold=True, before=8, after=4)
        for name, sch in STUDENTS:
            self.para(f"{name} ({sch}, Semester V)", align=C, after=0)
        self.para("B.Tech CSE, Pre-Final Year", align=C, after=0)
        self.para("Under the supervision of", align=C, size=14, bold=True, before=18, after=4)
        self.para(SUPERVISOR[0], align=C, after=0)
        self.para(SUPERVISOR[1], align=C, after=0)
        self.para(f"({SUPERVISOR[2]})", align=C, after=0)
        self.para("November 2026", align=C, bold=True, before=18, after=0)

    def new_section(self, fmt, start, margins=(1440, 1440, 1440, 1440)):
        sec = self.doc.add_section(WD_SECTION.NEW_PAGE)
        set_margins(sec, *margins)
        sec.footer.is_linked_to_previous = False
        fp = sec.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in list(fp.runs):
            r._r.getparent().remove(r._r)
        r = fp.add_run()
        r.font.size = Pt(11)
        add_field(r, "PAGE")
        page_numbering(sec, fmt, start)
        return sec

    def certificate(self):
        C = WD_ALIGN_PARAGRAPH.CENTER
        self.para("INDIAN INSTITUTE OF INFORMATION TECHNOLOGY", align=C, size=16, bold=True, before=12, after=0)
        self.para("BHOPAL (M.P.)", align=C, size=16, bold=True, after=12)
        self.logo(1.25)
        self.front_heading("CERTIFICATE", before=12)
        names = " and ".join(f"**{n.upper()} ({s})**" for n, s in STUDENTS)
        self.para(f"This is to certify that the work embodied in this report entitled **“{TITLE}”** has been "
                  f"satisfactorily completed by {names}. It is a bonafide piece of work, carried out under our guidance "
                  "in the **Department of Computer Science and Engineering**, **Indian Institute of Information "
                  "Technology, Bhopal** for the partial fulfillment of the Bachelor of Technology during the academic "
                  "year 2026–2027.", after=36, before=12)
        self.para("Date:", align=WD_ALIGN_PARAGRAPH.LEFT, after=60)
        t = self.doc.add_table(rows=1, cols=2)
        set_table_borders(t, on=False)
        set_col_widths(t, [3.2, 3.07])
        left = [("Supriyo Das,", True), ("Minor Project Mentor,", False),
                ("Department of Computer Science and Engineering,", False), ("IIIT Bhopal (M.P.)", False)]
        right = [("Head/Department Coordinator,", True), ("Department of CSE,", False), ("IIIT Bhopal (M.P.)", False)]
        for cell, lines in zip(t.rows[0].cells, (left, right)):
            cell.paragraphs[0].text = ""
            for i, (txt, b) in enumerate(lines):
                p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                r = p.add_run(txt)
                r.bold = b
                r.font.size = Pt(11)
            sig = cell.add_paragraph()
            sig.paragraph_format.space_before = Pt(18)
            r = sig.add_run("Signature: ____________________")
            r.font.size = Pt(11)

    def declaration(self):
        C = WD_ALIGN_PARAGRAPH.CENTER
        self.page_break()
        self.para("INDIAN INSTITUTE OF INFORMATION TECHNOLOGY", align=C, size=16, bold=True, before=12, after=0)
        self.para("BHOPAL (M.P.)", align=C, size=16, bold=True, after=12)
        self.logo(1.25)
        self.front_heading("DECLARATION", before=12)
        self.para("DRAFT — SUPERVISOR/DEPARTMENT APPROVAL REQUIRED", align=C, size=11, bold=True, after=12)
        self.para(f"We hereby declare that the minor project report entitled “{TITLE}” is presented in partial "
                  "fulfillment of the requirements for the award of the degree of Bachelor of Technology in Computer "
                  "Science and Engineering. It documents the project work carried out by us under the guidance of "
                  "Supriyo Das, Minor Project Mentor, Department of Computer Science and Engineering, IIIT Bhopal. The "
                  "project work presented has not been submitted in part or whole for the award of any degree or "
                  "professional diploma in any other institute or organization.")
        self.para("**Disclosure of assistance.** Parts of the project software and of the text of this report were "
                  "prepared with the help of an AI coding and writing assistant. All reported results were produced "
                  "by the project code on the locally stored eICU data and were checked against the saved result "
                  "files. The students must read and confirm the complete report before signing this declaration.")
        self.para("We, with this, declare that the facts mentioned above are true to the best of our knowledge. In case "
                  "of any unlikely discrepancy that may occur, we will be the ones to take responsibility.", after=36)
        t = self.doc.add_table(rows=len(STUDENTS), cols=2)
        set_table_borders(t, on=False)
        set_col_widths(t, [3.6, 2.67])
        for row, (n, s) in zip(t.rows, STUDENTS):
            row.cells[0].paragraphs[0].add_run(f"{n} ({s})")
            row.cells[1].paragraphs[0].add_run("Signature: ________________")
            row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            for c in row.cells:
                c.paragraphs[0].paragraph_format.space_after = Pt(24)

    def acknowledgement(self):
        self.page_break()
        self.front_heading("ACKNOWLEDGEMENT", before=24)
        self.para(f"We, Dhruv Savaliya and Menil Patel, would like to express our sincere gratitude to everyone who "
                  f"supported and guided us throughout our minor project titled “{TITLE}”.")
        self.para("First, we thank our Director, [Director’s Name], for providing the resources and facilities "
                  "needed to carry out this project.")
        self.para("We are grateful to our Project Mentor, Supriyo Das, for the guidance and advice given throughout the "
                  "project.")
        self.para("Our thanks also go to the Department Coordinator, [Coordinator’s Name], for the support provided "
                  "during the completion of the project, and to the faculty of the Department of Computer Science and "
                  "Engineering for their guidance.")
        self.para("We acknowledge the creators of the eICU Collaborative Research Database and the PhysioNet team for "
                  "making the data available for research under a data-use agreement, and the developers of the "
                  "open-source software used in this work.", after=30)
        for n, _ in STUDENTS:
            self.para(n, align=WD_ALIGN_PARAGRAPH.RIGHT, after=0)

    def area_of_work(self):
        self.page_break()
        self.front_heading("AREA OF WORK", before=24)
        self.para("Our project mainly focuses on **MACHINE LEARNING FOR HEALTHCARE**, in particular **federated "
                  "learning**, **graph neural networks** and **multimodal deep learning** on electronic health records "
                  "from intensive care units.")
        self.para("The work combines three areas. From deep learning on clinical data it uses a multilayer perceptron "
                  "for static patient information and a recurrent LSTM network for measurements taken over the first "
                  "24 hours of an ICU stay, joined by a learned fusion layer. From graph representation learning it "
                  "uses a patient-similarity graph built inside each hospital and a GraphSAGE network that lets similar "
                  "patients share information. From distributed machine learning it uses federated averaging, in which "
                  "hospitals train a shared model without pooling their records.")
        self.para("The application area is clinical risk prediction: estimating the risk of in-hospital death and of "
                  "sepsis documented later in the stay, using the multi-centre eICU Collaborative Research Database. "
                  "The project also covers the software engineering needed to make such a model reproducible and "
                  "demonstrable: data validation, experiment manifests, automated tests, an inference API, a React "
                  "web interface and a containerised deployment with Docker.")

    def listing_table(self, header, rows, widths):
        t = self.doc.add_table(rows=1, cols=3)
        set_table_borders(t, on=True)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for c, h in zip(t.rows[0].cells, header):
            c.paragraphs[0].text = ""
            r = c.paragraphs[0].add_run(h)
            r.bold = True
            r.font.size = Pt(12)
            set_cell_shading(c, "D9D9D9")
        repeat_header(t.rows[0])
        for sno, title, page, bold in rows:
            cells = t.add_row().cells
            for i, txt in enumerate((sno, title, page)):
                p = cells[i].paragraphs[0]
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                r = p.add_run(txt)
                r.font.size = Pt(11)
                r.bold = bold
                if i != 1:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_col_widths(t, widths)
        return t

    def toc_pages(self, blocks):
        self.page_break()
        self.front_heading("TABLE OF CONTENT", before=12)
        rows = [("", "Certificate", self.pg("front:CERTIFICATE"), False),
                ("", "Declaration", self.pg("front:DECLARATION"), False),
                ("", "Acknowledgement", self.pg("front:ACKNOWLEDGEMENT"), False),
                ("", "Area of Work", self.pg("front:AREA OF WORK"), False),
                ("", "List of Figures", self.pg("front:LIST OF FIGURES"), False),
                ("", "List of Tables", self.pg("front:LIST OF TABLES"), False),
                ("", "Abstract", self.pg("front:ABSTRACT"), False)]
        for b in blocks:
            if b[0] == "h1":
                rows.append((str(b[1]), b[2].title().replace("And", "and").replace("Dfd", "DFD"),
                             self.pg(f"h1:{b[1]}"), True))
            elif b[0] == "h2":
                num = b[1].split()[0]
                rows.append(("", "    " + b[1], self.pg(f"h2:{num}"), False))
        for n, (key, label) in enumerate(END_SECTIONS, start=10):
            rows.append((str(n), label, self.pg(f"end:{key}"), True))
        self.listing_table(["S.no", "Title", "Page No."], rows, [0.7, 4.77, 0.8])

    def lof_lot(self, blocks):
        figs, tabs, ch = [], [], 0
        nf = nt = 0
        for b in blocks:
            if b[0] == "h1":
                ch, nf, nt = b[1], 0, 0
            elif b[0] == "fig":
                nf += 1
                figs.append((f"{ch}.{nf}", b[2]))
            elif b[0] == "table":
                nt += 1
                tabs.append((f"{ch}.{nt}", b[1]))
        tabs.append(("A.1", APPENDIX_TABLE))
        self.page_break()
        self.front_heading("LIST OF FIGURES", before=12)
        self.listing_table(["Fig", "Description", "Page no."],
                           [(n, self.cites(c), self.pg(f"fig:{n}"), False) for n, c in figs], [0.7, 4.77, 0.8])
        self.page_break()
        self.front_heading("LIST OF TABLES", before=12)
        self.listing_table(["Table No", "Description", "Page no."],
                           [(n, self.cites(c), self.pg(f"tab:{n}"), False) for n, c in tabs], [0.8, 4.67, 0.8])

    def abstract(self):
        self.page_break()
        self.front_heading("ABSTRACT", before=12)
        gm, gs = G["evaluation"]["metrics"]["mort"], G["evaluation"]["metrics"]["sepsis"]
        nm, ns = NG["evaluation"]["metrics"]["mort"], NG["evaluation"]["metrics"]["sepsis"]
        gv = G["evaluation"]["validation_chosen_threshold"]["sepsis"]
        self.para("Clinical records are spread across hospitals that usually cannot pool them, and intensive-care data "
                  "combine static information with measurements over time. This project builds and evaluates a "
                  "federated, two-modality graph learning framework that predicts in-hospital mortality and sepsis "
                  "from the first 24 hours of an ICU stay in the eICU Collaborative Research Database. Each of 207 "
                  "hospitals is treated as a federated client, and hospitals are split into 145 for training, 31 for "
                  "validation and 31 for testing.")
        self.para("A multilayer perceptron encodes 28 tabular features and a two-layer LSTM encodes 74 time-series "
                  "features over six 4-hour bins. A learned fusion module combines the two 64-dimensional embeddings "
                  "into a 128-dimensional patient vector. A cosine k-nearest-neighbour graph (k = 5) is built inside "
                  "each batch of patients from one hospital, two GraphSAGE layers refine the fused vectors, and two "
                  "heads predict the outcomes. The model is trained with federated averaging (FedAvg) over 50 rounds "
                  "of ten simulated clients, and the checkpoint with the lowest validation loss is kept. The sepsis "
                  "target is a documentation-time proxy: a sepsis or septic-shock diagnosis first recorded after 24 "
                  "hours. Stays with sepsis documented earlier are excluded, and features that contain post-24-hour "
                  "information are removed.")
        self.para(f"On {G['evaluation']['n_patients']:,} stays from the unseen test hospitals, the model reached an "
                  f"AUROC of {f3(gm['auroc'])} and an AUPRC of {f3(gm['auprc'])} for in-hospital mortality, and an "
                  f"AUROC of {f3(gs['auroc'])} and an AUPRC of {f3(gs['auprc'])} for sepsis "
                  f"(prevalence {gs['prevalence']:.2%}). It was slightly better than a No-GNN ablation "
                  f"({f3(nm['auroc'])} and {f3(ns['auroc'])}). Sepsis detection was weak: at the 0.5 threshold no "
                  f"sepsis case was predicted, and at a threshold chosen on the validation hospitals the F1 score was "
                  f"{f3(gv['test']['f1'])}. A clinical-text encoder from the original plan was not implemented "
                  "because the data contain no notes, and the federation is a single-process simulation without "
                  "formal privacy guarantees. The trained model is served through an inference API, a web interface "
                  "for hypothetical inputs and a three-container Docker deployment.")
        self.para("**Keywords:** federated learning, graph neural networks, GraphSAGE, multimodal fusion, LSTM, eICU, "
                  "mortality prediction, sepsis prediction, FedAvg.", before=6)

    # ── body ──
    def h1(self, num, title):
        self.chapter, self.nfig, self.ntab = num, 0, 0
        p = self.doc.add_paragraph(style="Heading 1")
        p.paragraph_format.page_break_before = True
        p.add_run(f"CHAPTER {num}")
        p.add_run().add_break()
        p.add_run(title)

    def h_end(self, title):
        p = self.doc.add_paragraph(style="Heading 1")
        p.paragraph_format.page_break_before = True
        p.add_run(title)

    def figure(self, file, caption, width):
        self.nfig += 1
        from PIL import Image
        with Image.open(FIG / file) as im:
            w, h = im.size
        width = min(width, TEXT_W, 8.0 * w / h)  # keep images at most 8 in tall
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.line_spacing = 1.0
        p.add_run().add_picture(str(FIG / file), width=Inches(width))
        c = self.para(f"Figure {self.chapter}.{self.nfig}: {caption}", align=WD_ALIGN_PARAGRAPH.CENTER, size=11,
                      after=12)
        c.runs[0].bold = False

    def table(self, caption, header, rows, widths):
        self.ntab += 1
        tot = sum(widths)
        if tot > TEXT_W:
            widths = [w * TEXT_W / tot for w in widths]
        cap = self.para(f"Table {self.chapter}.{self.ntab}: {caption}", align=WD_ALIGN_PARAGRAPH.CENTER, size=11,
                        after=4, before=6, keep=True)
        t = self.doc.add_table(rows=1, cols=len(header))
        set_table_borders(t, on=True)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for c, h in zip(t.rows[0].cells, header):
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = True
            self.runs(p, h, size=10, bold=True)
            set_cell_shading(c, "D9D9D9")
        repeat_header(t.rows[0])
        cant_split(t.rows[0])
        for i, row in enumerate(rows):
            r = t.add_row()
            cant_split(r)
            for c, v in zip(r.cells, row):
                p = c.paragraphs[0]
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.0
                if len(rows) <= 12 and i < len(rows) - 1:
                    p.paragraph_format.keep_with_next = True
                self.runs(p, str(v), size=10)
        set_col_widths(t, widths)
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(4)
        sp.paragraph_format.line_spacing = 1.0

    def equation(self, file, number):
        from PIL import Image
        with Image.open(EQ / file) as im:
            w, _ = im.size
        width = min(w / 300, 5.45)
        t = self.doc.add_table(rows=1, cols=2)
        set_table_borders(t, on=False)
        set_col_widths(t, [5.6, 0.67])
        p = t.rows[0].cells[0].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.add_run().add_picture(str(EQ / file), width=Inches(width))
        q = t.rows[0].cells[1].paragraphs[0]
        q.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        q.add_run(f"({number})")
        from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
        for c in t.rows[0].cells:
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(2)
        sp.paragraph_format.line_spacing = 1.0

    def body(self, blocks):
        for b in blocks:
            kind = b[0]
            if kind == "h1":
                self.h1(b[1], b[2])
            elif kind == "h2":
                self.doc.add_paragraph(style="Heading 2").add_run(b[1])
            elif kind == "h3":
                self.doc.add_paragraph(style="Heading 3").add_run(b[1])
            elif kind == "p":
                self.para(b[1])
            elif kind == "bullets":
                for item in b[1]:
                    p = self.doc.add_paragraph(style="List Bullet")
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    self.runs(p, item)
            elif kind == "fig":
                self.figure(b[1], b[2], b[3])
            elif kind == "table":
                self.table(b[1], b[2], b[3], b[4])
            elif kind == "eq":
                self.equation(b[1], b[2])
            else:
                raise ValueError(kind)

    def end_matter(self):
        self.h_end("REFERENCES")
        for key, n in sorted(self.cite.items(), key=lambda kv: kv[1]):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format
            pf.left_indent, pf.first_line_indent = Inches(0.45), Inches(-0.45)
            pf.space_after, pf.line_spacing = Pt(4), 1.15
            r = p.add_run(f"[{n}]\t{REFS[key]}")
            r.font.size = Pt(10)
            pf.tab_stops.add_tab_stop(Inches(0.45))

        self.h_end("LIST OF PUBLICATION ACCEPTED/SUBMITTED/PUBLISHED")
        self.para("Not applicable. No paper based on this project has been accepted, submitted or published at the time "
                  "of writing this report.")

        self.h_end("APPENDIX: REPRODUCING THE RESULTS")
        self.para("The restricted eICU files are not part of the repository; they must be obtained from PhysioNet under "
                  "the data-use agreement and placed in the git-ignored `processed/` folder. With the data in place, "
                  "the results in Chapter 8 are reproduced on Windows with the commands below (on the CPU "
                  "used, each 50-round training run took about six minutes).")
        cmds = [
            (r".\.venv\Scripts\python.exe scripts\validate_data.py", "10 data checks"),
            (r".\.venv\Scripts\python.exe scripts\build_sepsis_labels.py", "builds processed\\sepsis_labels.parquet"),
            (r".\.venv\Scripts\python.exe scripts\final_model.py train --task mortality_sepsis --with-nognn",
             "trains both models, scores the test hospitals once, writes the manifest"),
            (r".\.venv\Scripts\python.exe scripts\final_model.py evaluate --task mortality_sepsis",
             "re-scores the saved checkpoints"),
            (r".\.venv\Scripts\python.exe scripts\final_model.py check --task mortality_sepsis --split val",
             "checks the validation metrics of the selected round"),
            (r".\.venv\Scripts\python.exe -m pytest tests -q", "runs the test suite"),
            (r"powershell -ExecutionPolicy Bypass -File .\run-demo.ps1", "starts the three-container demonstration"),
        ]
        self.chapter, self.ntab = "A", 0
        self.table(APPENDIX_TABLE, ["Command", "Purpose"],
                   [[f"`{c}`", d] for c, d in cmds], [4.2, 2.07])

        self.h_end("PLAG REPORT (THROUGH TURNITIN)")
        self.para("**1. Similarity plagiarism report:** [To be attached after the Turnitin check carried out through the "
                  "department. The format requires a similarity index below 12%.]")
        self.para("**2. AI plagiarism report:** [To be attached after the Turnitin check carried out through the "
                  "department.]")
        self.para("These reports cannot be produced by the authors themselves and are intentionally left as "
                  "placeholders in this version of the report.")

    def build(self, out):
        render_equations()
        blocks = content({"G": G, "NG": NG, "FG": FG, "FNG": FNG, "LC": LC, "SC": SEPSIS_COHORT, "SEP": SEP,
                          "LAB": LAB, "HIST": HIST, "f3": f3, "f4": f4})
        self.number_citations(blocks)
        self.title_page()
        self.new_section("lowerRoman", 1)
        self.certificate()
        self.declaration()
        self.acknowledgement()
        self.area_of_work()
        self.toc_pages(blocks)
        self.lof_lot(blocks)
        self.abstract()
        self.new_section("decimal", 1)
        self.body(blocks)
        self.end_matter()
        # the body section starts with a page break already (new section); drop the duplicate
        first = next(p for p in self.doc.paragraphs if p.text.startswith("CHAPTER 1"))
        first.paragraph_format.page_break_before = False
        props = self.doc.core_properties
        props.title, props.author = TITLE, ", ".join(n for n, _ in STUDENTS)
        props.subject = "B.Tech Minor Project Report, IIIT Bhopal, 2026"
        self.doc.save(out)
        print(f"saved {out}: {len(self.cite)} references")


APPENDIX_TABLE = "Commands used to reproduce the results and the demonstration"
END_SECTIONS = [("REFERENCES", "References"),
                ("LIST OF PUBLICATION ACCEPTED/SUBMITTED/PUBLISHED", "List of Publication Accepted/Submitted/Published"),
                ("APPENDIX: REPRODUCING THE RESULTS", "Appendix: Reproducing the Results"),
                ("PLAG REPORT (THROUGH TURNITIN)", "Plag Report (through Turnitin)")]


def locate(pdf, out_json):
    """Find the printed page label of every heading / caption in the exported PDF."""
    import pymupdf

    doc = pymupdf.open(pdf)
    found = {}
    for page in doc:
        d = page.get_text("dict")
        lines = []
        for blk in d["blocks"]:
            for ln in blk.get("lines", []):
                txt = "".join(s["text"] for s in ln["spans"]).strip()
                if txt:
                    lines.append((ln["bbox"][1], next(s["size"] for s in ln["spans"] if s["text"].strip()),
                                  any(s["flags"] & 16 for s in ln["spans"]), txt))
        if not lines:
            continue
        foot = max(lines, key=lambda x: x[0])
        label = foot[3] if re.fullmatch(r"[ivxlc]+|\d+", foot[3]) else None
        if label is None:
            continue
        roman = not label.isdigit()
        for _, size, bold, txt in lines:
            if roman and size > 15.5 and bold and txt in ("CERTIFICATE", "DECLARATION", "ACKNOWLEDGEMENT",
                                                          "AREA OF WORK", "LIST OF FIGURES", "LIST OF TABLES",
                                                          "ABSTRACT", "TABLE OF CONTENT"):
                found.setdefault(f"front:{txt}", label)
            if roman:
                continue
            m = re.fullmatch(r"CHAPTER (\d+)", txt)
            if m and size > 15.5:
                found.setdefault(f"h1:{m.group(1)}", label)
            m = re.match(r"(\d+\.\d+) ", txt)
            if m and 13.5 < size < 14.5 and bold:
                found.setdefault(f"h2:{m.group(1)}", label)
            m = re.match(r"(Figure|Table) ([\dA]+\.\d+):", txt)
            if m and 10.5 < size < 11.5:
                found.setdefault(f"{'fig' if m.group(1) == 'Figure' else 'tab'}:{m.group(2)}", label)
            for key, _ in END_SECTIONS:
                if size > 15.5 and (txt == key or key.startswith(txt) and len(txt) > 12):
                    found.setdefault(f"end:{key}", label)
    pathlib.Path(out_json).write_text(json.dumps(found, indent=1))
    print(f"located {len(found)} entries, {len(doc)} pages -> {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--pages")
    ap.add_argument("--locate")
    a = ap.parse_args()
    if a.locate:
        locate(a.locate, a.pages)
    else:
        pages = json.loads(pathlib.Path(a.pages).read_text()) if a.pages and pathlib.Path(a.pages).exists() else None
        Report(pages).build(a.out)
