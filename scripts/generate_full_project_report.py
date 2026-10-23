"""
generate_full_project_report.py
-------------------------------
Generates the complete, comprehensive B.Tech Final Project Report
strictly following the format guidelines specified in:
'B.Tech Project format - 2026.pdf' for IIIT Bhopal.

Formatting Rules Enforced:
- Font Type: Times New Roman throughout
- Main Heading: 16 pt, Bold
- Sub-Heading: 14 pt, Bold
- Sub-Sub-Heading: 13 pt, Bold
- Body Text: 12 pt, Justified, 1.15 line spacing, 6 pt after
- Captions: 11 pt (Table caption ABOVE table, Figure caption BELOW figure)
- References: 10 pt, IEEE Format [1] - [25]
- Margins: 1 inch (72 pt) all sides
- Page Numbers: Roman (i-xii) for front matter, Arabic (1-N) for chapters
- Headers/Footers: Institution name centered in header, page number centered in footer
"""

import glob
import json
import os
import pathlib
import statistics
import docx
from docx.shared import Inches, Pt, RGBColor, Cm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Corrected report: new figures and a new file name, so the original report is kept
ASSETS_DIR = PROJECT_ROOT / "report_assets" / "corrected"
DOCX_OUT_PATH = PROJECT_ROOT / "BTech_Project_Report_Corrected_2026.docx"

# The second prediction target is ICU mortality (icu_expired), not sepsis.
# If the department requires the originally registered title, change it here only.
TITLE = "Federated Multimodal Graph Learning Framework for In-Hospital and ICU Mortality Prediction"
TITLE_UPPER = TITLE.upper()
RESULTS_DIR = PROJECT_ROOT / "results" / "corrected"


def load_benchmark_summary():
    with open(RESULTS_DIR / "benchmark_results.json") as f:
        return json.load(f)["summary"]


def gnn_convergence(rounds):
    """Mean over seeds of fedavg_gnn validation metrics at the given rounds."""
    hists = [json.load(open(p)) for p in sorted(glob.glob(str(RESULTS_DIR / "fedavg_gnn" / "seed*" / "training_history.json")))]
    out = []
    for rnd in rounds:
        m = lambda key: statistics.mean(h[rnd - 1][key] for h in hists)
        out.append((f"Round {rnd}", f"{m('val_loss'):.4f}", f"{m('mort_auroc'):.4f}",
                    f"{m('mort_auprc'):.4f}", f"{m('icu_auroc'):.4f}"))
    return out, len(hists)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def set_cell_background(cell, fill_hex):
    """Sets background shading of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    """Sets internal padding of a table cell in twips."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def set_table_borders(table, color="000000", size=4):
    """Apply solid borders to a table (all sides)."""
    tbl = table._tbl
    tbl_pr = tbl.tblPr if tbl.tblPr is not None else OxmlElement('w:tblPr')
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        element = OxmlElement(f'w:{edge}')
        element.set(qn('w:val'), 'single')
        element.set(qn('w:sz'), str(size))
        element.set(qn('w:space'), '0')
        element.set(qn('w:color'), color)
        borders.append(element)
    tbl_pr.append(borders)


def add_para(doc, text="", space_after=6, line_spacing=1.15,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False,
             size=12, color=None, first_indent=None):
    """Add a formatted paragraph with Times New Roman font."""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = line_spacing
    if first_indent:
        p.paragraph_format.first_line_indent = Inches(first_indent)
    if text:
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        rFonts = run._element.rPr.rFonts if run._element.rPr is not None else None
        if rFonts is None:
            rPr = run._element.get_or_add_rPr()
            rFonts = OxmlElement('w:rFonts')
            rFonts.set(qn('w:eastAsia'), 'Times New Roman')
            rFonts.set(qn('w:cs'), 'Times New Roman')
            rPr.insert(0, rFonts)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        if color:
            run.font.color.rgb = color
    return p


def page_break(doc):
    """Page break on the last paragraph (not a new one), so a full page doesn't leave a blank page."""
    body = doc.element.body
    last = body[-2] if body[-1].tag == qn('w:sectPr') else body[-1]
    if last.tag == qn('w:p'):
        doc.paragraphs[-1].add_run().add_break(WD_BREAK.PAGE)
    else:  # last element is a table: a separate break paragraph is needed
        doc.add_page_break()


def add_empty_para(doc, space_after=6):
    """Add an empty paragraph for spacing."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    return p


def add_heading_1(doc, text):
    """Chapter-level heading: 16pt, Bold, Dark color."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(16)
    run.bold = True
    run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return p


def add_heading_2(doc, text):
    """Section heading: 14pt, Bold."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(14)
    run.bold = True
    run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return p


def add_heading_3(doc, text):
    """Sub-section heading: 13pt, Bold."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(13)
    run.bold = True
    run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return p


def add_table_caption(doc, text):
    """Table caption: ABOVE the table, 11pt, Bold, Centered."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run.bold = True
    return p


def add_figure_caption(doc, text):
    """Figure caption: BELOW the figure, 11pt, Bold, Centered."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(16)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run.bold = True
    return p


def add_image_centered(doc, image_path, width_inches=5.8):
    """Insert a centered image."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.add_run().add_picture(str(image_path), width=Inches(width_inches))
    return p


def style_table_professional(table, has_header=True):
    """Apply professional styling to a table with borders."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="333333", size=4)

    for i, row in enumerate(table.rows):
        # Keep each row on one page; repeat the header row when the table spans pages
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement('w:cantSplit'))
        if has_header and i == 0:
            tr_pr.append(OxmlElement('w:tblHeader'))
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            if has_header and i == 0:
                # Header row: dark background, white text
                set_cell_background(cell, "1F3864")
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.name = "Times New Roman"
                        r.font.size = Pt(10)
                        r.bold = True
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            else:
                # Alternating row colors
                if i % 2 == 0:
                    set_cell_background(cell, "F2F6FC")
                else:
                    set_cell_background(cell, "FFFFFF")
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.name = "Times New Roman"
                        r.font.size = Pt(10)
                        r.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)


def style_table_simple(table):
    """Light table style without header formatting."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="666666", size=4)
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=60, bottom=60, left=100, right=100)
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(11)


def add_horizontal_line(doc):
    """Add a horizontal rule."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '999999')
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_page_footer(section, text):
    """Add footer text to a section."""
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def add_page_header(section, text):
    """Add header text to a section."""
    header = section.header
    header.is_linked_to_previous = False
    p = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    run.italic = True
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def add_page_number_footer(section):
    """Centered 'Page N' footer using a live Word PAGE field (continuous numbering)."""
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for text in ("Page ",):
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    run = p.add_run()
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    begin = OxmlElement('w:fldChar'); begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.set(qn('xml:space'), 'preserve'); instr.text = 'PAGE'
    end = OxmlElement('w:fldChar'); end.set(qn('w:fldCharType'), 'end')
    run._r.append(begin); run._r.append(instr); run._r.append(end)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_report():
    doc = docx.Document()
    # Don't stretch justified lines that end in a manual line break (bullet lists, equations)
    doc.settings.element.find(qn('w:compat')).insert(0, OxmlElement('w:doNotExpandShiftReturn'))

    # ── Default style configuration ──────────────────────────────────────────
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.space_after = Pt(6)

    # ── Configure Margins to 1.0 inch ────────────────────────────────────────
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.0)
        sec.right_margin = Inches(1.0)
        # Institute name in the header, continuous page number in the footer
        add_page_header(sec, "Indian Institute of Information Technology, Bhopal")
        add_page_number_footer(sec)

    # ═════════════════════════════════════════════════════════════════════════
    # 1. TITLE PAGE
    # ═════════════════════════════════════════════════════════════════════════
    
    # Top spacing (kept small so the title page fits on one page)
    add_empty_para(doc, space_after=6)

    add_para(doc, "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=2)
    add_para(doc, "BHOPAL (M.P.)",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14, space_after=6)
    add_para(doc, "Department of Computer Science and Engineering",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=4)
    
    add_horizontal_line(doc)

    add_empty_para(doc, space_after=12)

    add_para(doc, "B.Tech Major Project Report",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14, space_after=4)
    add_para(doc, "(Phase – II)",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=20)
    
    add_para(doc, "FEDERATED MULTIMODAL GRAPH LEARNING\nFRAMEWORK FOR IN-HOSPITAL AND\nICU MORTALITY PREDICTION",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=17, space_after=24,
             line_spacing=1.3)
    
    add_horizontal_line(doc)
    
    add_empty_para(doc, space_after=8)
    
    add_para(doc, "Submitted in partial fulfillment for the award of the degree of",
             align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=12, space_after=4)
    add_para(doc, "BACHELOR OF TECHNOLOGY",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=15, space_after=4)
    add_para(doc, "in Computer Science and Engineering",
             align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=12, space_after=24)
    
    # Author table
    t_auth = doc.add_table(rows=3, cols=2)
    t_auth.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Row 0: Headers
    c_sub = t_auth.rows[0].cells[0].paragraphs[0]
    r_sub = c_sub.add_run("Submitted by:")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(12)
    r_sub.bold = True
    r_sub.underline = True
    
    c_guide = t_auth.rows[0].cells[1].paragraphs[0]
    r_guide = c_guide.add_run("Under the Supervision of:")
    r_guide.font.name = "Times New Roman"
    r_guide.font.size = Pt(12)
    r_guide.bold = True
    r_guide.underline = True
    
    # Row 1: Student 1 / Supervisor
    c_s1 = t_auth.rows[1].cells[0].paragraphs[0]
    r_s1 = c_s1.add_run("Dhruv Savaliya\nScholar No: 21U02035")
    r_s1.font.name = "Times New Roman"
    r_s1.font.size = Pt(11)
    
    c_sup = t_auth.rows[1].cells[1].paragraphs[0]
    r_sup = c_sup.add_run("Project Supervisor\nAssistant Professor\nDepartment of CSE\nIIIT Bhopal (M.P.)")
    r_sup.font.name = "Times New Roman"
    r_sup.font.size = Pt(11)
    
    # Row 2: Student 2
    c_s2 = t_auth.rows[2].cells[0].paragraphs[0]
    r_s2 = c_s2.add_run("Menil Patel\nScholar No: 21U02052")
    r_s2.font.name = "Times New Roman"
    r_s2.font.size = Pt(11)
    
    # Style the author table
    for row in t_auth.rows:
        for cell in row.cells:
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.15
    
    add_empty_para(doc, space_after=8)
    
    add_para(doc, "B.Tech CSE – Final Year (Semester VIII)",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=4)
    add_para(doc, "Academic Year: 2025 – 2026",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=4)
    add_para(doc, "October 2026",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=6)
    
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # 2. CERTIFICATE
    # ═════════════════════════════════════════════════════════════════════════
    add_empty_para(doc, space_after=12)
    
    add_para(doc, "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY, BHOPAL (M.P.)",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14, space_after=4)
    add_para(doc, "Department of Computer Science and Engineering",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=8)
    
    add_horizontal_line(doc)
    
    add_para(doc, "CERTIFICATE",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=20)

    cert_text = (
        "This is to certify that the work embodied in this project report entitled "
        f"\u201c{TITLE_UPPER}\u201d "
        "has been satisfactorily completed by the following students of the Department of Computer Science and Engineering, "
        "Indian Institute of Information Technology, Bhopal:"
    )
    add_para(doc, cert_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=12)

    # Student details in numbered format
    add_para(doc, "1.  Dhruv Savaliya (Scholar No. 21U02035)", size=12, space_after=4,
             align=WD_ALIGN_PARAGRAPH.LEFT, bold=True)
    add_para(doc, "2.  Menil Patel (Scholar No. 21U02052)", size=12, space_after=12,
             align=WD_ALIGN_PARAGRAPH.LEFT, bold=True)

    cert_text2 = (
        "It is a bonafide piece of work, carried out under our guidance for the partial fulfillment "
        "of the requirements of the degree of Bachelor of Technology in Computer Science and Engineering "
        "during the academic year 2025\u20132026. The work presented in this report has not been submitted "
        "elsewhere for the award of any other degree or diploma."
    )
    add_para(doc, cert_text2, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=24)

    add_para(doc, "Date: ____________________", align=WD_ALIGN_PARAGRAPH.LEFT, size=12, space_after=4)
    add_para(doc, "Place: Bhopal (M.P.)", align=WD_ALIGN_PARAGRAPH.LEFT, size=12, space_after=48)

    # Signature table
    t_cert = doc.add_table(rows=2, cols=2)
    t_cert.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    for col_idx, (name, title) in enumerate([
        ("Project Supervisor", "Department of CSE\nIIIT Bhopal (M.P.)"),
        ("Head of Department", "Department of CSE\nIIIT Bhopal (M.P.)")
    ]):
        sig_cell = t_cert.rows[0].cells[col_idx].paragraphs[0]
        sig_run = sig_cell.add_run("\n\n\n_________________________")
        sig_run.font.name = "Times New Roman"
        sig_run.font.size = Pt(11)
        
        name_cell = t_cert.rows[1].cells[col_idx].paragraphs[0]
        name_run = name_cell.add_run(f"{name}\n{title}")
        name_run.font.name = "Times New Roman"
        name_run.font.size = Pt(11)
        name_run.bold = True

    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. DECLARATION
    # ═════════════════════════════════════════════════════════════════════════
    add_empty_para(doc, space_after=12)
    
    add_para(doc, "INDIAN INSTITUTE OF INFORMATION TECHNOLOGY, BHOPAL (M.P.)",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14, space_after=4)
    add_para(doc, "Department of Computer Science and Engineering",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=8)
    
    add_horizontal_line(doc)
    
    add_para(doc, "DECLARATION",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=20)

    decl_text = (
        f"We hereby declare that the major project report entitled \u201c{TITLE_UPPER}\u201d "
        "submitted herein is a genuine record "
        "of research and engineering work carried out by us in partial fulfillment of the requirements for "
        "the award of the degree of Bachelor of Technology in Computer Science and Engineering from the "
        "Indian Institute of Information Technology, Bhopal.\n\n"
        "The work presented in this project report has not been submitted in part or in whole for the award "
        "of any degree or diploma in any other institute or university. We further declare that all data, "
        "source code, experimental results, figures, and analyses presented in this document are authentic and "
        "original. All external references have been duly cited in the IEEE format.\n\n"
        "In case of any discrepancy or factual inaccuracy found in this report, we shall bear full responsibility."
    )
    add_para(doc, decl_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=48)

    # Signature table for declaration
    t_decl = doc.add_table(rows=2, cols=2)
    t_decl.alignment = WD_TABLE_ALIGNMENT.CENTER

    for col_idx, (name, scholar) in enumerate([
        ("Dhruv Savaliya", "Scholar No: 21U02035"),
        ("Menil Patel", "Scholar No: 21U02052")
    ]):
        sig_cell = t_decl.rows[0].cells[col_idx].paragraphs[0]
        sig_run = sig_cell.add_run("\n\n\n_________________________")
        sig_run.font.name = "Times New Roman"
        sig_run.font.size = Pt(11)

        name_cell = t_decl.rows[1].cells[col_idx].paragraphs[0]
        name_run = name_cell.add_run(f"{name}\n{scholar}\nB.Tech CSE, Final Year")
        name_run.font.name = "Times New Roman"
        name_run.font.size = Pt(11)
        name_run.bold = True

    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. ACKNOWLEDGEMENT
    # ═════════════════════════════════════════════════════════════════════════
    add_para(doc, "ACKNOWLEDGEMENT",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=20)

    ack_paras = [
        "We would like to express our deepest and most sincere gratitude to everyone who supported, "
        "encouraged, and guided us throughout the successful conceptualization, implementation, and "
        f"completion of our B.Tech major project titled \u201c{TITLE}.\u201d",

        "First and foremost, we extend our heartfelt gratitude to the esteemed Director, Indian Institute "
        "of Information Technology Bhopal, for providing state-of-the-art computational infrastructure, "
        "laboratory facilities, and an inspiring academic environment that made rigorous machine learning "
        "research possible.",

        "We are profoundly grateful to our Project Supervisor for invaluable guidance, patient mentoring, "
        "perceptive critique, and continuous encouragement throughout the lifecycle of this research. Their "
        "deep domain expertise in artificial intelligence, graph representation learning, and federated "
        "systems provided us with the direction required to design and evaluate a federated "
        "predictive framework.",

        "We also convey our gratitude to the Head of the Department and Department Coordinator, Department "
        "of Computer Science and Engineering, for administrative support, curriculum coordination, and "
        "constructive evaluations during project progress reviews.",

        "Our sincere thanks are extended to all faculty members and technical staff of the CSE Department "
        "for theoretical guidance and technical support. We are equally grateful to our fellow classmates "
        "and peers for stimulating technical discussions and constructive feedback.",

        "Finally, we express our profound gratitude to our parents and family members for their unconditional "
        "love, moral support, and endless patience, which served as the cornerstone of our perseverance "
        "throughout this challenging endeavor."
    ]

    for para_text in ack_paras:
        add_para(doc, para_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=10)

    add_empty_para(doc, space_after=24)
    add_para(doc, "Dhruv Savaliya (21U02035)\nMenil Patel (21U02052)",
             align=WD_ALIGN_PARAGRAPH.RIGHT, bold=True, size=12)
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # 5. ABSTRACT
    # ═════════════════════════════════════════════════════════════════════════
    add_para(doc, "ABSTRACT",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=18)

    abstract_paras = [
        "In modern intensive care units (ICUs), early and accurate risk stratification for life-threatening "
        "outcomes such as in-hospital and ICU mortality is vital for timely clinical intervention and "
        "resource allocation. Although machine learning models trained on Electronic Health Records (EHRs) "
        "have shown significant promise, their real-world deployment faces two major structural barriers: "
        "(1) privacy regulations (e.g., HIPAA, GDPR, DPDPA) that restrict centralizing sensitive "
        "protected health information across hospital networks, and (2) the failure of "
        "traditional models to exploit relational inductive biases between clinically similar patients "
        "who share acute physiological phenotypes.",

        "To study this tension between clinical privacy and predictive utility, this project "
        "implements a Federated Multimodal Graph Learning Framework and evaluates it in a single-machine "
        "federated simulation on the multi-center eICU Collaborative Research Database (207 hospitals, "
        "200,826 ICU encounters). Hospitals are split into 145 training, 31 validation and 31 test hospitals. "
        "The task is early prediction: each model sees only the first 24 hours of an ICU stay and predicts "
        "in-hospital and ICU mortality for patients still in the ICU at 24 hours (132,878 encounters). "
        "For each patient, the record is decomposed into two "
        "complementary modalities: (i) static tabular variables (demographics, admission diagnoses, baseline "
        "organ scores) encoded via a multi-layer perceptron into a 64-dimensional latent embedding, and "
        "(ii) time-series observations in 4-hour bins (laboratory tests, infusion rates, medications and "
        "treatments) modeled via a 2-layer Long Short-Term Memory (LSTM) network with packed "
        "sequence handling into a 64-dimensional temporal embedding. A multimodal fusion layer projects "
        "the concatenated representations into a 128-dimensional unified patient space.",

        "Within each hospital client, a patient similarity graph is built for every training batch "
        "via k-nearest neighbors (k=5) using cosine similarity, so edges only connect patients of the same "
        "hospital. A Graph Neural Network (PatientGNN) leveraging GraphSAGE, "
        "Graph Attention Networks (GAT), or Graph Convolutional Networks (GCN) propagates neighborhood clinical "
        "signals, capturing phenotypic correlations among acute patients. Attached dual multi-task prediction "
        "heads simultaneously predict In-Hospital Mortality and ICU Mortality using a positive-class "
        "weighted multi-task loss function. A central server aggregates local model weights via "
        "Federated Averaging (FedAvg), weighted by the number of examples each hospital trained on, so "
        "raw patient rows are not sent to the server in the simulation.",

        "On the 31 held-out test hospitals (18,014 patients), averaged over three training seeds, the "
        "proposed FedAvg + GNN model reaches an in-hospital mortality AUROC of 0.819 ± 0.007 (AUPRC "
        "0.400 ± 0.009) and an ICU mortality AUROC of 0.853 ± 0.005. It is consistently better than "
        "local-only training on a single hospital (AUROC 0.769, 0.660 and 0.583 for hospitals of 1,562, "
        "458 and 87 patients), and the gap grows as the single hospital gets smaller. Removing the graph "
        "(No-GNN ablation) lowers the in-hospital mortality AUROC slightly to 0.806 ± 0.024, so the graph "
        "gives a small but consistent gain on this task rather than a large one. These results show that a "
        "simple federated multimodal graph model can learn useful early mortality risk from decentralized "
        "ICU data; they come from a single-machine simulation and are not a clinical validation.",
    ]

    for para_text in abstract_paras:
        add_para(doc, para_text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=8)

    # Keywords
    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_kw.paragraph_format.space_before = Pt(12)
    p_kw.paragraph_format.space_after = Pt(6)
    r_kw_label = p_kw.add_run("Keywords: ")
    r_kw_label.font.name = "Times New Roman"
    r_kw_label.font.size = Pt(12)
    r_kw_label.bold = True
    r_kw = p_kw.add_run(
        "Federated Learning, Graph Neural Networks, Multimodal Deep Learning, eICU Dataset, "
        "ICU Mortality Prediction, Mortality Risk Stratification, Federated Simulation, Patient Similarity Networks."
    )
    r_kw.font.name = "Times New Roman"
    r_kw.font.size = Pt(12)
    r_kw.italic = True

    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # 6. TABLE OF CONTENTS
    # ═════════════════════════════════════════════════════════════════════════
    add_para(doc, "TABLE OF CONTENTS",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=18)

    toc_data = [
        ("", "Certificate", "2"),
        ("", "Declaration", "3"),
        ("", "Acknowledgement", "4"),
        ("", "Abstract", "5"),
        ("", "Table of Contents", "7"),
        ("", "List of Figures", "10"),
        ("", "List of Tables", "10"),
        ("1", "Introduction", "12"),
        ("1.1", "Clinical Background & Motivation", "12"),
        ("1.2", "The Privacy-Utility Conundrum in Healthcare", "12"),
        ("1.3", "Challenges in Multimodal Critical Care Modeling", "13"),
        ("1.4", "Proposed Federated Graph Learning Paradigm", "13"),
        ("1.5", "Organization of the Report", "13"),
        ("2", "Literature Review", "15"),
        ("2.1", "Clinical Risk Stratification Systems", "15"),
        ("2.2", "Deep Learning for Electronic Health Records", "15"),
        ("2.3", "Multimodal Representation Learning in Medicine", "16"),
        ("2.4", "Graph Neural Networks & Patient Similarity Networks", "16"),
        ("2.5", "Federated Learning in Distributed Healthcare", "17"),
        ("2.6", "Critical Analysis and Research Gaps", "18"),
        ("3", "Problem Definition and Objectives", "19"),
        ("3.1", "Formal Problem Formulation", "19"),
        ("3.2", "Research Objectives", "19"),
        ("3.3", "Technical Scope & Boundary Conditions", "20"),
        ("4", "Proposed Methodology and Work Description", "21"),
        ("4.1", "System Architectural Overview", "21"),
        ("4.2", "Multimodal Data Ingestion & Canonical Alignment", "22"),
        ("4.3", "Modality-Specific Encoders", "22"),
        ("4.4", "Multimodal Non-Linear Fusion Layer", "23"),
        ("4.5", "Intra-Hospital Patient Graph Construction", "23"),
        ("4.6", "Graph Representation Learning (PatientGNN)", "24"),
        ("4.7", "Multi-Task Clinical Prediction Heads", "24"),
        ("4.8", "Joint Multi-Task Objective with Imbalance Weighting", "24"),
        ("5", "Proposed Algorithms", "26"),
        ("5.1", "Client-Side Local Multimodal Graph Training", "26"),
        ("5.2", "Server-Side Federated Averaging Coordination", "26"),
        ("6", "Flowchart, Block Diagram, and DFD", "28"),
        ("6.1", "System Architectural Block Diagram", "28"),
        ("6.2", "Data Flow Diagram (DFD)", "28"),
        ("6.3", "Graph Construction Flowchart", "29"),
        ("7", "Implementation Details & Technology Stack", "30"),
        ("7.1", "Hardware & Software Environments", "30"),
        ("7.2", "Dataset Specification (eICU Database)", "30"),
        ("7.3", "Data Pipeline Implementation", "31"),
        ("7.4", "Deep Learning & GNN Modules", "31"),
        ("7.5", "Federated Training Engine", "32"),
        ("8", "Result Discussion and Comparative Analysis", "34"),
        ("8.1", "Experimental Setup & Evaluation Protocol", "34"),
        ("8.2", "Clinical Evaluation Metrics", "34"),
        ("8.3", "Federated Convergence Dynamics", "34"),
        ("8.4", "Comparative Baseline Evaluation", "35"),
        ("8.5", "Patient Graph Visualization", "37"),
        ("9", "Conclusion and Future Scope", "39"),
        ("9.1", "Summary of Contributions", "39"),
        ("9.2", "Limitations of the Current Study", "39"),
        ("9.3", "Future Research Directions", "40"),
        ("", "References (IEEE Format)", "41"),
        ("", "List of Publications", "43"),
        ("", "Plagiarism & Authenticity Report", "44"),
    ]

    t_toc = doc.add_table(rows=len(toc_data) + 1, cols=3)
    t_toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_toc, color="444444", size=4)

    # Header
    for j, h in enumerate(["Section", "Title", "Page"]):
        r = t_toc.rows[0].cells[j].paragraphs[0].add_run(h)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        r.bold = True
    set_cell_background(t_toc.rows[0].cells[0], "1F3864")
    set_cell_background(t_toc.rows[0].cells[1], "1F3864")
    set_cell_background(t_toc.rows[0].cells[2], "1F3864")
    for cell in t_toc.rows[0].cells:
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for idx, (sec_no, title, page) in enumerate(toc_data):
        row = t_toc.rows[idx + 1]
        # Section number
        p0 = row.cells[0].paragraphs[0]
        r0 = p0.add_run(sec_no)
        r0.font.name = "Times New Roman"
        r0.font.size = Pt(10)
        p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Is it a chapter-level (no dot in section) or subsection?
        is_chapter = sec_no and '.' not in sec_no
        is_front = sec_no == ""
        
        # Title
        p1 = row.cells[1].paragraphs[0]
        r1 = p1.add_run(title)
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(10)
        if is_chapter or is_front:
            r1.bold = True
        
        # Page number
        p2 = row.cells[2].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(page)
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(10)
        
        # Subtle alternating rows
        if (idx + 1) % 2 == 0:
            for cell in row.cells:
                set_cell_background(cell, "F5F7FA")
        
        for cell in row.cells:
            set_cell_margins(cell, top=40, bottom=40, left=80, right=80)

    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # 7. LIST OF FIGURES & LIST OF TABLES
    # ═════════════════════════════════════════════════════════════════════════
    add_para(doc, "LIST OF FIGURES",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=18)

    figures_list = [
        ("Figure 1", "Federated Multimodal Graph Learning End-to-End System Architecture", "28"),
        ("Figure 2", "Data Flow Diagram (Level 0 Context and Level 1 Functional Pipeline)", "29"),
        ("Figure 3", "Hospital Patient Population Scale and Distribution across 207 eICU Sites", "31"),
        ("Figure 4", "Validation Loss and Mortality AUROC per Round (FedAvg + GNN vs No-GNN, mean ± SD over 3 seeds)", "35"),
        ("Figure 5", "Test-Hospital Performance: FedAvg + GNN vs No-GNN vs Local-Only", "37"),
        ("Figure 6", "Patient Similarity Graph Built by the Trained Model on Test Hospital 120", "38"),
    ]

    t_lof = doc.add_table(rows=len(figures_list) + 1, cols=3)
    t_lof.rows[0].cells[0].paragraphs[0].add_run("Figure No.").bold = True
    t_lof.rows[0].cells[1].paragraphs[0].add_run("Description").bold = True
    t_lof.rows[0].cells[2].paragraphs[0].add_run("Page No.").bold = True
    for i, (fno, desc, pg) in enumerate(figures_list):
        r = t_lof.rows[i + 1]
        r.cells[0].paragraphs[0].add_run(fno)
        r.cells[1].paragraphs[0].add_run(desc)
        p_pg = r.cells[2].paragraphs[0]
        p_pg.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_pg.add_run(pg)
    style_table_professional(t_lof)

    add_empty_para(doc, space_after=24)

    add_para(doc, "LIST OF TABLES",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16, space_after=18)

    tables_list = [
        ("Table 1", "Multimodal Feature Modalities in Critical Care Medicine", "16"),
        ("Table 2", "Comparison of Contemporary Machine Learning Healthcare Frameworks", "17"),
        ("Table 3", "Canonical Feature Dimensions and Modality Specifications", "21"),
        ("Table 4", "Hyperparameter Configuration for Federated Patient Graph Model", "32"),
        ("Table 5", "Validation Convergence of the Proposed FedAvg + GNN Model (mean of 3 seeds)", "35"),
        ("Table 6", "Test-Hospital Results (mean ± SD over 3 seeds)", "36"),
    ]

    t_lot = doc.add_table(rows=len(tables_list) + 1, cols=3)
    t_lot.rows[0].cells[0].paragraphs[0].add_run("Table No.").bold = True
    t_lot.rows[0].cells[1].paragraphs[0].add_run("Description").bold = True
    t_lot.rows[0].cells[2].paragraphs[0].add_run("Page No.").bold = True
    for i, (tno, desc, pg) in enumerate(tables_list):
        r = t_lot.rows[i + 1]
        r.cells[0].paragraphs[0].add_run(tno)
        r.cells[1].paragraphs[0].add_run(desc)
        p_pg = r.cells[2].paragraphs[0]
        p_pg.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_pg.add_run(pg)
    style_table_professional(t_lot)

    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 1: INTRODUCTION
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 1\nINTRODUCTION")

    add_heading_2(doc, "1.1 Clinical Background & Motivation")
    add_para(doc, (
        "The Intensive Care Unit (ICU) represents the most technologically intensive and data-dense "
        "environment in modern healthcare. Critically ill patients admitted to ICUs suffer from severe, "
        "rapidly progressing pathologies such as septic shock, multi-organ failure, and acute respiratory "
        "distress syndrome (ARDS). Among these conditions, sepsis remains a leading cause of in-hospital "
        "mortality and healthcare expenditure worldwide, accounting for over 11 million global deaths annually "
        "[4]. In critical care settings, patient deterioration often unfolds across hours or even minutes. "
        "Traditional clinical monitoring relies heavily on episodic physician examinations and heuristic "
        "scoring systems such as the Acute Physiology and Chronic Health Evaluation (APACHE) [1] and "
        "Sequential Organ Failure Assessment (SOFA) [3]. While clinically grounded, these systems are "
        "fundamentally static, labor-intensive, and inadequate for dynamic, real-time risk stratification."
    ))
    add_para(doc, (
        "The ubiquity of Electronic Health Record (EHR) systems in modern hospitals has made continuous "
        "physiological data available at unprecedented scale. Every hour, bedside monitors record physiological "
        "waveforms, clinical laboratories produce biochemistry panels, and clinicians administer titrated "
        "vasopressors and fluid boluses. Developing predictive artificial intelligence (AI) systems capable of "
        "using these multi-stream inputs to estimate the risk of death early in the ICU stay, while there "
        "is still time to act, is therefore of clear clinical importance [5]."
    ))

    add_heading_2(doc, "1.2 The Privacy-Utility Conundrum in Healthcare")
    add_para(doc, (
        "Despite the availability of powerful machine learning architectures, real-world deployment across "
        "multi-hospital health systems faces a critical bottleneck: data governance and patient confidentiality. "
        "Electronic health records contain highly sensitive Protected Health Information (PHI). Legal frameworks "
        "such as the Health Insurance Portability and Accountability Act (HIPAA) in the United States, the "
        "General Data Protection Regulation (GDPR) in the European Union, and the Digital Personal Data "
        "Protection Act (DPDPA) in India strictly govern the transmission and centralized pooling of healthcare "
        "records across jurisdictional and organizational boundaries."
    ))
    add_para(doc, (
        "Historically, training deep learning models required centralizing raw patient records into a single "
        "data warehouse. For independent hospitals, this centralized model introduces insurmountable legal "
        "liabilities, regulatory compliance hazards, and commercial vulnerabilities. Conversely, training "
        "models on data from a single hospital in isolation causes severe algorithmic bias and over-fitting, "
        "as individual sites exhibit idiosyncratic demographics, local clinical workflows, and limited "
        "statistical samples. This tension between institutional data compartmentalization and cross-site "
        "statistical generalizability is known as the healthcare privacy-utility conundrum [22]."
    ))

    add_heading_2(doc, "1.3 Challenges in Multimodal Critical Care Modeling")
    add_para(doc, (
        "Modeling intensive care EHR datasets introduces three distinct technical challenges that demand "
        "specialized engineering solutions:"
    ))
    add_para(doc, (
        "Challenge 1 \u2014 Heterogeneous Modality Representation: ICU patient records are inherently multimodal. "
        "They encompass static tabular demographics (age, biological sex, admission type, chronic comorbidities) "
        "and dynamic time-series (laboratory results, intravenous infusions, medications and treatments). "
        "Static variables capture baseline physiological reserve, whereas "
        "temporal streams capture acute deterioration trajectories. Fusing these fundamentally disparate "
        "temporal granularities requires specialized modality encoders [11]."
    ))
    add_para(doc, (
        "Challenge 2 \u2014 Inter-Patient Clinical Relatedness: Conventional machine learning models treat each "
        "patient admission as an independent, identically distributed (I.I.D.) vector in Euclidean space. "
        "However, clinical pathology is profoundly relational. Patients exhibiting similar hemodynamic "
        "phenotypes (e.g., refractory hypotension coupled with elevated serum lactate) share underlying "
        "pathophysiological risk trajectories. Ignoring patient similarity networks discards valuable "
        "relational inductive biases [15]."
    ))
    add_para(doc, (
        "Challenge 3 \u2014 Class Imbalance & Non-I.I.D. Hospitals: death is a low-frequency, high-stakes "
        "outcome: in the eICU data used here, 9.0% of ICU encounters end in in-hospital death and 5.4% in "
        "ICU death. Furthermore, patient volumes across community hospitals and tertiary academic medical "
        "centers vary by orders of magnitude (from 3 to 7,059 encounters per hospital in the eICU "
        "database), creating strong statistical heterogeneity [25]."
    ))

    add_heading_2(doc, "1.4 Proposed Federated Graph Learning Paradigm")
    add_para(doc, (
        "To overcome these challenges, this project presents an end-to-end Federated Multimodal Graph "
        "Learning Framework for early In-Hospital and ICU Mortality prediction. In the federated design, raw "
        "patient records stay with each hospital; in this project the hospitals are simulated as separate "
        "clients inside one program on one machine. Each hospital client independently "
        "executes feature extraction, modality encoding (Tabular MLP + Time-series LSTM), and multimodal "
        "fusion. Within each hospital, a patient graph is constructed for every mini-batch using k-nearest "
        "neighbors (k-NN) over unified patient embeddings. A Graph Neural Network (PatientGNN) then "
        "passes information between similar patients. Only model weights are sent to the central "
        "server for Federated Averaging (FedAvg) aggregation [20]. The project combines multimodal "
        "representation learning, graph-based patient similarity and federated optimization in one "
        "prototype, and measures on held-out hospitals whether each part helps."
    ))

    add_heading_2(doc, "1.5 Organization of the Report")
    add_para(doc, (
        "The remainder of this report is organized as follows:\n\n"
        "\u2022 Chapter 2 conducts a comprehensive literature review of clinical risk scores, EHR deep learning, "
        "multimodal fusion techniques, graph neural networks, and federated learning systems.\n"
        "\u2022 Chapter 3 defines the formal problem statement, mathematical notation, and project objectives.\n"
        "\u2022 Chapter 4 elaborates the proposed system methodology and component architectures.\n"
        "\u2022 Chapter 5 formalizes the client-side and server-side algorithms.\n"
        "\u2022 Chapter 6 presents architectural block diagrams and data flow diagrams.\n"
        "\u2022 Chapter 7 details the software implementation and experimental environment.\n"
        "\u2022 Chapter 8 presents empirical results, convergence curves, and comparative benchmarks.\n"
        "\u2022 Chapter 9 concludes the study and outlines future research avenues."
    ))
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 2: LITERATURE REVIEW
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 2\nLITERATURE REVIEW")

    add_heading_2(doc, "2.1 Clinical Risk Stratification Systems in Intensive Care")
    add_para(doc, (
        "Risk stratification in critical care has traditionally relied on physiological scoring indices "
        "developed through multi-center logistic regression. The Acute Physiology and Chronic Health "
        "Evaluation (APACHE II, III, IV) developed by Knaus et al. [1] computes a composite integer score "
        "based on worst-observed vital signs and laboratory parameters during the initial 24 hours of ICU "
        "admission. Similarly, the Simplified Acute Physiology Score (SAPS II) [2] and the Sequential Organ "
        "Failure Assessment (SOFA) [3] quantify acute morbidity across cardiovascular, respiratory, "
        "neurological, renal, hepatic, and hematological organ systems."
    ))
    add_para(doc, (
        "In the context of sepsis, the Third International Consensus Definitions for Sepsis and Septic "
        "Shock (Sepsis-3) [4] operationalized sepsis as life-threatening organ dysfunction caused by a "
        "dysregulated host response to infection, clinically identified by an acute increase in SOFA score "
        "of 2 or more points. Although universally utilized for clinical benchmarking, these scoring systems "
        "suffer from profound structural limitations: (i) they rely on coarse physiological discretizations, "
        "(ii) they fail to model complex non-linear interactions across organ systems, (iii) they discard "
        "the rich temporal dynamics of continuous vital sign signals, and (iv) they cannot adapt dynamically "
        "to evolving clinical trajectories."
    ))

    add_heading_2(doc, "2.2 Deep Learning for Electronic Health Records (EHR)")
    add_para(doc, (
        "The advent of deep neural networks enabled direct learning of hierarchical feature representations "
        "from high-dimensional EHR data [5]. Early breakthroughs utilized Recurrent Neural Networks (RNNs) "
        "and Long Short-Term Memory (LSTM) networks to model irregular temporal sequences. Choi et al. [6] "
        "introduced Doctor AI, applying recurrent networks to predict differential diagnoses and medication "
        "orders over longitudinal patient visits. Lipton et al. [7] demonstrated that LSTMs directly handling "
        "missing values and measurement frequencies outperformed classical clinical baselines in predicting "
        "128 diagnostic categories from pediatric intensive care time-series."
    ))
    add_para(doc, (
        "Subsequent research explored Transformer-based self-attention mechanisms. Vaswani et al. [8] "
        "revolutionized sequence modeling, leading to clinical adaptations such as Med-BERT [9] and "
        "BEHRT [10], which pre-train bidirectional attention representations over structured medical codes. "
        "While attention mechanisms excel at capturing long-range dependencies, standard recurrent and "
        "transformer models typically process each patient in isolation, ignoring the non-Euclidean similarity "
        "relationships existing across patient populations admitted with comparable disease etiologies."
    ))

    add_heading_2(doc, "2.3 Multimodal Representation Learning in Medicine")
    add_para(doc, (
        "Clinical medicine is intrinsically multimodal. Diagnostic reasoning synthesizes demographic history, "
        "physical examinations, continuous telemetry, laboratory biochemistry, and clinical documentation. "
        "Multimodal machine learning frameworks aim to integrate these diverse data modalities into a joint "
        "latent representation space [11]. In critical care modeling, two primary modalities dominate: static "
        "tabular variables (demographics, admission severity, chronic comorbidities) and dynamic multivariate "
        "time-series (continuous vitals, hourly laboratory values, medication titration)."
    ))
    add_para(doc, (
        "Fusion techniques are categorized into early fusion (concatenation of raw feature vectors), late "
        "fusion (combining modality-specific decision logits via ensemble averaging), and intermediate fusion "
        "(learning joint representation spaces via non-linear projection or cross-attention) [12]. Recent "
        "work by Bardak and Tan [13] showed that combining entities extracted from clinical notes with ICU "
        "time series improved mortality and length-of-stay prediction over a time-series-only model. However, "
        "most multimodal "
        "architectures operate almost exclusively in centralized environments, failing to account for "
        "institutional data privacy boundaries."
    ))

    # Table 1
    add_table_caption(doc, "Table 1: Multimodal Feature Modalities in Critical Care Medicine")
    t1 = doc.add_table(rows=5, cols=4)
    headers = ["Modality Type", "Clinical Data Streams", "Mathematical Representation", "Primary Neural Encoder"]
    for j, h in enumerate(headers):
        t1.rows[0].cells[j].paragraphs[0].add_run(h)
    rows_t1 = [
        ("Static Tabular", "Demographics (Age, Gender, BMI), Admission Type, Comorbidities",
         "Fixed-size vector x_tab \u2208 \u211d\u2076\u2077", "Multi-Layer Perceptron (MLP)"),
        ("Dynamic Time-Series", "Laboratory panels, infusion rates, medications, treatments (4-hour bins)",
         "Sequential tensor X_ts \u2208 \u211d\u1d40\u02e3\u2077\u2074", "2-Layer LSTM"),
        ("Relational Graph", "Patient similarity edges within a hospital mini-batch",
         "Adjacency matrix A \u2208 \u211d\u1d3a\u02e3\u1d3a", "GraphSAGE / GAT / GCN"),
        ("Multimodal Fused", "Joint tabular-temporal latent patient representation",
         "Unified embedding z \u2208 \u211d\u00b9\u00b2\u2078", "Non-linear Projection Layer"),
    ]
    for i, row in enumerate(rows_t1):
        for j, val in enumerate(row):
            t1.rows[i + 1].cells[j].paragraphs[0].add_run(val)
    style_table_professional(t1)

    add_heading_2(doc, "2.4 Graph Neural Networks & Patient Similarity Networks")
    add_para(doc, (
        "Graph Neural Networks (GNNs) extend deep learning to non-Euclidean graph-structured domains [14]. "
        "In healthcare informatics, GNNs have been applied along two major axes: knowledge graph modeling "
        "(e.g., drug-drug interaction graphs, medical ontology hierarchies) and Patient Similarity Networks "
        "(PSNs) [15]. In a PSN, nodes represent individual patients, and edges represent clinical similarity "
        "computed over diagnostic codes, genetic profiles, or physiological trajectories."
    ))
    add_para(doc, (
        "Pioneering architectures include Graph Convolutional Networks (GCN) by Kipf & Welling [16], which "
        "perform spectral first-order localized graph convolutions; GraphSAGE by Hamilton et al. [17], which "
        "enables inductive representation learning on large graphs via neighborhood sampling and aggregation; "
        "and Graph Attention Networks (GAT) by Veli\u010dkovi\u0107 et al. [18], which employ self-attention to "
        "dynamically weight neighbor importance. In intensive care modeling, Rocheteau et al. [19] combined "
        "an LSTM with a GNN over a diagnosis-based patient graph and improved length-of-stay prediction on "
        "eICU. Such graph learning methods, however, assume access to one centralized patient graph, which "
        "is not available when hospitals cannot share patient data."
    ))

    add_heading_2(doc, "2.5 Federated Learning in Distributed Healthcare Systems")
    add_para(doc, (
        "Federated Learning (FL), pioneered by McMahan et al. [20] through the Federated Averaging (FedAvg) "
        "algorithm, enables decentralized training of machine learning models without centralizing raw "
        "training data. Participating client devices or institutions train local models on private datasets "
        "and share only parameter weights or gradient updates with an aggregating central coordinator."
    ))
    add_para(doc, (
        "In healthcare, cross-silo federated learning has emerged as the premier privacy-preserving "
        "paradigm [21]. Rieke et al. [22] argued that federated learning can let hospitals collaborate "
        "without pooling patient data while reaching performance close to centralized training. "
        "Multi-institution studies have demonstrated federated tumor "
        "segmentation (Sheller et al. [23]) and COVID-19 triage (Dayan et al. [24]) across international "
        "healthcare networks."
    ))
    add_para(doc, (
        "However, healthcare federated learning faces significant domain-specific hurdles: (1) System "
        "Heterogeneity: hospitals possess highly disparate computational hardware; (2) Statistical "
        "Heterogeneity (Non-I.I.D. Data): patient populations, disease prevalence, and clinical protocols "
        "differ dramatically across geographic and socio-economic lines; and (3) Relational Graph "
        "Constraints: standard FL aggregates models across isolated vectors, lacking mechanisms to incorporate "
        "intra-institutional relational graph topologies."
    ))

    # Table 2
    add_table_caption(doc, "Table 2: Comparison of Contemporary Machine Learning Healthcare Frameworks")
    t2 = doc.add_table(rows=5, cols=5)
    headers2 = ["Framework / Paradigm", "Data Privacy", "Multimodal Support", "Graph Topology", "Decentralized"]
    for j, h in enumerate(headers2):
        t2.rows[0].cells[j].paragraphs[0].add_run(h)
    rows_t2 = [
        ("Centralized DL [7, 8]", "None", "High", "None", "No"),
        ("APACHE/SOFA [1, 3]", "High (Local)", "Low", "None", "Yes"),
        ("Standard FedAvg [20, 22]", "High (Weights)", "Moderate", "None", "Yes"),
        ("This project", "Weights only (simulated)", "Tabular + time series (67D + 74D)", "k-NN graph per batch",
         "145 training hospitals (simulated)"),
    ]
    for i, row in enumerate(rows_t2):
        for j, val in enumerate(row):
            t2.rows[i + 1].cells[j].paragraphs[0].add_run(val)
    style_table_professional(t2)

    add_heading_2(doc, "2.6 Critical Analysis and Research Gaps")
    add_para(doc, (
        "A rigorous synthesis of the surveyed literature reveals three fundamental research gaps that "
        "this project directly addresses:"
    ))
    add_para(doc, (
        "Gap 1 \u2014 Disconnect Between GNNs and Federated Privacy: Existing patient similarity graph "
        "models assume that a global graph spanning all patients across all medical centers can be "
        "constructed. In practice, sharing patient records across hospitals is restricted by privacy law "
        "and data agreements. Few architectures keep graph convolutions inside each hospital while "
        "aggregating model parameters globally."
    ))
    add_para(doc, (
        "Gap 2 \u2014 Limited Multimodal Integration in Federated Healthcare: Prior healthcare FL studies "
        "have focused almost exclusively on computer vision (medical imaging like CT/MRI) or isolated "
        "unimodal EHR codes. Jointly modeling static tabular clinical variables and high-frequency dynamic "
        "physiological time-series within a federated paradigm remains underexplored."
    ))
    add_para(doc, (
        "Gap 3 \u2014 Neglect of Extreme Clinical Imbalance in Multi-Task FL: Simultaneous prediction of "
        "correlated critical care events (in-hospital and ICU mortality) within federated learning under "
        "positive outcome scarcity (often <10%) is rarely evaluated on large-scale real-world multi-center "
        "benchmarks."
    ))
    add_para(doc, (
        "This project explores these gaps with an end-to-end prototype that combines canonical multimodal "
        "EHR encoding, intra-hospital patient graph learning, positive-class weighted multi-task prediction "
        "and federated averaging, evaluated on eICU with a hospital-level train / validation / test split."
    ))
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 3: PROBLEM DEFINITION AND OBJECTIVES
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 3\nPROBLEM DEFINITION AND OBJECTIVES")

    add_heading_2(doc, "3.1 Formal Problem Formulation")
    add_para(doc, (
        "Let H = {H\u2081, H\u2082, ..., H\u2096} denote the K = 145 training hospitals (clients) of the multi-center "
        "eICU Collaborative Research Database; a further 31 hospitals are used for validation and 31 for "
        "testing. Each hospital H\u2096 holds a local dataset D\u2096 of N\u2096 ICU stays that lasted at least 24 hours:"
    ))
    add_para(doc, (
        "    D\u2096 = {(x_tab\u207a\u2096\u00b7\u2071, X_ts\u207a\u2096\u00b7\u2071, y_mort\u207a\u2096\u00b7\u2071, "
        "y_icu\u207a\u2096\u00b7\u2071)}\u1d62\u208c\u2081\u1d3a\u2096"
    ), align=WD_ALIGN_PARAGRAPH.CENTER, size=11, space_after=8, italic=True)
    add_para(doc, (
        "where:\n"
        "\u2022 x_tab \u2208 \u211d\u2076\u2077 is a 67-dimensional vector of static demographic, severity, and admission variables.\n"
        "\u2022 X_ts \u2208 \u211d^(T\u1d62 \u00d7 74) is a time-series matrix of 74 variables (labs, infusions, medications, "
        "treatments) in 4-hour bins from the first 24 hours of the ICU stay (T\u1d62 \u2264 6 bins).\n"
        "\u2022 y_mort \u2208 {0, 1} is the ground-truth binary label indicating in-hospital mortality.\n"
        "\u2022 y_icu \u2208 {0, 1} is the ground-truth binary label indicating ICU mortality.\n"
        "Because only stays of at least 24 hours are included, the input window always ends before "
        "the outcome is known."
    ))
    add_para(doc, (
        "In a real deployment, D\u2096 would stay inside hospital H\u2096 and the server S would only receive model "
        "weights; this project simulates that setting on one machine. The objective is to train a global "
        "parameter vector W* across the K hospitals that minimizes the global multi-task empirical risk:"
    ))
    add_para(doc, (
        "    min_W F(W) = \u2211\u2096\u208c\u2081\u1d37 (N\u2096 / N_total) \u00b7 F\u2096(W)"
    ), align=WD_ALIGN_PARAGRAPH.CENTER, size=11, space_after=8, italic=True)
    add_para(doc, (
        "where N_total = \u2211\u2096 N\u2096 is the total patient population across the federation, and F\u2096(W) "
        "is the local multi-task loss evaluated exclusively on local data D\u2096. In practice, each "
        "communication round samples 10 hospitals and weights their updates by the number of examples "
        "they trained on (Chapter 5)."
    ))

    add_heading_2(doc, "3.2 Research Objectives")
    add_para(doc, (
        "The primary engineering and scientific objectives of this research project are:"
    ))
    objectives = [
        "Objective 1 \u2014 Canonical Multimodal Standardization: Design and implement an automated schema "
        "alignment pipeline resolving heterogeneous feature dimensions across all 207 eICU hospitals, "
        "standardizing tabular features to exactly 67 dimensions and continuous time-series to 74 dimensions.",

        "Objective 2 \u2014 Modality-Specific Feature Learning: Develop specialized deep learning encoders\u2014"
        "a Multi-Layer Perceptron (MLP) for static features, and a 2-layer LSTM network with packed sequence "
        "handling for variable-length physiological trajectories.",

        "Objective 3 \u2014 Intermediate Multimodal Fusion: Formulate a non-linear projection layer combining "
        "tabular and temporal embeddings into a unified 128-dimensional latent patient representation space.",

        "Objective 4 \u2014 Intra-Hospital Patient Graph Construction: Construct local patient similarity "
        "graphs inside each hospital using k-nearest neighbors (k=5) over cosine similarity, built per "
        "mini-batch of 32 patients so that no edge crosses hospitals.",

        "Objective 5 \u2014 Graph Representation Learning: Implement Graph Neural Network layers (GraphSAGE, "
        "GAT, GCN) that pass information between similar patients; the experiments use GraphSAGE.",

        "Objective 6 \u2014 Multi-Task Outcome Prediction: Engineer dual classification heads for simultaneous "
        "In-Hospital and ICU Mortality prediction, optimized using positive-class weighted loss.",

        "Objective 7 \u2014 Federated Orchestration & Benchmarking: Implement Federated Averaging (FedAvg), "
        "training across hospital clients and benchmarking against Local-Only and No-GNN ablation baselines "
        "on held-out test hospitals.",
    ]
    for obj in objectives:
        add_para(doc, obj, space_after=8)

    add_heading_2(doc, "3.3 Technical Scope & Boundary Conditions")
    add_para(doc, (
        "To ensure clinical fidelity, reproducibility, and computational rigor, the scope of this project "
        "is governed by the following constraints:\n\n"
        "\u2022 Modality Scope: Strictly restricted to 2 clinical modalities: Structured Static Tabular data "
        "and Dynamic Temporal Time-Series. Unstructured clinical notes are not used.\n"
        "\u2022 Graph Boundary: Patient similarity graphs are built within one hospital (and one mini-batch). "
        "Edges connect patients of the same hospital only.\n"
        "\u2022 Federated Aggregation: Only neural network weights are passed to the server step. Raw data, "
        "embeddings, and graph structures stay in the client step (simulated on one machine).\n"
        "\u2022 Prediction Window: Time-series inputs are limited to the first 24 hours of the ICU stay (six "
        "4-hour bins), and only patients still in the ICU at 24 hours are included, so the input never "
        "contains the outcome.\n"
        "\u2022 Evaluation Protocol: Hospitals are split into 145 training, 31 validation (checkpoint selection "
        "only) and 31 test hospitals (final evaluation only)."
    ))
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 4: PROPOSED METHODOLOGY AND WORK DESCRIPTION
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 4\nPROPOSED METHODOLOGY AND WORK DESCRIPTION")

    add_heading_2(doc, "4.1 System Architectural Overview")
    add_para(doc, (
        "The proposed Federated Multimodal Graph Learning Framework is structured into an end-to-end, "
        "multi-stage processing pipeline spanning decentralized hospital clients and an aggregating central "
        "coordinator. Each hospital client executes an autonomous five-stage workflow: (1) Data Ingestion and "
        "Canonical Feature Alignment, (2) Modality-Specific Feature Encoding, (3) Intermediate Multimodal "
        "Fusion, (4) Intra-Hospital Patient Graph Assembly and GNN Propagation, and (5) Multi-Task Loss "
        "Computation. The central coordinator orchestrates communication rounds, samples participating "
        "hospital clients, aggregates parameter weights via FedAvg (weighted by examples trained), and "
        "selects the best round on 31 validation hospitals. A separate set of 31 test hospitals is used "
        "only once, for the final benchmark in Chapter 8."
    ))

    # Table 3
    add_table_caption(doc, "Table 3: Canonical Feature Dimensions and Modality Specifications")
    t3 = doc.add_table(rows=9, cols=4)
    headers3 = ["Feature Category", "Raw Data Source", "Canonical Dimension", "Normalization"]
    for j, h in enumerate(headers3):
        t3.rows[0].cells[j].paragraphs[0].add_run(h)
    rows_t3 = [
        ("Tabular: demographics & admission", "patient.csv", "10 variables", "Per-hospital z-score"),
        ("Tabular: APACHE scores, GCS, day-1 labs, comorbidities", "apachePatientResult.csv, apachePredVar.csv",
         "18 variables", "Per-hospital z-score"),
        ("Tabular: diagnosis categories", "diagnosis.csv", "18 variables (17 flags + count)", "Per-hospital z-score"),
        ("Tabular: admission drugs", "admissionDrug.csv", "21 variables (20 flags + count)", "Per-hospital z-score"),
        ("Time series: laboratory tests", "lab.csv", "20 variables", "Pre-standardized in processed files"),
        ("Time series: infusions", "infusionDrug.csv", "20 variables", "Pre-standardized in processed files"),
        ("Time series: medications", "medication.csv", "20 variables", "Pre-standardized in processed files"),
        ("Time series: treatments", "treatment.csv", "14 variables", "Pre-standardized in processed files"),
    ]
    for i, row in enumerate(rows_t3):
        for j, val in enumerate(row):
            t3.rows[i + 1].cells[j].paragraphs[0].add_run(val)
    style_table_professional(t3)

    add_heading_2(doc, "4.2 Multimodal Data Ingestion & Canonical Alignment")
    add_para(doc, (
        "A critical innovation in our framework is the canonical feature alignment engine implemented in "
        "datasets.py. Across the 207 eICU hospitals with both modalities, local feature parquets differ in "
        "their columns because hospitals record different laboratory tests, drugs and treatments. For "
        "example, the processed time-series file of Hospital 102 has 51 columns and that of Hospital 108 "
        "has 53. Feeding these different matrices directly into one shared model would cause parameter "
        "dimension mismatches in federated averaging."
    ))
    add_para(doc, (
        "To resolve this, we established two canonical schema registries: canonical_tabular_features.json "
        "(67 variables) and canonical_timeseries_features.json (74 variables). When EICUTabularDataset and "
        "EICUTimeSeriesDataset instantiate on any hospital directory, the dataset reads the local parquet, "
        "aligns columns against the canonical list in fixed order, fills features that the hospital does "
        "not record and missing values with 0.0, and discards non-canonical columns. Tabular features are "
        "then z-scored with the hospital's own mean and standard deviation, while time-series values are "
        "already standardized in the processed files. Only ICU stays of at least 24 hours are kept, and only "
        "time-series bins from the first 24 hours are used. The two modalities are then joined on the "
        "patient identifier (patientunitstayid), so every training example contains the tabular row, the "
        "time series and the labels of the same ICU stay. Every hospital therefore produces tensors of "
        "identical shapes: (B, 67) for tabular data and (B, 6, 74) for time-series sequences."
    ))

    add_heading_2(doc, "4.3 Modality-Specific Feature Learning")
    add_heading_3(doc, "4.3.1 Tabular MLP Encoder (TabularEncoder)")
    add_para(doc, (
        "The 67-dimensional static vector x_tab is processed through a 3-layer Multi-Layer Perceptron "
        "(MLP). The architecture consists of: Linear(67 \u2192 128) \u2192 BatchNorm \u2192 ReLU \u2192 "
        "Dropout(0.2) \u2192 Linear(128 \u2192 64) \u2192 BatchNorm \u2192 ReLU \u2192 Dropout(0.2) \u2192 "
        "Linear(64 \u2192 64). The encoder compresses baseline clinical variables into a compact latent "
        "embedding "
        "h_tab \u2208 \u211d\u2076\u2074."
    ))

    add_heading_3(doc, "4.3.2 Tabular Autoencoder (Self-Supervised Pre-Training)")
    add_para(doc, (
        "For optional self-supervised pre-training, a symmetric decoder maps h_tab back to \u211d\u2076\u2077 through: "
        "Linear(64 \u2192 64) \u2192 BatchNorm \u2192 ReLU \u2192 Dropout \u2192 Linear(64 \u2192 128) \u2192 "
        "BatchNorm \u2192 ReLU \u2192 Dropout \u2192 Linear(128 \u2192 67), optimized via Mean Squared Error "
        "(MSE) reconstruction loss (scripts/pretrain_tabular_ae.py). The federated experiments in "
        "Chapter 8 do not load these pre-trained weights: all encoders are trained from random "
        "initialization."
    ))

    add_heading_3(doc, "4.3.3 Temporal LSTM Encoder (TimeSeriesLSTM)")
    add_para(doc, (
        "The time-series matrix X_ts \u2208 \u211d^(T \u00d7 74) holds the measurements of the first 24 hours "
        "in 4-hour bins (T \u2264 6). To efficiently process sequences of variable "
        "length without computing gradients over padded zero-tokens, the dataset returns the true unpadded "
        "sequence length L\u1d62. The sequences are packed using PyTorch\u2019s pack_padded_sequence and fed "
        "into a 2-layer Long Short-Term Memory (LSTM) network with hidden dimension 64 and dropout 0.2 "
        "between the two layers. The final valid hidden state of the top LSTM layer is extracted and passed "
        "through a linear projection layer to yield the temporal patient embedding h_ts \u2208 \u211d\u2076\u2074."
    ))

    add_heading_2(doc, "4.4 Multimodal Non-Linear Fusion Layer")
    add_para(doc, (
        "To construct a coherent patient state representation, the learned static embedding h_tab and "
        "temporal embedding h_ts must be synthesized. The MultimodalFusion module executes intermediate "
        "fusion:"
    ))
    add_para(doc, "    h_concat = [h_tab || h_ts] \u2208 \u211d\u00b9\u00b2\u2078",
             align=WD_ALIGN_PARAGRAPH.CENTER, size=11, italic=True, space_after=8)
    add_para(doc, (
        "The concatenated vector is passed through a non-linear smoothing projection block: "
        "Linear(128 \u2192 128) \u2192 BatchNorm1d(128) \u2192 ReLU \u2192 Dropout(0.2) \u2192 "
        "Linear(128 \u2192 128) \u2192 BatchNorm1d(128) \u2192 ReLU. This projection layer lets the model learn "
        "cross-modal interactions (e.g., how age changes the meaning of an abnormal laboratory trend), "
        "yielding a unified patient embedding z\u1d62 \u2208 \u211d\u00b9\u00b2\u2078 for each patient admission i."
    ))

    add_heading_2(doc, "4.5 Intra-Hospital Patient Graph Construction Algorithm")
    add_para(doc, (
        "Unlike standard healthcare models that treat patients as isolated entities, clinical pathology is "
        "profoundly relational. Patients with similar acute phenotypes exhibit correlated mortality risks. "
        "Within each hospital H\u2096, a patient similarity graph G\u2096 = (V\u2096, E\u2096) is built for every "
        "mini-batch of 32 patients, both in training and in evaluation:"
    ))
    add_para(doc, (
        "\u2022 Node Set V\u2096: Each node v\u1d62 \u2208 V\u2096 represents a patient admission i in "
        "hospital H\u2096, with initial node feature vector z\u1d62 \u2208 \u211d\u00b9\u00b2\u2078.\n"
        "\u2022 Edge Construction via k-NN: For each patient i in V\u2096, pairwise cosine similarity scores "
        "are computed against all other patients j in V\u2096 within the batch:\n\n"
        "    S_cos(z\u1d62, z\u2c7c) = (z\u1d62 \u00b7 z\u2c7c) / (\u2016z\u1d62\u2016\u2082 \u00b7 \u2016z\u2c7c\u2016\u2082)\n\n"
        "For every patient i, the k = 5 most similar other patients in the batch are selected, and an edge "
        "is added from each of these neighbours to i. PyTorch Geometric passes messages from edge_index[0] "
        "to edge_index[1], so patient i receives messages from exactly its own k nearest neighbours. The "
        "graph is directed and is not symmetrised. No explicit self-loops are added, because GraphSAGE keeps "
        "a separate term for the node's own features and GAT/GCN add self-loops internally. Because the graph "
        "is rebuilt per batch, a patient's neighbours (and therefore its prediction) depend on which patients "
        "share its batch."
    ))
    add_para(doc, (
        "\u2022 Hospital Boundary: Each mini-batch contains patients of one hospital only, so edges never connect "
        "patients from different hospitals."
    ))

    add_heading_2(doc, "4.6 Graph Representation Learning with PatientGNN")
    add_para(doc, (
        "Once the patient similarity graph G\u2096 is assembled, localized graph representation learning "
        "is executed by PatientGNN. The module supports three prominent GNN architectures:"
    ))
    add_para(doc, (
        "1. GraphSAGE (Sample and Aggregate): For each node v\u1d62, neighborhood aggregation is computed "
        "via mean-pooling with residual skip connections. The final representation combines self-embedding "
        "and aggregated neighbor representations through learned weight matrices.\n\n"
        "2. Graph Attention Networks (GAT): GAT dynamically attends to neighboring patients by calculating "
        "normalized attention coefficients \u03b1\u1d62\u2c7c, weighting neighbors with higher phenotypic "
        "relevance more heavily.\n\n"
        "3. Graph Convolutional Networks (GCN): Computes normalized first-order spectral convolutions; GCN is "
        "the only variant that uses the cosine similarity as an edge weight.\n\n"
        "The experiments in Chapter 8 use GraphSAGE with two layers."
    ))
    add_para(doc, (
        "PatientGNN incorporates residual skip projections around each graph layer and batch normalization. "
        "If a batch has no edges (for example a single patient), the PyTorch Geometric layers simply use "
        "each node's own features. The module transforms "
        "the 128D fused embeddings into 64D graph-enhanced representations g\u1d62 \u2208 \u211d\u2076\u2074."
    ))

    add_heading_2(doc, "4.7 Multi-Task Dual Prediction Heads")
    add_para(doc, (
        "Attached directly to the graph-enhanced embeddings g\u1d62 \u2208 \u211d\u2076\u2074, the "
        "MultiTaskPredictionHeads module performs simultaneous risk stratification for two clinical endpoints:\n\n"
        "\u2022 Task 1 \u2014 In-Hospital Mortality Head: A 2-layer MLP with BatchNorm and Dropout computing "
        "mortality logits: \u0177_mort = W\u2082 \u00b7 ReLU(BatchNorm(W\u2081 \u00b7 g\u1d62)) + b\u2082\n\n"
        "\u2022 Task 2 \u2014 ICU Mortality Head: An identical independent prediction head computing "
        "ICU mortality logits: \u0177_icu = W\u2032\u2082 \u00b7 ReLU(BatchNorm(W\u2032\u2081 \u00b7 g\u1d62)) + b\u2032\u2082\n\n"
        "Both heads output unbounded real-valued logits suitable for numerically stable binary cross-entropy "
        "computation via PyTorch\u2019s BCEWithLogitsLoss."
    ))

    add_heading_2(doc, "4.8 Joint Multi-Task Objective with Imbalance Weighting")
    add_para(doc, (
        "In critical care populations, adverse outcomes are imbalanced: in the 24-hour training cohort, "
        "in-hospital mortality is 8.9% and ICU mortality 5.2%. Standard unweighted cross-entropy loss "
        "causes gradient starvation for the minority positive class. To counteract this, MultiTaskLoss "
        "incorporates positive-class weighting (pos_weight = 3.0):"
    ))
    add_para(doc, (
        "    L_task(\u0177, y) = \u2212[pos_weight \u00b7 y \u00b7 log(\u03c3(\u0177)) + (1\u2212y) \u00b7 log(1\u2212\u03c3(\u0177))]"
    ), align=WD_ALIGN_PARAGRAPH.CENTER, size=11, italic=True, space_after=8)
    add_para(doc, (
        "The overall multi-task training objective is formulated as:\n\n"
        "    L_total = w_mort \u00b7 L_mort + w_icu \u00b7 L_icu\n\n"
        "where w_mort = 1.0 and w_icu = 1.0. Gradients from L_total backpropagate jointly through the "
        "Prediction Heads, PatientGNN, Multimodal Fusion, and both Modality Encoders via end-to-end "
        "gradient descent."
    ))
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 5: PROPOSED ALGORITHMS
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 5\nPROPOSED ALGORITHMS")

    add_heading_2(doc, "5.1 Algorithm 1: Client-Side Local Multimodal Graph Training")
    add_para(doc, (
        "Algorithm 1 details the execution flow carried out privately within each participating hospital "
        "client H\u2096 during communication round t. The hospital receives global model weights W_t from "
        "the server, clones them into a local model instance, constructs its private patient graph, and "
        "computes local updates without transmitting raw clinical records."
    ))

    # Algorithm 1 as a bordered table for professional look
    alg1_lines = [
        "Algorithm 1: Local Client Training at Hospital H_k",
        "\u2500" * 60,
        "Input: Global parameters W_t, local D_tab^(k), D_ts^(k),",
        "       local epochs E, batch size B, learning rate \u03b7",
        "Output: Updated local parameters W_k^(t+1), examples trained n_k",
        "\u2500" * 60,
        " 1: Initialize local model M_k with parameters W_t",
        " 2: Load stays >= 24 h; join tabular and time series by patient ID",
        " 3: Set step=0, n_k=0, optimizer=Adam(M_k, lr=\u03b7, wd=1e-5)",
        " 4: for epoch = 1 to E do",
        " 5:   for batch (x_tab, X_ts, lens, y) in DataLoader(shuffle) do",
        " 6:     if batch_size < 2 then continue  // BatchNorm",
        " 7:     h_tab \u2190 TabularEncoder(x_tab)    // 67D \u2192 64D",
        " 8:     h_ts  \u2190 TimeSeriesLSTM(X_ts, lens) // T\u00d774D \u2192 64D",
        " 9:     z \u2190 MultimodalFusion(h_tab, h_ts) // 128D \u2192 128D",
        "10:     edges \u2190 build_knn_graph(z, k=5)  // neighbour\u2192patient",
        "11:     g \u2190 PatientGNN(z, edges)       // 128D \u2192 64D",
        "12:     \u0177_m, \u0177_i \u2190 MultiTaskHeads(g)",
        "13:     loss \u2190 MultiTaskLoss(\u0177_m, \u0177_i, y, pw=3.0)",
        "14:     optimizer.zero_grad(); loss.backward()",
        "15:     clip_grad_norm_(M_k, max_norm=5.0)",
        "16:     optimizer.step(); step += 1; n_k += |batch|",
        "17:     if step \u2265 S_max (= 30) then break",
        "18: return M_k.state_dict(), n_k  // examples trained",
        "\u2500" * 60,
    ]
    alg1_text = "\n".join(alg1_lines)
    p_alg1 = add_para(doc, alg1_text, align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5, space_after=18)
    p_alg1.paragraph_format.line_spacing = 1.0
    p_alg1.paragraph_format.keep_together = True  # do not split the algorithm across pages

    add_heading_2(doc, "5.2 Algorithm 2: Server-Side Federated Averaging Coordination")
    add_para(doc, (
        "Algorithm 2 details the server-side aggregation protocol. Across each communication round, the "
        "server samples 10 training hospitals, collects their local model weights, and executes "
        "Federated Averaging (FedAvg) weighted by the number of examples each client actually trained on. "
        "Hospitals are processed one after another in this single-machine simulation."
    ))

    alg2_lines = [
        "Algorithm 2: Central Server FedAvg Orchestration",
        "\u2500" * 60,
        "Input: Rounds R=50, clients per round m=10,",
        "       H_train (145), H_val (31); H_test (31) is not used here",
        "Output: Optimal global parameters W*",
        "\u2500" * 60,
        " 1: Initialize W_0 randomly (PyTorch default initialization)",
        " 2: Read the fixed split from hospital_split.json",
        " 3: for round t = 1 to R do",
        " 4:   (m = 10 clients per round)",
        " 5:   S_t \u2190 RandomSample(H_train, m)",
        " 6:   updates \u2190 []",
        " 7:   for each H_k in S_t do  // one after another",
        " 8:     W_k, n_k \u2190 ClientUpdate(H_k, W_{t-1})",
        " 9:     updates.append((W_k, n_k))",
        "10:   N_round \u2190 \u2211 n_k  (examples actually trained)",
        "11:   W_t \u2190 \u2211 (n_k/N_round) \u00b7 W_k  // FedAvg (float tensors)",
        "12:   metrics \u2190 Evaluate(W_t, all 31 H_val hospitals)",
        "13:   if metrics.loss < best_loss then",
        "14:     best_loss \u2190 metrics.loss; SaveCheckpoint(W_t)",
        "15: return the checkpoint with the lowest validation loss",
        "\u2500" * 60,
    ]
    alg2_text = "\n".join(alg2_lines)
    p_alg2 = add_para(doc, alg2_text, align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5, space_after=18)
    p_alg2.paragraph_format.line_spacing = 1.0
    p_alg2.paragraph_format.keep_together = True  # do not split the algorithm across pages
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 6: FLOWCHART, BLOCK DIAGRAM, AND DFD
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 6\nPROPOSED FLOWCHART, BLOCK DIAGRAM, AND DFD")

    add_heading_2(doc, "6.1 System Architectural Block Diagram")
    add_para(doc, (
        "Figure 1 illustrates the high-level system architecture of the Federated Multimodal Graph "
        "Learning framework. The architecture is bifurcated into two primary domains: (i) the Central "
        "Federated Server responsible for parameter distribution, FedAvg aggregation, and validation "
        "monitoring, and (ii) Hospital Clients, each working only on its own hospital's data. "
        "Each hospital client autonomously processes local patient parquets, extracts static and temporal "
        "representations, constructs local patient similarity graphs, and optimizes local model weights. "
        "Only parameter vectors are passed to the server. In this project the clients and the server run "
        "in one program on one machine; the diagram shows the intended deployment."
    ))

    fig1_path = ASSETS_DIR / "fig1_system_architecture.png"
    if fig1_path.exists():
        add_image_centered(doc, fig1_path, width_inches=5.8)
        add_figure_caption(doc, "Figure 1: Federated Multimodal Graph Learning End-to-End System Architecture")

    add_heading_2(doc, "6.2 Data Flow Diagram (DFD Level 0 & Level 1)")
    add_para(doc, (
        "Figure 2 presents the Data Flow Diagram (DFD) delineating the lifecycle of clinical records. "
        "At Level 0 (Context Level), the raw eICU repository is partitioned by hospital and the 207 "
        "hospitals are split into 145 training, 31 validation and 31 test hospitals. At Level 1 (Functional "
        "Pipeline), data from the first 24 hours of each ICU stay is split into Static Tabular (67D) and "
        "Time-Series (6 × 74D) streams. Encoders generate "
        "latent embeddings, the intermediate fusion layer constructs unified 128D representations, the "
        "k-NN graph builder links clinically correlated patients, and PatientGNN computes contextual "
        "graph representations feeding the in-hospital and ICU mortality prediction heads."
    ))

    fig2_path = ASSETS_DIR / "fig2_data_flow_diagram.png"
    if fig2_path.exists():
        add_image_centered(doc, fig2_path, width_inches=5.8)
        add_figure_caption(doc, "Figure 2: Data Flow Diagram (Level 0 Context and Level 1 Functional Pipeline)")

    add_heading_2(doc, "6.3 Intra-Hospital Patient Graph Construction Flowchart")
    add_para(doc, (
        "The algorithmic logic of intra-hospital patient graph assembly proceeds through three sequential "
        "phases:\n\n"
        "Phase 1 \u2014 Pairwise Cosine Similarity Computation: For an active patient batch B\u2096, "
        "normalized dot products S\u1d62\u2c7c = (z\u1d62 \u00b7 z\u2c7c) / (\u2016z\u1d62\u2016 \u00b7 \u2016z\u2c7c\u2016) "
        "are computed across all pairs, forming an affinity matrix A \u2208 \u211d^(|B\u2096| \u00d7 |B\u2096|).\n\n"
        "Phase 2 \u2014 Top-k Neighbor Selection: For each node i, the top k=5 neighbors with the highest "
        "cosine affinity are extracted via torch.topk.\n\n"
        "Phase 3 \u2014 Edge Direction: For each node i, an edge is added from each of its k neighbours to i, "
        "so that i aggregates messages from its own neighbours. The graph is not symmetrised and no "
        "explicit self-loops are added. Figure 6 (Chapter 8) shows a graph built this way by the trained "
        "model on a test hospital."
    ))
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 7: IMPLEMENTATION DETAILS & TECHNOLOGY STACK
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 7\nIMPLEMENTATION DETAILS & TECHNOLOGY STACK")

    add_heading_2(doc, "7.1 Hardware & Software Environments")
    add_para(doc, (
        "The corrected experiments in Chapter 8 were run on a single laptop CPU (no GPU). The following "
        "table lists the hardware and the software versions used:"
    ))

    # Technology stack table
    add_table_caption(doc, "Table 3a: Hardware & Software Environment Specifications")
    t_hw = doc.add_table(rows=10, cols=2)
    hw_data = [
        ("Hardware", "AMD Ryzen 7 7445HS CPU, 16 GB RAM, no GPU used"),
        ("Operating System", "Microsoft Windows 11 Pro"),
        ("Python Version", "Python 3.14.7"),
        ("Deep Learning Framework", "PyTorch 2.14.1 (CPU build)"),
        ("Graph Neural Networks", "PyTorch Geometric (PyG) 2.8.1"),
        ("Data Processing", "Pandas 3.0.5, NumPy 2.5.2, PyArrow 25.0.1"),
        ("ML Evaluation Metrics", "Scikit-Learn 1.9.0"),
        ("Visualization", "Matplotlib 3.11.1, NetworkX 3.7"),
        ("Document Generation", "python-docx 1.2.0, Microsoft Word (PDF export)"),
    ]
    t_hw.rows[0].cells[0].paragraphs[0].add_run("Component").bold = True
    t_hw.rows[0].cells[1].paragraphs[0].add_run("Specification").bold = True
    for i, (comp, spec) in enumerate(hw_data):
        t_hw.rows[i + 1].cells[0].paragraphs[0].add_run(comp)
        t_hw.rows[i + 1].cells[1].paragraphs[0].add_run(spec)
    style_table_professional(t_hw)

    add_heading_2(doc, "7.2 Dataset Specification (eICU Collaborative Database)")
    add_para(doc, (
        "The empirical evaluation utilized the multi-center eICU Collaborative Research Database [25], "
        "comprising over 200,000 ICU stays across 208 hospitals in the United States between 2014 "
        "and 2015. It is distributed through PhysioNet under a credentialed data use agreement; no "
        "patient-level data is reproduced in this report. After preprocessing, 207 hospitals with 200,826 "
        "ICU encounters have both modalities. As illustrated in Figure 3, hospital size is highly skewed: "
        "the largest hospital (Hospital 73) contains 7,059 encounters, the median hospital (Hospital 421) "
        "567, and the smallest only 3. For the early-prediction task, only stays of at least 24 hours are "
        "used: 93,764 encounters in the 145 training hospitals, 21,100 in the 31 validation hospitals and "
        "18,014 in the 31 test hospitals (132,878 in total)."
    ))

    fig3_path = ASSETS_DIR / "fig3_hospital_distribution.png"
    if fig3_path.exists():
        add_image_centered(doc, fig3_path, width_inches=5.8)
        add_figure_caption(doc, "Figure 3: Hospital Patient Population Scale and Distribution across 207 eICU Sites")

    add_heading_2(doc, "7.3 Data Pipeline Implementation (datasets.py)")
    add_para(doc, (
        "The data loading module provides three PyTorch Dataset classes:\n\n"
        "\u2022 EICUTabularDataset: Reads tabular_features.parquet, aligns columns against "
        "canonical_tabular_features.json (67 variables), keeps ICU stays of at least 24 hours, fills "
        "missing values, and fits z-score scalers on the hospital's own data.\n\n"
        "\u2022 EICUTimeSeriesDataset: Reads timeseries_features.parquet, keeps the 4-hour bins of the first "
        "24 hours, groups them by patientunitstayid, aligns against canonical_timeseries_features.json "
        "(74 variables), and pads sequences to 6 steps while keeping the true lengths for packed LSTM "
        "processing.\n\n"
        "\u2022 EICUMultimodalDataset: Joins the two datasets on patientunitstayid so that each example holds "
        "the tabular row, time series and labels of the same ICU stay; it raises an error on duplicate "
        "patient IDs. One DataLoader (one shuffle) is used for both modalities. An earlier version zipped "
        "two separately shuffled loaders, which paired labels with other patients' tabular features; this "
        "was found and fixed before the experiments in Chapter 8."
    ))

    add_heading_2(doc, "7.4 Deep Learning & GNN Implementation Modules")
    add_para(doc, (
        "The neural architectures are encapsulated in modular, maintainable source files organized "
        "within the src/ directory:\n\n"
        "\u2022 src/encoders.py: Implements TabularEncoder, TabularAutoencoder, and TimeSeriesLSTM classes.\n"
        "\u2022 src/fusion.py: Implements MultimodalFusion with batch normalization and non-linear projections.\n"
        "\u2022 src/graph_builder.py: Implements build_knn_graph and build_threshold_graph using vectorized "
        "PyTorch operations.\n"
        "\u2022 src/gnn.py: Implements PatientGNN supporting GraphSAGE, GAT, and GCN layers with residual "
        "connections.\n"
        "\u2022 src/models.py: Implements MultiTaskPredictionHeads, FederatedPatientModel (with a use_gnn switch "
        "for the No-GNN ablation), and MultiTaskLoss.\n"
        "\u2022 src/splits.py and src/evaluation.py: The fixed train / validation / test hospital split and the "
        "single evaluation routine shared by training and benchmarking.\n"
        "\u2022 src/metrics.py: AUROC, AUPRC (average precision), F1, precision, recall, specificity and Brier "
        "score; it raises an error on invalid input instead of returning default values.\n"
        "\u2022 tests/: Data-free unit tests (pytest) for patient alignment, target columns, the hospital split, "
        "graph edge direction, empty graphs, FedAvg weighting and metrics."
    ))

    add_heading_2(doc, "7.5 Federated Training & Distributed Simulation Engine")
    add_para(doc, (
        "The federated training orchestration is implemented in scripts/train_federated.py. It simulates "
        "cross-silo federated learning in a single Python process: each hospital is a client whose data "
        "is only used inside its own local training step. It does not implement networking, secure "
        "aggregation or differential privacy. Key implementation details include:\n\n"
        "\u2022 Hospital sampling: Each round samples 10 of the 145 training hospitals (own random generator, "
        "so runs with the same seed sample the same hospitals).\n"
        "\u2022 Client local training: Each sampled hospital starts from the global weights and trains for one "
        "local epoch, capped at 30 optimizer steps of 32 patients.\n"
        "\u2022 Server aggregation: FedAvg weights each client by the examples it actually trained on.\n"
        "\u2022 Validation: All 31 validation hospitals score the global model after every round; the round "
        "with the lowest validation loss is saved.\n"
        "\u2022 Bookkeeping: Each run writes run_info.json (settings, code version) and training_history.json "
        "(sampled hospitals, optimizer steps, examples and unique patients trained, validation metrics) "
        "to results/corrected/<run>/seed<k>/.\n"
        "\u2022 Baselines: The No-GNN ablation and the local-only baselines use the same script "
        "(--no-gnn, --train-hospitals), and scripts/run_experiments.py runs every configuration for "
        "three seeds."
    ))

    # Table 4
    add_table_caption(doc, "Table 4: Hyperparameter Configuration for Federated Patient Graph Model")
    t4 = doc.add_table(rows=17, cols=3)
    headers4 = ["Hyperparameter", "Value", "Design Rationale"]
    for j, h in enumerate(headers4):
        t4.rows[0].cells[j].paragraphs[0].add_run(h)
    rows_t4 = [
        ("Tabular Latent Dimension", "64", "Balances compression and representation"),
        ("LSTM Hidden Dimension", "64", "Small model for six 4-hour steps"),
        ("Fused Embedding Dimension", "128", "Concatenation of 64D + 64D"),
        ("k-NN Neighbors (k)", "5", "Localized similarity without over-smoothing"),
        ("Graph Metric", "Cosine", "Scale-invariant angular similarity"),
        ("GNN Architecture", "GraphSAGE", "Inductive mean aggregation with residuals"),
        ("Optimizer / Learning Rate", "Adam (\u03b7=1e-3)", "Adaptive gradient, weight decay 1e-5"),
        ("Gradient Clipping", "Max Norm = 5.0", "Prevents exploding gradients in LSTM"),
        ("Loss Positive Weight", "pos_weight = 3.0", "Counters ~9% (hospital) / ~5% (ICU) positive rate"),
        ("Dropout", "0.2", "Regularization in encoders, fusion, GNN and heads"),
        ("Observation Window", "First 24 h (6 \u00d7 4-h bins)", "Input ends before the outcome"),
        ("Communication Rounds", "50", "Reaches ~140 of 145 training hospitals"),
        ("Clients per Round", "10", "Sampled from 145 training hospitals"),
        ("Local Training", "1 epoch, max 30 steps, batch 32", "Same budget cap for every client"),
        ("Checkpoint Selection", "Lowest validation loss", "31 validation hospitals, all rounds"),
        ("Training Seeds", "42, 43, 44", "Hospital split fixed (split seed 42)"),
    ]
    for i, row in enumerate(rows_t4):
        for j, val in enumerate(row):
            t4.rows[i + 1].cells[j].paragraphs[0].add_run(val)
    style_table_professional(t4)
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 8: RESULT DISCUSSION AND COMPARATIVE ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 8\nRESULT DISCUSSION AND COMPARATIVE ANALYSIS")

    add_heading_2(doc, "8.1 Experimental Setup & Evaluation Protocol")
    add_para(doc, (
        "The 207 eICU hospitals were split once, at the hospital level, into 145 training, 31 validation "
        "and 31 test hospitals (processed/hospital_split.json, seed 42). After keeping only ICU stays of at "
        "least 24 hours, this gives 93,764 training, 21,100 validation and 18,014 test encounters. Training "
        "uses the training hospitals; the validation hospitals are used only to pick the best round of each "
        "run; the 31 test hospitals are used only once, for the final benchmark in this chapter, and for "
        "the demonstration graph in Figure 6. The test hospitals are never seen during training, checkpoint "
        "selection or tuning, so the test metrics are out-of-sample at the hospital level. Every configuration "
        "is run with three training seeds (42, 43, 44) with the split held fixed, and results are reported "
        "as mean ± standard deviation over the three seeds."
    ))

    add_heading_2(doc, "8.2 Clinical Evaluation Metrics")
    add_para(doc, (
        "Because the positive outcomes are rare, plain accuracy is misleading, so performance is reported "
        "with the following metrics (src/metrics.py):\n\n"
        "\u2022 AUROC (Area Under the ROC Curve): threshold-free measure of how well the model ranks patients "
        "who died above those who survived. The main metric in this report.\n\n"
        "\u2022 AUPRC (Area Under the Precision-Recall Curve): computed as scikit-learn's average precision; "
        "more informative than AUROC when positives are rare.\n\n"
        "\u2022 Brier score: mean squared error between predicted probability and outcome (lower is better); "
        "a simple measure of calibration.\n\n"
        "\u2022 F1, precision, recall and specificity: reported at a fixed decision threshold of 0.5, which is "
        "not tuned; they are secondary to the threshold-free AUROC and AUPRC.\n\n"
        "AUROC and AUPRC are undefined when only one class is present; the evaluation always had both "
        "classes present, so no such case occurred."
    ))

    add_heading_2(doc, "8.3 Federated Convergence Dynamics across Communication Rounds")
    add_para(doc, (
        "The proposed model was trained for 50 communication rounds, sampling 10 of the 145 training "
        "hospitals per round. As shown in Table 5 and Figure 4 (mean of three seeds), the validation loss "
        "falls quickly from about 1.04 in round 1 to about 0.49 by round 10 and then stays roughly flat, "
        "while the validation in-hospital mortality AUROC rises from about 0.65 to about 0.81. The best "
        "round (lowest validation loss) is used as the checkpoint; across the three seeds the best round "
        "was 43, 32 and 32. Figure 4 also overlays the No-GNN ablation: on the validation hospitals the two "
        "curves almost coincide, so most of the learning comes from the multimodal encoders and FedAvg "
        "rather than from the graph."
    ))

    # Table 5 (loaded from results/corrected/fedavg_gnn)
    conv_rows, n_seeds_conv = gnn_convergence([1, 10, 20, 30, 40, 50])
    add_table_caption(doc, f"Table 5: Validation convergence of the proposed FedAvg + GNN model (mean of {n_seeds_conv} seeds, 31 validation hospitals)")
    headers5 = ["Round", "Val. Loss", "Mort. AUROC", "Mort. AUPRC", "ICU AUROC"]
    t5 = doc.add_table(rows=len(conv_rows) + 1, cols=len(headers5))
    for j, h in enumerate(headers5):
        t5.rows[0].cells[j].paragraphs[0].add_run(h)
    for i, row in enumerate(conv_rows):
        for j, val in enumerate(row):
            t5.rows[i + 1].cells[j].paragraphs[0].add_run(val)
    style_table_professional(t5)

    fig4_path = ASSETS_DIR / "fig4_training_convergence.png"
    if fig4_path.exists():
        add_image_centered(doc, fig4_path, width_inches=5.8)
        add_figure_caption(doc, "Figure 4: Validation loss and in-hospital mortality AUROC per round (mean ± SD over 3 seeds), FedAvg + GNN vs No-GNN")

    add_heading_2(doc, "8.4 Comparative Baseline Evaluation")
    add_para(doc, (
        "To separate the benefit of federated collaboration from the benefit of the graph, five "
        "configurations were scored on the 31 held-out test hospitals (scripts/benchmark_baselines.py), "
        "each with three seeds:\n\n"
        "\u2022 FedAvg + GNN (proposed): the full model, trained across the 145 training hospitals.\n\n"
        "\u2022 FedAvg No-GNN (ablation): the same training loop and the same sampled clients each round, but "
        "with the patient graph and GNN removed (the heads read the fused embedding directly). This "
        "isolates the effect of the graph.\n\n"
        "\u2022 Local-only (three hospitals): the full model trained on a single training hospital, for three "
        "hospital sizes \u2014 hospital 183 (1,562 patients), hospital 396 (458) and hospital 61 (87). This "
        "shows what one hospital can achieve alone, and how it depends on the hospital's size.\n\n"
        "Every model is selected on the validation hospitals by lowest validation loss and evaluated on "
        "the same 18,014 test patients. The local-only models train on much less data than the federated "
        "models; this is the point of the comparison, and Table 6 should be read with that in mind."
    ))

    # Table 6 (loaded from results/corrected/benchmark_results.json)
    summary = load_benchmark_summary()
    model_order = [
        ("fedavg_gnn", "FedAvg + GNN (proposed)"),
        ("fedavg_nognn", "FedAvg No-GNN (ablation)"),
        ("local_183", "Local-only (hosp. 183)"),
        ("local_396", "Local-only (hosp. 396)"),
        ("local_61", "Local-only (hosp. 61)"),
    ]
    add_table_caption(doc, f"Table 6: Test-hospital results (mean ± SD over {summary['fedavg_gnn']['n_seeds']} seeds; 31 held-out hospitals, 18,014 patients)")
    headers6 = ["Model", "Mort. AUROC", "Mort. AUPRC", "ICU AUROC", "ICU AUPRC", "Mort. Brier"]
    keys6 = ["mort_auroc", "mort_auprc", "icu_auroc", "icu_auprc", "mort_brier"]
    t6 = doc.add_table(rows=len(model_order) + 1, cols=len(headers6))
    for j, h in enumerate(headers6):
        t6.rows[0].cells[j].paragraphs[0].add_run(h)

    def _fmt6(s):
        return f"{s['mean']:.3f} ± {s['sd']:.3f}" if s['sd'] is not None else f"{s['mean']:.3f}"
    for i, (key, label) in enumerate(model_order):
        s = summary[key]
        cells = [label] + [_fmt6(s[k]) for k in keys6]
        for j, val in enumerate(cells):
            r = t6.rows[i + 1].cells[j].paragraphs[0].add_run(val)
            if i == 0:
                r.bold = True
    style_table_professional(t6)

    fig5_path = ASSETS_DIR / "fig5_baseline_comparisons.png"
    if fig5_path.exists():
        add_image_centered(doc, fig5_path, width_inches=5.6)
        add_figure_caption(doc, "Figure 5: Test-hospital performance (mean ± SD over 3 seeds): FedAvg + GNN vs No-GNN vs Local-only")

    add_para(doc, (
        "Key Findings from the Empirical Analysis:"
    ), bold=True, size=12, space_after=8)
    add_para(doc, (
        "Finding 1 \u2014 Federated training helps, and more so for small hospitals: Both federated models "
        "clearly beat all three local-only models on the test hospitals (in-hospital mortality AUROC "
        "0.819 and 0.806, versus 0.769, 0.660 and 0.583 for local-only on hospitals of 1,562, 458 and 87 "
        "patients). The local-only result drops sharply as the single hospital gets smaller, while the "
        "federated models, which learn from 145 hospitals, stay strong. This is the clearest result of "
        "the study."
    ))
    add_para(doc, (
        "Finding 2 \u2014 The graph gives a small, consistent gain: Adding the patient graph improves the "
        "mean in-hospital mortality AUROC from 0.806 to 0.819 and the ICU mortality AUROC from 0.842 to "
        "0.853. The gain is small and the standard-deviation bands overlap, but the GNN was at least as "
        "good as the No-GNN model in every one of the three seeds and had lower variance, so on this task "
        "the graph helps modestly rather than dramatically. We do not claim a large improvement from the "
        "graph. Both federated models share the same training budget (same sampled clients, 6,945 optimizer "
        "steps), so this comparison is like-for-like."
    ))

    add_heading_2(doc, "8.5 Patient Graph Visualization")
    add_para(doc, (
        "Figure 6 shows the patient graph that the trained model actually builds on one held-out test "
        "hospital (hospital 120, 68 patients with at least a 24-hour stay). Each node is coloured by the "
        "model's predicted in-hospital mortality probability (green = low, red = high), and patients who "
        "were recorded as dying in hospital are drawn with a dark red ring. Because the graph is built per "
        "mini-batch of 32 patients, the nodes fall into a few dense clusters that correspond to the "
        "batches. Within these clusters, most patients have low predicted risk, a few have higher "
        "predicted risk, and the recorded deaths are a mix: some are predicted high-risk and some are not, "
        "which is consistent with the moderate AUROC and low AUPRC in Table 6. The figure is a qualitative "
        "illustration of the model output, not evidence of clean phenotype separation."
    ))

    fig6_path = ASSETS_DIR / "fig6_patient_graph_topology.png"
    if fig6_path.exists():
        add_image_centered(doc, fig6_path, width_inches=5.2)
        add_figure_caption(doc, "Figure 6: Patient similarity graph built by the trained model on test hospital 120 (node colour = predicted in-hospital mortality; red ring = recorded death)")
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 9: CONCLUSION AND FUTURE SCOPE
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "CHAPTER 9\nCONCLUSION AND FUTURE SCOPE")

    add_heading_2(doc, "9.1 Summary of Contributions")
    add_para(doc, (
        "This project designed and implemented a Federated Multimodal Graph Learning prototype for early "
        "In-Hospital and ICU Mortality prediction, and evaluated it in a single-machine federated "
        "simulation on the eICU Collaborative Research Database (207 hospitals). The main contributions "
        "are summarized as follows:"
    ))
    add_para(doc, (
        "Contribution 1 \u2014 Multimodal Architecture: We implemented a tabular MLP encoder and a time-series "
        "LSTM encoder (with packed padded sequences) that project static variables (67D) and the first "
        "24 hours of time series (74D, six 4-hour bins) into a shared 128D space through a non-linear "
        "fusion layer, with the two modalities joined per patient by patient ID."
    ))
    add_para(doc, (
        "Contribution 2 \u2014 Intra-Hospital Patient Graphs: We implemented a k-nearest neighbour patient "
        "similarity graph built per mini-batch within a single hospital, so edges never connect patients "
        "from different hospitals. In a real deployment this keeps graph construction local; in this "
        "project the hospitals are simulated as separate clients in one process."
    ))
    add_para(doc, (
        "Contribution 3 \u2014 Graph Neural Network: We integrated PatientGNN (GraphSAGE, GAT and GCN) with "
        "residual connections to pass information between similar patients; the experiments use GraphSAGE. "
        "The edge direction was corrected so that each patient aggregates messages from its own nearest "
        "neighbours, and batches without edges fall back to each node's own features."
    ))
    add_para(doc, (
        "Contribution 4 \u2014 Federated Multi-Task Training and Honest Evaluation: We implemented an "
        "end-to-end multi-task objective with positive-class weighting and FedAvg aggregation weighted by "
        "the examples each client trained on. All models (federated GNN, No-GNN ablation and local-only "
        "baselines) are trained with the same loop and the same checkpoint rule, and compared on 31 "
        "held-out test hospitals never used in training or model selection (Chapter 8)."
    ))

    add_heading_2(doc, "9.2 Limitations of the Current Study")
    add_para(doc, (
        "Several limitations should be kept in mind when reading the results:\n\n"
        "\u2022 Single-Machine Simulation: The federated setup runs in one Python process on one CPU. It does "
        "not implement networking, secure aggregation or differential privacy, so no formal privacy "
        "guarantee is claimed. The results show feasibility, not clinical validation.\n\n"
        "\u2022 Batch-Level Graph: The patient graph is rebuilt per mini-batch of 32 patients rather than over "
        "the whole hospital, so a patient's neighbours (and prediction) depend on which patients share its "
        "batch. It is also static, not evolving over time within the 24-hour window.\n\n"
        "\u2022 Diagnosis Feature Timing: The tabular diagnosis features are derived from the stay's diagnosis "
        "table without a strict 24-hour cut-off. On a sample hospital they matched the first-24-hour "
        "diagnoses about 99.6% of the time, so the effect is small, but a small amount of later information "
        "may leak into these features.\n\n"
        "\u2022 Simple Aggregation: The server uses standard FedAvg. Personalization methods (e.g., FedProx, "
        "Ditto) could handle the large differences in hospital size better.\n\n"
        "\u2022 Excluded Modalities: Unstructured clinical notes are not used; the model uses only the tabular "
        "and time-series features in the processed eICU files.\n\n"
        "\u2022 Threshold-Based Metrics: F1, precision and recall use a fixed 0.5 threshold, which is not tuned; "
        "AUROC and AUPRC (threshold-free) are the main metrics."
    ))

    add_heading_2(doc, "9.3 Future Research Directions")
    add_para(doc, (
        "Future enhancements to this framework can proceed along several promising research trajectories:\n\n"
        "1. Dynamic Temporal Patient Graphs: Construct continuous-time dynamic graphs where patient nodes "
        "and similarity edges evolve as new time-series measurements arrive, capturing temporal evolution "
        "of clinical phenotypes.\n\n"
        "2. Differential Privacy & Secure Enclaves: Integrate formal Differential Privacy (DP-SGD) with "
        "gradient perturbation or homomorphic encryption to defend against model inversion and reconstruction "
        "attacks on shared parameters.\n\n"
        "3. Attention-Based Cross-Modal Fusion: Replace intermediate concatenation fusion with multi-head "
        "cross-attention mechanisms enabling bidirectional conditioning between the static and time-series streams, "
        "potentially capturing more nuanced cross-modal clinical interactions.\n\n"
        "4. Genuine Sepsis Label: Add a clinically defined sepsis label (e.g., from the diagnosis and "
        "microbiology tables with a fixed time horizon), rebuilding the infection-related input features to "
        "avoid leakage, so that a sepsis task can be reported honestly alongside mortality.\n\n"
        "5. Model Explainability: Apply methods such as GNNExplainer or Integrated Gradients to explain "
        "individual risk predictions.\n\n"
        "6. Personalized Federated Learning: Explore per-hospital personalization (e.g., FedProx, Ditto, "
        "or meta-learning) to better handle the large differences in hospital size and case mix."
    ))
    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # REFERENCES (IEEE FORMAT - 25 REFERENCES)
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "REFERENCES")
    add_para(doc, "All references follow the IEEE citation standard as required by IIIT Bhopal project guidelines.",
             size=10, italic=True, space_after=14)

    references = [
        '[1] W. A. Knaus, E. A. Draper, D. P. Wagner, and J. E. Zimmerman, "APACHE II: a severity of disease classification system," Critical Care Medicine, vol. 13, no. 10, pp. 818\u2013829, Oct. 1985.',
        '[2] J. R. Le Gall, S. Lemeshow, and F. Saulnier, "A new Simplified Acute Physiology Score (SAPS II) based on a European/North American multicenter study," JAMA, vol. 270, no. 24, pp. 2957\u20132963, Dec. 1993.',
        '[3] J. L. Vincent, R. Moreno, J. Takala, et al., "The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure," Intensive Care Medicine, vol. 22, no. 7, pp. 707\u2013710, Jul. 1996.',
        '[4] M. Singer, C. S. Deutschman, C. W. Seymour, et al., "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)," JAMA, vol. 315, no. 8, pp. 801\u2013810, Feb. 2016.',
        '[5] A. Rajkomar, E. Oren, K. Chen, et al., "Scalable and accurate deep learning with electronic health records," npj Digital Medicine, vol. 1, no. 18, pp. 1\u201310, May 2018.',
        '[6] E. Choi, M. T. Bahadori, A. Schuetz, et al., "Doctor AI: Predicting Clinical Events via Recurrent Neural Networks," in Proc. JMLR Workshop and Conference, vol. 56, pp. 301\u2013318, 2016.',
        '[7] Z. C. Lipton, D. C. Kale, C. Elkan, and R. Wetzel, "Learning to Diagnose with LSTM Recurrent Neural Networks," in Proc. ICLR, 2016.',
        '[8] A. Vaswani, N. Shazeer, N. Parmar, et al., "Attention is all you need," in Advances in NeurIPS, vol. 30, pp. 5998\u20136008, 2017.',
        '[9] L. Rasmy, Y. Xiang, Z. Xie, C. Tao, and D. Zhi, "Med-BERT: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction," npj Digital Medicine, vol. 4, no. 86, May 2021.',
        '[10] Y. Li, S. Rao, J. R. A. Solares, et al., "BEHRT: Transformer for Electronic Health Records," Scientific Reports, vol. 10, no. 7155, pp. 1\u201312, Apr. 2020.',
        '[11] T. Baltru\u0161aitis, C. Ahuja, and L. P. Morency, "Multimodal Machine Learning: A Survey and Taxonomy," IEEE Trans. PAMI, vol. 41, no. 2, pp. 423\u2013443, Feb. 2019.',
        '[12] S. C. Huang, A. Pareek, S. Seyyedi, et al., "Fusion of medical imaging and electronic health records using deep learning: a systematic review," npj Digital Medicine, vol. 3, no. 136, pp. 1\u201314, Oct. 2020.',
        '[13] B. Bardak and M. Tan, "Improving clinical outcome predictions using convolution over medical entities with multimodal learning," Artificial Intelligence in Medicine, vol. 117, art. no. 102112, 2021.',
        '[14] Z. Wu, S. Pan, F. Chen, et al., "A Comprehensive Survey on Graph Neural Networks," IEEE Trans. NNLS, vol. 32, no. 1, pp. 4\u201324, Jan. 2021.',
        '[15] S. Pai and G. D. Bader, "Patient similarity networks for precision medicine," Journal of Molecular Biology, vol. 430, no. 18, pp. 2924\u20132938, 2018.',
        '[16] T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," in Proc. ICLR, 2017.',
        '[17] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive Representation Learning on Large Graphs," in Advances in NeurIPS, vol. 30, pp. 1024\u20131034, 2017.',
        '[18] P. Veli\u010dkovi\u0107, G. Cucurull, A. Casanova, et al., "Graph Attention Networks," in Proc. ICLR, 2018.',
        '[19] E. Rocheteau, C. Tong, P. Veli\u010dkovi\u0107, N. Lane, and P. Li\u00f2, "Predicting patient outcomes with graph representation learning," arXiv:2101.03940, 2021.',
        '[20] B. McMahan, E. Moore, D. Ramage, et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," in Proc. AISTATS, pp. 1273\u20131282, 2017.',
        '[21] P. Kairouz, H. B. McMahan, B. Avent, et al., "Advances and open problems in federated learning," Foundations and Trends in ML, vol. 14, no. 1\u20132, pp. 1\u2013210, 2021.',
        '[22] N. Rieke, J. Hancox, W. Li, et al., "The future of digital health with federated learning," npj Digital Medicine, vol. 3, no. 119, pp. 1\u20137, Sep. 2020.',
        '[23] M. J. Sheller, B. Edwards, G. A. Reina, et al., "Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data," Scientific Reports, vol. 10, no. 12598, Jul. 2020.',
        '[24] I. Dayan, H. R. Roth, A. Zhong, et al., "Federated learning for predicting clinical outcomes in patients with COVID-19," Nature Medicine, vol. 27, no. 10, pp. 1735\u20131743, Oct. 2021.',
        '[25] T. J. Pollard, A. E. Johnson, J. D. Raffa, et al., "The eICU Collaborative Research Database," Scientific Data, vol. 5, no. 180178, pp. 1\u201313, Sep. 2018.',
    ]

    for ref in references:
        p_ref = add_para(doc, ref, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=10, space_after=4)
        p_ref.paragraph_format.line_spacing = 1.0
        p_ref.paragraph_format.left_indent = Inches(0.3)
        p_ref.paragraph_format.first_line_indent = Inches(-0.3)

    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # LIST OF PUBLICATIONS
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "LIST OF PUBLICATIONS")
    # The guideline asks only for accepted / submitted / published papers ("If Any")
    add_para(doc, (
        "Not applicable. No paper based on this project work has been submitted, accepted, or "
        "published at the time of this report."
    ), size=12, space_after=24)

    page_break(doc)

    # ═════════════════════════════════════════════════════════════════════════
    # PLAGIARISM & AUTHENTICITY DECLARATION
    # ═════════════════════════════════════════════════════════════════════════
    add_heading_1(doc, "PLAGIARISM & AUTHENTICITY DECLARATION")

    add_para(doc, "Authenticity Declaration",
             size=12, italic=True, bold=True, space_after=14,
             align=WD_ALIGN_PARAGRAPH.CENTER)

    add_para(doc, (
        "This project report has been prepared in rigorous compliance with the anti-plagiarism guidelines "
        "established by the Department of Computer Science and Engineering, Indian Institute of Information "
        "Technology Bhopal. The following declarations are made:"
    ), space_after=12)

    add_para(doc, (
        "\u2022 Similarity Index: The similarity score is the one given by the institute's official Turnitin "
        "report for this document; no score is claimed here.\n\n"
        "\u2022 Figures: Figures 1 and 2 are diagrams drawn with matplotlib. Figures 3 to 6 are generated by "
        "scripts/generate_report_figures.py from the processed eICU data and from the saved experiment "
        "outputs in results/corrected/ (Figure 6 shows the graph built by the trained model, with "
        "anonymous patient labels).\n\n"
        "\u2022 Code Authenticity: All source code modules (datasets.py, encoders.py, fusion.py, "
        "graph_builder.py, gnn.py, models.py, train_federated.py, benchmark_baselines.py) represent "
        "original engineering work implemented by the project investigators.\n\n"
        "\u2022 Data Attribution: The eICU Collaborative Research Database [25] is available to credentialed "
        "researchers through PhysioNet under a data use agreement. No patient-level records or patient "
        "identifiers are reproduced in this report.\n\n"
        "\u2022 AI Assistance: An AI coding assistant (Claude, Anthropic) was used to audit the code, correct "
        "errors and revise this report. All reported results come from running the code in the project "
        "repository.\n\n"
        "\u2022 Overall Authenticity: The text, code implementations, mathematical formulations, and "
        "experimental results presented in this document represent bonafide academic and engineering "
        "contributions carried out by the project investigators, with the AI assistance stated above."
    ), space_after=36)

    # Signature table
    t_sign = doc.add_table(rows=2, cols=2)
    t_sign.alignment = WD_TABLE_ALIGNMENT.CENTER

    for col_idx, (name, scholar) in enumerate([
        ("Dhruv Savaliya", "21U02035"),
        ("Menil Patel", "21U02052")
    ]):
        sig_cell = t_sign.rows[0].cells[col_idx].paragraphs[0]
        sig_run = sig_cell.add_run("\n\n\n_________________________")
        sig_run.font.name = "Times New Roman"
        sig_run.font.size = Pt(11)

        name_cell = t_sign.rows[1].cells[col_idx].paragraphs[0]
        name_run = name_cell.add_run(f"Student Signature\n({name} \u2013 {scholar})")
        name_run.font.name = "Times New Roman"
        name_run.font.size = Pt(11)
        name_run.bold = True

    # ── Save Word Document ───────────────────────────────────────────────────
    doc.save(str(DOCX_OUT_PATH))
    print(f"\n{'='*70}")
    print(f"  REPORT GENERATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"  Word Document: {DOCX_OUT_PATH}")
    print(f"  Total Sections: 9 Chapters + Front Matter + References")
    print(f"  Tables: 8 professional tables with borders")
    print(f"  Figures: 6 publication-quality illustrations")
    print(f"  References: 25 (IEEE Format)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    build_report()
