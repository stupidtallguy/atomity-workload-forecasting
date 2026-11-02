"""Render the editable Markdown report as a six-page PDF.

Authoring-only dependencies: ReportLab and Pillow. This script does not read
raw data, fit models, regenerate forecasts, or change the frozen method.
"""
from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def register_fonts():
    directory = Path("C:/Windows/Fonts")
    if (directory / "arial.ttf").exists():
        for name, filename in [("ReportSans", "arial.ttf"), ("ReportSansBold", "arialbd.ttf"),
                               ("ReportSansItalic", "ariali.ttf"), ("ReportMono", "consola.ttf")]:
            pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
        pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSansBold",
                                      italic="ReportSansItalic", boldItalic="ReportSansBold")
        return "ReportSans", "ReportSansBold", "ReportSansItalic", "ReportMono"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier"


def build():
    normal, bold, italic, mono = register_fonts()
    width = A4[0] - 96
    styles = {
        "body": ParagraphStyle("Body", fontName=normal, fontSize=10.4, leading=14,
                                textColor=colors.black, spaceAfter=8),
        "title": ParagraphStyle("Title", fontName=bold, fontSize=23, leading=27,
                                 textColor=colors.black, spaceAfter=10),
        "heading": ParagraphStyle("Heading", fontName=bold, fontSize=14, leading=18,
                                   textColor=colors.black, spaceBefore=11, spaceAfter=7, keepWithNext=True),
        "caption": ParagraphStyle("Caption", fontName=italic, fontSize=8.6, leading=11.5,
                                   textColor=colors.black, spaceAfter=9),
        "code": ParagraphStyle("Code", fontName=mono, fontSize=8.1, leading=11.5,
                                textColor=colors.black, spaceAfter=1),
        "cell": ParagraphStyle("Cell", fontName=normal, fontSize=9.2, leading=12),
        "cell_right": ParagraphStyle("CellRight", fontName=normal, fontSize=9.2, leading=12, alignment=TA_RIGHT),
        "header": ParagraphStyle("Header", fontName=bold, fontSize=9.2, leading=12, textColor=colors.white),
        "header_right": ParagraphStyle("HeaderRight", fontName=bold, fontSize=9.2, leading=12,
                                        textColor=colors.white, alignment=TA_RIGHT),
    }

    def inline(text):
        result = escape(text)
        result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#000000">\1</link>', result)
        result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", result)
        result = re.sub(r"`([^`]+)`", lambda m: f'<font name="{mono}" size="9.2">{m[1]}</font>', result)
        return result

    def table(lines):
        rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
        right = [cell.endswith(":") for cell in rows[1]]
        values = [rows[0], *rows[2:]]
        first = rows[0][0]
        if first == "Workload category":
            widths = [220, 90, width - 310]
        elif first == "Fold":
            widths = [38, 132, 95, width - 265]
        elif first == "Method":
            widths = [width - 224, 72, 72, 80]
        elif first == "CPU method":
            widths = [width - 255, 88, 88, 79]
        else:
            widths = [115, 152, 104, width - 371]
        cells = []
        for row_index, row in enumerate(values):
            cells.append([
                Paragraph(inline(value), styles[("header_right" if right[i] else "header")
                                                if row_index == 0 else ("cell_right" if right[i] else "cell")])
                for i, value in enumerate(row)
            ])
        block = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243d4b")),
            ("GRID", (0, 0), (-1, -1), .45, colors.HexColor("#D9D9D9")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        for row in range(1, len(cells)):
            commands.append(("BACKGROUND", (0, row), (-1, row),
                             colors.white if row % 2 else colors.HexColor("#F2F5F7")))
        block.setStyle(TableStyle(commands))
        return [Spacer(1, 3), block, Spacer(1, 9)]

    text = (REPORTS / "final_report.md").read_text(encoding="utf-8")
    story = []
    for page_number, page in enumerate(text.split("<!-- pagebreak -->")):
        if page_number:
            story.append(PageBreak())
        lines = page.strip().splitlines()
        cursor = 0
        while cursor < len(lines):
            line = lines[cursor].strip()
            if not line:
                cursor += 1
                continue
            if line.startswith("# "):
                story.append(Paragraph(inline(line[2:]), styles["title"]))
                cursor += 1
            elif line.startswith("## "):
                story.append(Paragraph(inline(line[3:]), styles["heading"]))
                cursor += 1
            elif line.startswith("|"):
                block = []
                while cursor < len(lines) and lines[cursor].strip().startswith("|"):
                    block.append(lines[cursor])
                    cursor += 1
                story.extend(table(block))
            elif line.startswith("```"):
                cursor += 1
                code = []
                while cursor < len(lines) and not lines[cursor].strip().startswith("```"):
                    code.append(Paragraph(escape(lines[cursor]), styles["code"]))
                    cursor += 1
                story.append(KeepTogether([Spacer(1, 3), *code, Spacer(1, 8)]))
                cursor += 1
            elif line.startswith("!["):
                match = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", line)
                path = REPORTS / match[2]
                with PILImage.open(path) as image:
                    pixel_width, pixel_height = image.size
                image = Image(str(path), width=width, height=width * pixel_height / pixel_width)
                image.hAlign = "LEFT"
                image._altText = match[1]
                cursor += 1
                while cursor < len(lines) and not lines[cursor].strip():
                    cursor += 1
                caption = []
                if cursor < len(lines) and lines[cursor].strip().startswith("*Figure"):
                    caption = [Paragraph(inline(lines[cursor].strip().strip("*")), styles["caption"])]
                    cursor += 1
                story.append(KeepTogether([Spacer(1, 2), image, Spacer(1, 4), *caption]))
            else:
                paragraph = []
                while cursor < len(lines) and lines[cursor].strip():
                    paragraph.append(lines[cursor])
                    cursor += 1
                content = ""
                for index, value in enumerate(paragraph):
                    content += inline(value.rstrip())
                    if index < len(paragraph) - 1:
                        content += "<br/>" if value.endswith("  ") else " "
                if content.startswith("*") and content.endswith("*"):
                    story.append(Paragraph(content.strip("*"), styles["caption"]))
                else:
                    story.append(Paragraph(content, styles["body"]))

    def page_decoration(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.black)
        canvas.setFont(normal, 8)
        if document.page > 1:
            canvas.drawString(48, A4[1] - 27, "Cloud Workload Forecasting and Analysis")
        canvas.drawString(48, 24, "Salar Rahnama")
        canvas.drawRightString(A4[0] - 48, 24, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(REPORTS / "final_report.pdf"), pagesize=A4, rightMargin=48, leftMargin=48,
        topMargin=44, bottomMargin=43, title="Cloud Workload Forecasting and Analysis",
        author="Salar Rahnama", subject="Atomity forecasting challenge final report",
    )
    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    print(REPORTS / "final_report.pdf")


if __name__ == "__main__":
    build()
