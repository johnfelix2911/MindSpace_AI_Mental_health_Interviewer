"""
PDF reporting helpers for interview results.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import re
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FIELD_LABELS = {
    "name": "Name",
    "age": "Age",
    "gender": "Gender",
    "country": "Country",
    "role": "Role",
    "stage": "Stage",
    "focus": "Focus Area",
    "sleep_duration": "Sleep Duration",
    "workload": "Workload",
    "screen_time": "Screen Time",
    "living_situation": "Living Situation",
    "support_system": "Support System",
    "stressors": "Stressors",
}


SEVERITY_LEGENDS = {
    "depression": [
        ("0-4", "No significant depressive symptoms"),
        ("5-9", "Mild depressive symptoms"),
        ("10-14", "Moderate depressive symptoms"),
        ("15-19", "Moderately severe depressive symptoms"),
        ("20-24", "Severe depressive symptoms"),
    ],
    "anxiety": [
        ("0-4", "Minimal anxiety symptoms"),
        ("5-9", "Mild anxiety symptoms"),
        ("10-14", "Moderate anxiety symptoms"),
        ("15-24", "Severe anxiety symptoms"),
    ],
    "stress": [
        ("0-29%", "Minimal stress load"),
        ("30-49%", "Mild stress load"),
        ("50-69%", "Moderate stress load"),
        ("70-100%", "Severe stress load"),
    ],
}


def _fmt_assessment(value: str) -> str:
    if value == "all":
        return "Complete assessment"
    return str(value or "").replace("_", " ").title()


def _fmt_label(value: str) -> str:
    return str(value or "").replace("_", " ").title()


def _fmt_score(key: str, value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if key == "stress":
        return f"{value * 100:.0f}%"
    return f"{value:.1f}"


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value).strip()


def _report_value(value: Any) -> str:
    normalized = _normalize_value(value)
    return normalized if normalized else "N/A"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "participant"


def build_report_filename(demographics: dict[str, Any]) -> str:
    name = _normalize_value(demographics.get("name")) or "participant"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"mindspace-report-{_slugify(name)}-{stamp}.pdf"


def build_pdf_report(
    results: dict[str, Any],
    demographics: dict[str, Any],
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#10263e"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1b4965"),
        spaceAfter=6,
        spaceBefore=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#22313f"),
    )
    muted_style = ParagraphStyle(
        "Muted",
        parent=body_style,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#5b6b7b"),
    )

    story = [
        Paragraph("MindSpace Mental Health Screening Report", title_style),
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            muted_style,
        ),
        Spacer(1, 10),
    ]

    partial_warning = _normalize_value(results.get("partial_warning"))
    if partial_warning:
        story.append(Paragraph("Data Quality Warning", section_style))
        warning_table = Table([[partial_warning]], colWidths=[165 * mm])
        warning_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4e5")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#7c4a03")),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#f2b04c")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("LEADING", (0, 0), (-1, -1), 12),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.extend([warning_table, Spacer(1, 8)])

    participant_rows = [["Field", "Value"]]
    for key in (
        "name",
        "age",
        "gender",
        "country",
        "role",
        "stage",
        "focus",
        "sleep_duration",
        "workload",
        "screen_time",
        "living_situation",
        "support_system",
        "stressors",
    ):
        participant_rows.append([FIELD_LABELS[key], _report_value(demographics.get(key))])

    story.append(Paragraph("Participant Details", section_style))
    participant_table = Table(participant_rows, colWidths=[45 * mm, 120 * mm])
    participant_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10263e")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d6e5")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([participant_table, Spacer(1, 10)])

    story.append(Paragraph("Assessment Summary", section_style))
    summary_rows = [["Assessment", "Score", "Severity"]]
    for key in ("depression", "anxiety", "stress"):
        if key in results.get("scores", {}):
            summary_rows.append(
                [
                    _fmt_assessment(key),
                    _fmt_score(key, results["scores"].get(key)),
                    _fmt_label(results.get("labels", {}).get(key, "")),
                ]
            )
    summary_table = Table(summary_rows, colWidths=[55 * mm, 35 * mm, 75 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d6e5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 10)])

    for key in ("depression", "anxiety", "stress"):
        if key not in results.get("scores", {}):
            continue
        story.append(Paragraph(f"{_fmt_assessment(key)} Severity Legend", section_style))
        legend_rows = [["Range", "Interpretation"]]
        legend_rows.extend(SEVERITY_LEGENDS[key])
        legend_table = Table(legend_rows, colWidths=[35 * mm, 130 * mm])
        legend_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f1fa")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4dee8")),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.extend([legend_table, Spacer(1, 8)])

    recommendation = results.get("recommendation", {}) or {}
    summary_text = _normalize_value(recommendation.get("summary"))
    if summary_text:
        story.append(Paragraph("Recommendation Summary", section_style))
        story.extend([Paragraph(summary_text, body_style), Spacer(1, 8)])

    actions = [str(item).strip() for item in recommendation.get("recommendations", []) if str(item).strip()]
    if actions:
        story.append(Paragraph("Suggested Actions", section_style))
        story.append(
            ListFlowable(
                [ListItem(Paragraph(item, body_style)) for item in actions],
                bulletType="bullet",
                start="circle",
                leftIndent=16,
            )
        )
        story.append(Spacer(1, 8))

    resources = [str(item).strip() for item in recommendation.get("resources", []) if str(item).strip()]
    if resources:
        story.append(Paragraph("Helpful Resources", section_style))
        story.append(
            ListFlowable(
                [ListItem(Paragraph(item, body_style)) for item in resources],
                bulletType="bullet",
                start="circle",
                leftIndent=16,
            )
        )
        story.append(Spacer(1, 8))

    encouragement = _normalize_value(recommendation.get("encouragement"))
    if encouragement:
        story.append(Paragraph("Encouragement", section_style))
        story.extend([Paragraph(encouragement, body_style), Spacer(1, 8)])

    story.append(Paragraph("Important Note", section_style))
    story.append(
        Paragraph(
            "This screening report is informational only and does not replace diagnosis, treatment, or emergency support from a licensed professional.",
            body_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()
