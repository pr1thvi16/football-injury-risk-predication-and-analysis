"""Downloadable prediction reports (CSV and a simple PDF)."""
from __future__ import annotations

import io

import pandas as pd

from src.config import RESEARCH_DISCLAIMER

LIMITATIONS = [
    "Trained and evaluated only on SYNTHETIC data; no real-world validity has been established.",
    "Learned feature associations reflect the simulation mechanism, not real causal injury mechanisms.",
    "Probability is a research output, not a diagnosis, clinical risk score or return-to-play decision.",
    "Risk categories use illustrative, configurable cut-offs that are not clinically validated.",
    "Feature contributions (SHAP / Integrated Gradients) describe model behaviour, not why a real injury occurs.",
]


def build_report(*, player_id, observation_date, model_name, probability, threshold, risk_category, predicted_label,
                 input_summary: dict, contributions: pd.DataFrame | None, explanation_method: str | None, hypothetical: bool = False) -> dict:
    return {
        "title": "Injury-risk prediction report (research prototype, SYNTHETIC data)",
        "player_id": player_id, "observation_date": observation_date, "hypothetical_input": hypothetical,
        "model": model_name, "estimated_injury_probability": float(probability), "decision_threshold": float(threshold),
        "predicted_label_at_threshold": int(predicted_label), "risk_category": risk_category,
        "prediction_horizon": "next 7 days", "input_summary": input_summary,
        "explanation_method": explanation_method,
        "top_contributing_features": [] if contributions is None else contributions.to_dict("records"),
        "limitations": LIMITATIONS, "disclaimer": RESEARCH_DISCLAIMER,
    }


def report_to_csv(report: dict) -> bytes:
    rows = [("report", report["title"]), ("player_id", report["player_id"]), ("observation_date", report["observation_date"]),
            ("hypothetical_input", report["hypothetical_input"]), ("model", report["model"]),
            ("estimated_injury_probability", f"{report['estimated_injury_probability']:.4f}"),
            ("decision_threshold", f"{report['decision_threshold']:.4f}"),
            ("predicted_label_at_threshold", report["predicted_label_at_threshold"]), ("risk_category", report["risk_category"]),
            ("prediction_horizon", report["prediction_horizon"]), ("explanation_method", report["explanation_method"] or "n/a")]
    rows += [(f"input: {k}", v) for k, v in report["input_summary"].items()]
    for i, c in enumerate(report["top_contributing_features"], 1):
        rows.append((f"contribution {i}", f"{c['feature']} ({c['modality']}): {c['attribution']:+.4f} [{c['effect']}]"))
    rows += [(f"limitation {i}", t) for i, t in enumerate(report["limitations"], 1)]
    rows.append(("disclaimer", report["disclaimer"]))
    return pd.DataFrame(rows, columns=["field", "value"]).to_csv(index=False).encode()


def report_to_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    el = [Paragraph(esc(report["title"]), ss["Title"]),
          Paragraph("<b>SYNTHETIC DATA - research prototype. Not a medical diagnosis.</b>", ss["Normal"]), Spacer(1, 6)]
    summary = [["Player", report["player_id"] + (" (hypothetical input)" if report["hypothetical_input"] else "")],
               ["Observation date", report["observation_date"]], ["Model", report["model"]],
               ["Estimated 7-day injury probability", f"{report['estimated_injury_probability']:.1%}"],
               ["Decision threshold", f"{report['decision_threshold']:.3f}"],
               ["Risk category (illustrative)", report["risk_category"]]]

    def table(data, widths, header=False):
        t = Table([[Paragraph(esc(c), ss["BodyText"]) for c in row] for row in data], colWidths=widths)
        style = [("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEFE6")))
        t.setStyle(TableStyle(style))
        return t

    el += [table(summary, [65 * mm, 105 * mm]), Spacer(1, 8), Paragraph("Input summary", ss["Heading3"]),
           table([[k, str(v)] for k, v in report["input_summary"].items()], [65 * mm, 105 * mm])]
    if report["top_contributing_features"]:
        el += [Spacer(1, 8), Paragraph(f"Top contributing features ({esc(report['explanation_method'])})", ss["Heading3"]),
               table([["Feature", "Modality", "Contribution", "Effect on model output"]] +
                     [[c["feature"], c["modality"], f"{c['attribution']:+.3f}", c["effect"]] for c in report["top_contributing_features"]],
                     [62 * mm, 30 * mm, 25 * mm, 53 * mm], header=True),
               Paragraph("Contributions describe the model's behaviour (an association), not a cause of injury.", ss["Italic"])]
    el += [Spacer(1, 8), Paragraph("Limitations", ss["Heading3"])] + [Paragraph("- " + esc(t), ss["BodyText"]) for t in report["limitations"]]
    el += [Spacer(1, 8), Paragraph(esc(report["disclaimer"]), ss["Italic"])]
    doc.build(el)
    return buf.getvalue()
