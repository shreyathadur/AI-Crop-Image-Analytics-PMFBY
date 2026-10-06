from io import BytesIO
from textwrap import wrap

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
from pathlib import Path

from backend.config import DISCLAIMER
import logging

logger = logging.getLogger(__name__)


def make_pdf(analysis):
    output = BytesIO(); pdf = canvas.Canvas(output, pagesize=letter)
    width, height = letter
    pdf.setTitle("Preliminary Crop Image Assessment")
    pdf.setPageCompression(0)
    pdf.setFont("Helvetica-Bold", 17); pdf.drawString(48, height - 54, "Crop Image Analytics — Preliminary Report")
    values = [
        ("Analysis ID", analysis.get("id")), ("Date/time", analysis.get("created_at")),
        ("Uploaded image", analysis.get("image_name")), ("Crop", analysis.get("crop")),
        ("Prediction", analysis.get("prediction")), ("Health status", analysis.get("health_status")),
        ("Confidence", f"{analysis.get('confidence', 0):.1%}"),
        ("Estimated Visual Damage Indicator", _indicator_value(analysis)),
        ("Severity Category", analysis.get("severity")),
        ("Severity Method", analysis.get("severity_method", "Not provided")),
        ("Explanation", analysis.get("explanation")), ("Disclaimer", DISCLAIMER),
        ("Severity note", analysis.get("severity_explanation")),
        ("Important:", "The visual damage indicator is an AI-assisted preliminary image-based estimate. It does not represent an official PMFBY claim percentage, crop-loss certification, or government decision."),
    ]
    weather = analysis.get("weather")
    if isinstance(weather, dict) and weather.get("available"):
        values.extend([
            ("ENVIRONMENTAL CONTEXT", ""),
            ("Weather source", f"{weather.get('provider', 'Open-Meteo')} (https://open-meteo.com/)"),
            ("Location", weather.get("location")),
            ("Temperature", f"{weather['temperature_c']:.1f} C" if isinstance(weather.get("temperature_c"), (int, float)) else None),
            ("Humidity", f"{weather['humidity_percent']:.1f}%" if isinstance(weather.get("humidity_percent"), (int, float)) else None),
            ("Precipitation", f"{weather['precipitation_mm']:.1f} mm" if isinstance(weather.get("precipitation_mm"), (int, float)) else None),
            ("Wind", f"{weather['wind_speed_kmh']:.1f} km/h" if isinstance(weather.get("wind_speed_kmh"), (int, float)) else None),
            ("Condition", weather.get("condition")),
            ("Observation time", weather.get("observed_at")),
            ("Environmental disclaimer", "Environmental data is provided as contextual information. It does not establish the cause of crop disease, actual crop loss, insurance eligibility, or insurance compensation."),
        ])
    explanation = None
    try:
        from backend.services.gradcam import generate_gradcam
        if analysis.get("image_path") and analysis.get("prediction"):
            explanation = generate_gradcam(analysis["image_path"], analysis["prediction"], include_heatmap=False)
    except Exception:
        logger.exception("Grad-CAM unavailable while generating PDF for analysis %s", analysis.get("id"))
    y = height - 90
    for label, value in values:
        label_lines = wrap(str(label), width=22) or [""]
        pdf.setFont("Helvetica-Bold", 10)
        for line_number, line in enumerate(label_lines):
            pdf.drawString(48, y - line_number * 14, line)
        pdf.setFont("Helvetica", 10)
        value_lines = wrap(str(value or ""), width=55) or [""]
        text = pdf.beginText(220, y); text.setLeading(14)
        for line in value_lines:
            text.textLine(line)
        pdf.drawText(text); y -= max(28, 14 * max(len(label_lines), len(value_lines)))
        if y < 72:
            pdf.showPage(); y = height - 54
    image_path = analysis.get("image_path")
    if image_path and Path(image_path).is_file():
        pdf.showPage()
        pdf.setFont("Helvetica-Bold", 13); pdf.drawString(48, height - 54, "Uploaded image")
        with Image.open(image_path) as image:
            pdf.drawImage(ImageReader(image.convert("RGB")), 48, 190, width=500, height=500, preserveAspectRatio=True, anchor="c")
    if explanation:
        pdf.showPage()
        pdf.setFont("Helvetica-Bold", 13); pdf.drawString(48, height - 54, "AI MODEL EXPLANATION")
        pdf.setFont("Helvetica", 9); pdf.drawString(48, height - 75, "Grad-CAM Visualization")
        pdf.drawImage(ImageReader(BytesIO(explanation["original_png"])), 48, 280, width=245, height=245,
                      preserveAspectRatio=True, anchor="c")
        pdf.drawImage(ImageReader(BytesIO(explanation["overlay_png"])), 318, 280, width=245, height=245,
                      preserveAspectRatio=True, anchor="c")
        note = ("Grad-CAM highlights image regions that contributed to the model prediction. It is a qualitative explanation and should not be interpreted as an exact disease boundary, disease severity measurement, crop-loss percentage, or official insurance assessment.")
        text = pdf.beginText(48, 250); text.setLeading(13)
        for line in wrap(note, width=95):
            text.textLine(line)
        pdf.drawText(text)
    pdf.save(); output.seek(0); return output.getvalue()


def _indicator_value(analysis):
    value = analysis.get("estimated_visual_damage_indicator")
    if value is None:
        value = analysis.get("estimated_damage_indicator")
    return f"{value}%" if value is not None else "Not provided"
