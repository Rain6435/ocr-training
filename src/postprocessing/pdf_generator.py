import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from PIL import Image


def create_searchable_pdf(
    image_path: str,
    text: str,
    output_path: str,
    confidence: float = None,
) -> str:
    """
    Create a searchable PDF with the original image as background
    and an invisible text layer on top for search/copy.

    Returns path to output PDF.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Get image dimensions
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Scale to fit page
    page_w, page_h = letter
    scale = min(page_w / img_w, page_h / img_h) * 0.9
    display_w = img_w * scale
    display_h = img_h * scale
    x_offset = (page_w - display_w) / 2
    y_offset = (page_h - display_h) / 2

    c = canvas.Canvas(output_path, pagesize=letter)

    # Draw background image
    c.drawImage(
        image_path, x_offset, y_offset,
        width=display_w, height=display_h,
        preserveAspectRatio=True,
    )

    # Overlay invisible text for searchability
    # White text, small font — invisible but selectable
    c.setFillColorRGB(1, 1, 1, alpha=0)  # Fully transparent
    c.setFont("Helvetica", 1)  # Tiny font

    lines = text.split("\n") if text else []
    if lines:
        line_height = display_h / max(len(lines), 1)
        for i, line in enumerate(lines):
            y = y_offset + display_h - (i + 1) * line_height
            c.drawString(x_offset + 5, y, line)

    # Add metadata
    if confidence is not None:
        c.setAuthor("Historical Document OCR Pipeline")
        c.setSubject(f"OCR Confidence: {confidence:.2%}")

    c.save()
    return output_path
