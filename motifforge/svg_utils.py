"""Shared SVG helpers for MotifForge: extract, validate, render and score."""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET

SYSTEM_PROMPT = (
    "You are MotifForge, a vector graphic designer for apparel prints, logos and simple icons. "
    "Given a short description, reply with one complete SVG document and nothing else. "
    "Start with <svg and end with </svg>, include a viewBox, and use only self-contained shapes, "
    "paths, colors and simple gradients. Never use markdown fences, scripts, external images, "
    "web fonts, text elements or links. Keep the drawing centered, clean and editable, with as "
    "few elements as needed."
)

FORBIDDEN_TAGS = {
    "script",
    "foreignobject",
    "iframe",
    "video",
    "audio",
    "image",
    "text",
    "tspan",
    "textpath",
}


def _local(tag):
    return str(tag).rsplit("}", 1)[-1].lower()


def _has_canvas(root):
    view_box = root.get("viewBox") or ""
    if re.search(r"-?\d", view_box):
        return True
    width = root.get("width") or ""
    height = root.get("height") or ""
    return bool(re.search(r"\d", width) and re.search(r"\d", height))


def extract_svg(text):
    """Return the first complete SVG document inside a model reply, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:svg|xml)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = re.search(r"<svg\b", cleaned, flags=re.IGNORECASE)
    if not start:
        return None
    end = cleaned.lower().rfind("</svg>")
    if end == -1:
        return None
    return cleaned[start.start() : end + len("</svg>")].strip()


def validate_svg(svg, max_chars=200_000):
    """Check that an SVG string is safe, self-contained and renderable in principle."""
    if not isinstance(svg, str) or not svg.strip():
        return False, "empty"
    if len(svg) > max_chars:
        return False, "too large"
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        return False, f"xml: {exc}"
    if _local(root.tag) != "svg":
        return False, "root is not svg"
    if not _has_canvas(root):
        return False, "missing viewBox or width/height"
    if "@import" in svg.lower():
        return False, "css import"
    for element in root.iter():
        name = _local(element.tag)
        if name in FORBIDDEN_TAGS:
            return False, f"forbidden tag {name}"
        for key, value in element.attrib.items():
            attr = _local(key)
            if attr == "href" and re.match(r"^\s*(https?:|file:|//|data:)", value, re.IGNORECASE):
                return False, "external href"
            if attr == "style" and re.search(r"url\(\s*['\"]?(https?:|//)", value, re.IGNORECASE):
                return False, "external style url"
    return True, "ok"


def render_svg(svg, size=512, background="white"):
    """Rasterize an SVG string to a PIL RGB image, or return None on failure."""
    try:
        import cairosvg
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install cairosvg and pillow to render SVG") from exc
    try:
        png = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=int(size),
            output_height=int(size),
            background_color=background,
        )
        return Image.open(io.BytesIO(png)).convert("RGB")
    except Exception:
        return None


def ssim_score(image_a, image_b, size=256):
    """Structural similarity between two images, resized to a common square."""
    try:
        import numpy as np
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError("Install scikit-image and numpy to score") from exc
    a = np.asarray(image_a.convert("L").resize((size, size)), dtype=np.float32)
    b = np.asarray(image_b.convert("L").resize((size, size)), dtype=np.float32)
    return float(structural_similarity(a, b, data_range=255))


def ink_ratio(image, threshold=250):
    """Fraction of non-white pixels, used to flag blank outputs."""
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype="uint8")
    return float((gray < threshold).mean())


def contact_sheet(images, cols=3, pad=8, background=(255, 255, 255)):
    """Combine images into a grid for quick visual review."""
    from PIL import Image

    images = [image for image in images if image is not None]
    if not images:
        return None
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows = (len(images) + cols - 1) // cols
    width = cols * cell_w + (cols + 1) * pad
    height = rows * cell_h + (rows + 1) * pad
    sheet = Image.new("RGB", (width, height), background)
    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        sheet.paste(image, (x, y))
    return sheet


def one_line(text, limit=90):
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text[:limit] + ("..." if len(text) > limit else "")
