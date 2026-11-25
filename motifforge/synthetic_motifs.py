"""Generate an original, license-clean text-to-SVG motif dataset.

This is an offline fallback and a textile-style supplement to the public SVG data.
Every SVG is produced by code in this file, so the generated data can be released
under MIT together with this repository.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg_utils import SYSTEM_PROMPT

PALETTES = [
    ("deep teal", "#1f6f78", "warm sand", "#e9d8a6", "soft cream", "#faf3e0"),
    ("navy", "#1d3557", "mustard", "#e9c46a", "pale blue", "#eef4f8"),
    ("charcoal", "#2b2d42", "coral", "#ef8354", "off white", "#f7f4ef"),
    ("forest", "#2d6a4f", "blush", "#f4acb7", "mint", "#eaf4ee"),
    ("plum", "#5f0f40", "gold", "#e0a458", "light sand", "#f6efe3"),
    ("slate", "#3d5a80", "terracotta", "#e4572e", "warm white", "#fbf7f0"),
]


def svg_doc(body, background):
    rect = f'<rect width="512" height="512" fill="{background}"/>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">{rect}{body}</svg>'


def star_points(cx, cy, outer, inner, points):
    coords = []
    for step in range(points * 2):
        radius = outer if step % 2 == 0 else inner
        angle = -math.pi / 2 + step * math.pi / points
        coords.append(f"{cx + radius * math.cos(angle):.1f},{cy + radius * math.sin(angle):.1f}")
    return " ".join(coords)


def petals(rng, palette):
    name, color, accent, bg = palette[0], palette[1], palette[3], palette[5]
    count = rng.randint(5, 12)
    rx, ry = rng.randint(26, 46), rng.randint(70, 104)
    parts = []
    for index in range(count):
        angle = index * 360 / count
        parts.append(
            f'<ellipse cx="256" cy="176" rx="{rx}" ry="{ry}" fill="{color}" '
            f'transform="rotate({angle:.1f} 256 256)"/>'
        )
    parts.append(f'<circle cx="256" cy="256" r="{rng.randint(18, 34)}" fill="{accent}"/>')
    caption = f"A centered {name} floral motif with {count} petals on a {palette[4]} background, flat vector style."
    return svg_doc("".join(parts), bg), caption


def concentric(rng, palette):
    name, color, accent, bg = palette[0], palette[1], palette[3], palette[5]
    count = rng.randint(3, 7)
    step = 220 // (count + 1)
    parts = []
    for index in range(count):
        radius = 220 - index * step
        stroke = color if index % 2 == 0 else accent
        parts.append(
            f'<circle cx="256" cy="256" r="{radius}" fill="none" stroke="{stroke}" '
            f'stroke-width="{max(4, 14 - index)}"/>'
        )
    caption = f"A {name} target motif of {count} concentric rings on a {palette[4]} background, minimal line art."
    return svg_doc("".join(parts), bg), caption


def polka(rng, palette):
    name, color, accent, bg = palette[0], palette[1], palette[3], palette[5]
    cols = rng.randint(4, 7)
    cell = 512 // cols
    radius = rng.randint(8, max(9, cell // 4))
    parts = []
    for row in range(cols):
        for col in range(cols):
            fill = color if (row + col) % 2 == 0 else accent
            offset = cell // 2 if row % 2 else 0
            cx = col * cell + cell // 2 + offset
            cy = row * cell + cell // 2
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}"/>')
    caption = f"A {palette[4]} repeating tile with {cols} rows of {name} polka dots and {palette[2]} accents."
    return svg_doc("".join(parts), bg), caption


def diamonds(rng, palette):
    name, color, accent, bg = palette[0], palette[1], palette[3], palette[5]
    cols = rng.randint(3, 6)
    cell = 512 // cols
    size = cell // 3
    parts = []
    for row in range(cols):
        for col in range(cols):
            fill = color if (row + col) % 2 == 0 else accent
            cx, cy = col * cell + cell // 2, row * cell + cell // 2
            parts.append(
                f'<rect x="{cx - size}" y="{cy - size}" width="{2 * size}" height="{2 * size}" '
                f'fill="{fill}" transform="rotate(45 {cx} {cy})"/>'
            )
    caption = f"A geometric tile of {name} and {palette[2]} diamonds on a {palette[4]} background, flat print style."
    return svg_doc("".join(parts), bg), caption


def chevron(rng, palette):
    name, color, accent, bg = palette[0], palette[1], palette[3], palette[5]
    rows = rng.randint(4, 8)
    width = 32
    parts = []
    for row in range(rows):
        y = 40 + row * (432 // rows)
        stroke = color if row % 2 == 0 else accent
        points = [f"M-16 {y}"]
        x = -16
        while x < 528:
            points.append(f"l{width} -28 l{width} 28")
            x += 2 * width
        parts.append(
            f'<path d="{" ".join(points)}" fill="none" stroke="{stroke}" stroke-width="{rng.randint(6, 16)}"/>'
        )
    caption = f"A {name} chevron stripe motif on a {palette[4]} background, repeating textile tile."
    return svg_doc("".join(parts), bg), caption


def stripes(rng, palette):
    name, color, accent, bg = palette[0], palette[1], palette[3], palette[5]
    step = rng.randint(28, 56)
    parts = []
    for index, x in enumerate(range(-512, 1024, step)):
        stroke = color if index % 3 else accent
        parts.append(
            f'<line x1="{x}" y1="0" x2="{x + 512}" y2="512" stroke="{stroke}" '
            f'stroke-width="{rng.randint(8, 22)}"/>'
        )
    caption = f"Diagonal {name} stripes with {palette[2]} accents on a {palette[4]} background, simple border print."
    return svg_doc("".join(parts), bg), caption


def stars(rng, palette):
    name, color, accent, bg = palette[0], palette[1], palette[3], palette[5]
    points = rng.choice([4, 5, 6, 8])
    outer = rng.randint(110, 170)
    inner = int(outer * 0.45)
    parts = [
        f'<circle cx="256" cy="256" r="{outer + 22}" fill="none" stroke="{accent}" stroke-width="10"/>',
        f'<polygon points="{star_points(256, 256, outer, inner, points)}" fill="{color}"/>',
    ]
    caption = f"A {name} {points} point star badge with a {palette[2]} ring on a {palette[4]} background, logo style."
    return svg_doc("".join(parts), bg), caption


def scallop(rng, palette):
    name, color, accent, bg = palette[0], palette[1], palette[3], palette[5]
    rows = rng.randint(3, 6)
    radius = rng.randint(28, 46)
    parts = []
    for row in range(rows):
        y = 60 + row * (400 // rows)
        stroke = color if row % 2 == 0 else accent
        x = 0
        path = [f"M{x} {y}"]
        while x < 512:
            path.append(f"a{radius} {radius} 0 0 1 {2 * radius} 0")
            x += 2 * radius
        parts.append(
            f'<path d="{" ".join(path)}" fill="none" stroke="{stroke}" stroke-width="10"/>'
        )
    caption = f"A scallop shell repeat of {name} arcs with {palette[2]} rows on a {palette[4]} background, border motif."
    return svg_doc("".join(parts), bg), caption


FAMILIES = [petals, concentric, polka, diamonds, chevron, stripes, stars, scallop]


def make_row(rng):
    palette = rng.choice(PALETTES)
    family = rng.choice(FAMILIES)
    svg, caption = family(rng, palette)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": caption},
            {"role": "assistant", "content": svg},
        ]
    }, caption


def build_split(name, target, seed, seen):
    rng = random.Random(seed)
    rows = []
    attempts = 0
    limit = max(target * 60, 1000)
    while len(rows) < target and attempts < limit:
        attempts += 1
        row, caption = make_row(rng)
        if caption.lower() in seen:
            continue
        seen.add(caption.lower())
        rows.append(row)
    if len(rows) < target:
        print(f"warning: {name}: generated {len(rows)} of {target}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-valid", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seen = set()
    plan = [("train", args.n_train, args.seed), ("valid", args.n_valid, args.seed + 1), ("eval", args.n_eval, args.seed + 2)]
    counts = {}
    for name, target, seed in plan:
        rows = build_split(name, target, seed, seen)
        counts[name] = len(rows)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (args.out / f"{name}.jsonl").write_text(text, encoding="utf-8")
        print(f"{name}: {len(rows)} examples")
    manifest = {
        "dataset": "motifforge synthetic motifs",
        "license": "MIT",
        "license_note": "Original SVG and captions generated by motifforge/synthetic_motifs.py.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "splits": counts,
        "system_prompt": SYSTEM_PROMPT,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Next: python scripts/validate_dataset.py", args.out / "train.jsonl", args.out / "valid.jsonl", args.out / "eval.jsonl")


if __name__ == "__main__":
    main()
