"""Score base versus fine-tuned SVG outputs and write a reproducible report."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama_client
from svg_utils import contact_sheet, extract_svg, ink_ratio, one_line, render_svg, ssim_score, validate_svg


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def score_svg(svg, reference=None, size=256):
    clean = extract_svg(svg) if isinstance(svg, str) else None
    clean = clean or (svg if isinstance(svg, str) else "")
    result = {"valid": False, "reason": "empty", "render": False, "ssim": None, "ink": None, "chars": len(clean)}
    if not clean.strip():
        return result
    ok, reason = validate_svg(clean)
    result["valid"] = bool(ok)
    result["reason"] = reason
    if not ok:
        return result
    image = render_svg(clean, size=size)
    if image is None:
        result["reason"] = "render failed"
        return result
    result["render"] = True
    result["reason"] = "ok"
    result["ink"] = round(ink_ratio(image), 4)
    if isinstance(reference, str) and reference.strip():
        ref_clean = extract_svg(reference) or reference
        ref_image = render_svg(ref_clean, size=size)
        if ref_image is not None:
            result["ssim"] = round(ssim_score(image, ref_image), 4)
    return result


def summarize(rows, key):
    scored = [row[key] for row in rows]
    n = len(scored)
    if not n:
        return {"n": 0}
    ssims = [item["ssim"] for item in scored if item["ssim"] is not None]
    return {
        "n": n,
        "valid_rate": round(sum(item["valid"] for item in scored) / n, 3),
        "render_rate": round(sum(item["render"] for item in scored) / n, 3),
        "blank_rate": round(sum(1 for item in scored if item["render"] and item["ink"] < 0.001) / n, 3),
        "mean_ssim": round(statistics.fmean(ssims), 4) if ssims else None,
        "median_ssim": round(statistics.median(ssims), 4) if ssims else None,
        "ssim_pairs": len(ssims),
    }


def status_text(item):
    if not item["valid"]:
        return "invalid: " + one_line(item["reason"], 40)
    if not item["render"]:
        return "render failed"
    if item["ssim"] is None:
        return f"ok, ink {item['ink']}"
    return f"ok, SSIM {item['ssim']}"


def write_images(out, index, record, size):
    stem = f"{index:03d}"
    reference = extract_svg(record.get("reference_svg", "")) or ""
    reference_image = render_svg(reference, size=size) if reference else None
    if reference_image is not None:
        reference_image.save(out / f"ref_{stem}.png")
    images = [reference_image]
    for name, key in [("base", "base_svg"), ("tuned", "tuned_svg")]:
        clean = extract_svg(record.get(key, "")) or ""
        image = render_svg(clean, size=size) if clean else None
        if image is not None:
            image.save(out / f"{name}_{stem}.png")
        images.append(image)
    sheet = contact_sheet(images, cols=3)
    if sheet is not None:
        sheet.save(out / f"pair_{stem}.png")


def build_records(eval_path, base_model, tuned_model, base_url, tuned_url, limit, max_new_tokens):
    rows = load_jsonl(eval_path)[:limit]
    records = []
    for index, row in enumerate(rows, 1):
        messages = row.get("messages") or []
        user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        reference = messages[-1].get("content", "") if messages and messages[-1].get("role") == "assistant" else ""
        base_text, _, _ = ollama_client.chat(base_url, base_model, user, max_new_tokens=max_new_tokens)
        tuned_text, _, _ = ollama_client.chat(tuned_url, tuned_model, user, max_new_tokens=max_new_tokens)
        records.append(
            {"prompt": user, "reference_svg": reference, "base_svg": base_text, "tuned_svg": tuned_text}
        )
        print(f"[{index}/{len(rows)}] generated base and tuned SVG")
    return records


def report_markdown(summary, rows):
    base, tuned = summary["base"], summary["tuned"]
    lines = [
        "# MotifForge held-out evaluation",
        "",
        f"Examples scored: {summary['n']}. Preview size: {summary['size']} px.",
        "Valid means well-formed and self-contained. SSIM is a secondary pixel signal, not a quality verdict.",
        "",
        "| Metric | Base | Fine-tuned |",
        "| --- | ---: | ---: |",
        f"| Valid SVG rate | {base.get('valid_rate')} | {tuned.get('valid_rate')} |",
        f"| Render rate | {base.get('render_rate')} | {tuned.get('render_rate')} |",
        f"| Blank rate | {base.get('blank_rate')} | {tuned.get('blank_rate')} |",
        f"| Mean SSIM | {base.get('mean_ssim')} | {tuned.get('mean_ssim')} |",
        f"| Median SSIM | {base.get('median_ssim')} | {tuned.get('median_ssim')} |",
        f"| SSIM pairs | {base.get('ssim_pairs')} | {tuned.get('ssim_pairs')} |",
        "",
        "## Per example",
        "",
        "| # | Prompt | Base | Fine-tuned |",
        "| ---: | --- | --- | --- |",
    ]
    for index, row in enumerate(rows[:20], 1):
        lines.append(
            f"| {index} | {one_line(row['prompt'], 70)} | {status_text(row['base'])} | "
            f"{status_text(row['tuned'])} |"
        )
    lines += [
        "",
        "Rendered comparisons are saved in this folder as pair_001.png and so on.",
        "The reference design is never shown to either model.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, help="comparison.json written by the training notebook")
    parser.add_argument("--eval-jsonl", type=Path, help="held-out chat JSONL to query through Ollama")
    parser.add_argument("--base-model", help="Ollama model name for the base control")
    parser.add_argument("--tuned-model", help="Ollama model name for the fine-tuned model")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--tuned-url", help="defaults to --base-url")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all records")
    parser.add_argument("--png-limit", type=int, default=40)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.records:
        records = json.loads(args.records.read_text(encoding="utf-8"))
    elif args.eval_jsonl and args.base_model and args.tuned_model:
        limit = args.limit or 10**9
        records = build_records(
            args.eval_jsonl,
            args.base_model,
            args.tuned_model,
            args.base_url,
            args.tuned_url or args.base_url,
            limit,
            args.max_new_tokens,
        )
    else:
        parser.error("Provide --records, or --eval-jsonl with --base-model and --tuned-model")

    records = records[: args.limit] if args.limit else records
    rows = []
    for index, record in enumerate(records, 1):
        base = score_svg(record.get("base_svg", ""), record.get("reference_svg", ""), args.size)
        tuned = score_svg(record.get("tuned_svg", ""), record.get("reference_svg", ""), args.size)
        rows.append({"prompt": record.get("prompt", ""), "base": base, "tuned": tuned})
        if index <= args.png_limit:
            write_images(args.out, index, record, args.size)

    summary = {"n": len(rows), "size": args.size, "base": summarize(rows, "base"), "tuned": summarize(rows, "tuned")}
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out / "scored.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(report_markdown(summary, rows), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Wrote", args.out / "report.md")


if __name__ == "__main__":
    main()
