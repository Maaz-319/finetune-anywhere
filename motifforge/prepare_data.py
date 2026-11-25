"""Build text-to-SVG chat JSONL from a captioned SVG dataset.

The default source is a captioned subset of the public SVG-Stack research data.
Review the dataset terms and attribute StarVector before publishing a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg_utils import SYSTEM_PROMPT, extract_svg, validate_svg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "JackertheHacker/svg-stack-qwen-captioned-subset"
DROPPED_TAGS = ("<text", "<tspan", "<foreignobject", "<script", "<image")


def normalize(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def estimate_tokens(text):
    return max(1, len(text) // 3)


def load_tokenizer(alias):
    import yaml
    from transformers import AutoTokenizer

    registry = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))
    return AutoTokenizer.from_pretrained(registry["models"][alias]["id"])


def accept(row, tokenizer, max_svg_tokens):
    caption = row.get("Caption") or row.get("caption") or ""
    raw_svg = row.get("Svg") or row.get("svg") or ""
    caption = re.sub(r"\s+", " ", str(caption)).strip()
    if not 15 <= len(caption) <= 300:
        return None, "caption length"
    svg = extract_svg(str(raw_svg))
    if not svg:
        return None, "no svg"
    ok, reason = validate_svg(svg)
    if not ok:
        return None, reason
    lowered = svg.lower()
    if any(tag in lowered for tag in DROPPED_TAGS):
        return None, "text or raster content"
    if len(svg) < 80:
        return None, "svg too small"
    if tokenizer is not None:
        tokens = len(tokenizer.encode(svg, add_special_tokens=False))
    else:
        tokens = estimate_tokens(svg)
    if tokens > max_svg_tokens:
        return None, "svg too long"
    return {"caption": caption, "svg": svg, "tokens": tokens}, "ok"


def to_messages(caption, svg):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": caption},
            {"role": "assistant", "content": svg},
        ]
    }


def collect(split, target, tokenizer, max_svg_tokens, used_prompts, used_hashes, scan_limit):
    rows = []
    skipped = {}
    scanned = 0
    for row in split:
        scanned += 1
        if len(rows) >= target or scanned > scan_limit:
            break
        item, reason = accept(row, tokenizer, max_svg_tokens)
        if not item:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        prompt_key = normalize(item["caption"])
        svg_key = hashlib.sha256(item["svg"].encode("utf-8")).hexdigest()
        if prompt_key in used_prompts or svg_key in used_hashes:
            skipped["duplicate"] = skipped.get("duplicate", 0) + 1
            continue
        used_prompts.add(prompt_key)
        used_hashes.add(svg_key)
        rows.append(to_messages(item["caption"], item["svg"]))
    return rows, skipped, scanned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-valid", type=int, default=400)
    parser.add_argument("--n-eval", type=int, default=60)
    parser.add_argument("--max-svg-tokens", type=int, default=1800)
    parser.add_argument("--tokenizer", help="registry alias for exact token counts, for example starter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-multiplier", type=int, default=25)
    parser.add_argument(
        "--license-note",
        default=(
            "Derived from the public SVG-Stack research dataset. Attribute StarVector and review "
            "the upstream terms before publishing a model or data."
        ),
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc

    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else None
    dataset = load_dataset(args.dataset)
    required = {"train", "validation", "test"}
    missing = required - set(dataset.keys())
    if missing:
        raise SystemExit(f"Dataset is missing splits: {sorted(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    used_prompts, used_hashes = set(), set()
    plan = [
        ("eval", dataset["test"], args.n_eval),
        ("valid", dataset["validation"], args.n_valid),
        ("train", dataset["train"], args.n_train),
    ]
    counts, skipped_all = {}, {}
    for name, split, target in plan:
        rows, skipped, scanned = collect(
            split.shuffle(seed=args.seed),
            target,
            tokenizer,
            args.max_svg_tokens,
            used_prompts,
            used_hashes,
            target * args.scan_multiplier,
        )
        if len(rows) < target:
            print(f"warning: {name}: wanted {target}, kept {len(rows)} after scanning {scanned}")
        counts[name] = len(rows)
        for key, value in skipped.items():
            skipped_all[key] = skipped_all.get(key, 0) + value
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (args.out / f"{name}.jsonl").write_text(text, encoding="utf-8")
        print(f"{name}: {len(rows)} examples from {scanned} scanned")

    manifest = {
        "dataset": args.dataset,
        "dataset_url": f"https://huggingface.co/datasets/{args.dataset}",
        "license_note": args.license_note,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_svg_tokens": args.max_svg_tokens,
        "tokenizer": args.tokenizer or "character estimate",
        "splits": counts,
        "skipped": dict(sorted(skipped_all.items(), key=lambda item: -item[1])),
        "system_prompt": SYSTEM_PROMPT,
        "sha256": {
            name: hashlib.sha256((args.out / f"{name}.jsonl").read_bytes()).hexdigest()
            for name in counts
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print("Validate with:")
    print(
        "  python scripts/validate_dataset.py",
        args.out / "train.jsonl",
        args.out / "valid.jsonl",
        args.out / "eval.jsonl",
        "--tokenizer",
        args.tokenizer or "starter",
        "--max-tokens",
        "2048",
    )


if __name__ == "__main__":
    main()
