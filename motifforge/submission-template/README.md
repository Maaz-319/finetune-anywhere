# MotifForge submission form

Replace every placeholder with your real result. This folder is a copy target, not a published entry.

## What it does

MotifForge turns a short description such as "a three petal floral motif in teal and sand" into one clean, editable SVG document. Designers use it to sketch vectors for apparel prints, logos and simple icons without starting from a blank canvas.

## Why we built it

Print artwork must be vector so it scales to any size without pixels. Tracing tools turn an existing PNG into noisy paths that are hard to edit. A small language model fine-tuned on captioned SVG data can generate a compact, semantic vector directly from intent, which is faster to edit and cheaper to run locally.

## Dataset

- Source: `<dataset name and URL>`
- License and attribution: `<state the exact terms you relied on and credit StarVector>`
- Training examples: `<count>`
- Validation examples: `<count>`
- Held-out evaluation examples: `<count>`
- Filters: `<summarize length, safety and text filters>`
- Original prompt set: `<describe the 20 textile prompt evaluation file>`

## Training

- Tool: Unsloth with QLoRA
- Platform: Kaggle free GPU, T4 x2
- Base model and revision: `<resolved ID and revision>`
- Method: 4-bit QLoRA, rank `<r>`, alpha `<alpha>`, learning rate `<lr>`, `<epochs>` epochs
- Context: `<max sequence length>`
- Software versions: `<from run.json>`
- Training time: `<minutes>`
- Peak VRAM: `<GiB>`

## Run it locally

```bash
ollama create motifforge -f Modelfile
ollama run motifforge
python motifforge/app.py --model motifforge
```

- Quantization: Q4_K_M
- Runtime: Ollama
- Hardware tested: Lenovo ThinkBook 14 G6, 16 GB DDR5, Ubuntu, CPU only
- Model size: `<GB>`
- Generation speed: `<tokens per second>`
- Peak RAM: `<GB>`
- System prompt: see `motifforge/Modelfile`

## Before and after

Prompt: `<one held-out prompt>`

Base output: `<short base result or note>`

Fine-tuned output: `<short tuned result or note>`

How we judged it: `<valid SVG rate, render rate, SSIM, human 1 to 5 score, and the rubric>`.

Limitations: `<one honest failure, for example complex illustrations or unusual shapes>`.

Rendered examples: `<link to images or name the files in this folder>`.

## What we learned

`<What helped, what failed, what you would change with more time.>`

## Team

- Maaz, GitHub `<handle>`
- Ahmad, GitHub `<handle>`
- Waleed, GitHub `<handle>`

Data and tools credited: `<Qwen base model, Unsloth, StarVector data, Ollama>`.
