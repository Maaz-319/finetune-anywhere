# MotifForge: text to SVG motifs on a laptop

MotifForge is a small Qwen3 text model fine-tuned to turn a short description into a **single, clean, editable SVG** icon, logo or textile motif. It runs quantized on a normal laptop with no GPU.

This folder is the complete kit: data preparation, a Kaggle training notebook, an evaluation harness, a local Ollama setup and a Gradio demo.

## Snapshot

| Item | Value |
| --- | --- |
| Task | Description to one SVG document |
| Base model | Qwen3 1.7B, Apache-2.0, alias `starter` |
| Method | QLoRA, 4-bit, Unsloth, SFT |
| Data | Captioned SVG-Stack subset, plus optional original synthetic motifs |
| Training | Kaggle free GPU, T4 x2 |
| Output | LoRA adapter, merged model, Q4_K_M GGUF |
| Local runtime | Ollama on CPU, Ubuntu, 16 GB RAM |
| Demo | Gradio app on localhost |

The base model ID is resolved from the repository registry, never copied into this folder:

```bash
python scripts/model_id.py starter
```

## Why this task

Apparel, signage and logo work needs vector art that scales to any print size. Tracing tools already convert an existing PNG into paths, but they produce noisy, hard-to-edit outlines. A fine-tuned language model can generate a compact, semantic SVG directly from a design brief: named shapes, few elements, editable in Inkscape or Illustrator.

Honest scope: this model handles **simple icons, logos and motifs**, not photographs or detailed illustrations. It is a vector-first designer, not a replacement for tracing.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r motifforge/requirements.txt

# Optional: build an original license-clean teaching set first
.venv/bin/python motifforge/synthetic_motifs.py --out outputs/motifforge-synthetic

# Or download and filter the public captioned SVG data
.venv/bin/python motifforge/prepare_data.py --out outputs/motifforge-data --tokenizer starter

.venv/bin/python scripts/validate_dataset.py \
  outputs/motifforge-data/train.jsonl \
  outputs/motifforge-data/valid.jsonl \
  outputs/motifforge-data/eval.jsonl \
  --tokenizer starter --max-tokens 2048
```

Then open `motifforge/train_kaggle.ipynb` on Kaggle, set your fork URL, and run the cells.

## Data

Primary source: `JackertheHacker/svg-stack-qwen-captioned-subset` on Hugging Face. It adds natural-language captions to the public SVG-Stack research dataset from StarVector. Fields used are `Caption` and `Svg`.

Filters applied by `motifforge/prepare_data.py`:

| Filter | Reason |
| --- | --- |
| XML parses, root is `svg`, has `viewBox` or width/height | trainable, renderable targets |
| No script, image, foreignObject, text or external href | safety and small-model quality |
| SVG token length at or below the budget | fits the 2048 context |
| Caption length 15 to 300 characters | useful prompts |
| Deduplicate SVG and prompt across splits | no leakage |
| Evaluation taken from the dataset test split first | held-out examples stay unseen |

Every run writes `manifest.json` with counts, filters, hashes and the system prompt.

License note: the upstream SVG-Stack data is a research dataset that aggregates public SVG sources, and the Hugging Face card does not state a single license. Attribute StarVector (https://github.com/joanrod/star-vector) and review the upstream terms before publishing a model or data. If you need a fully clean dataset, use the original generator in `motifforge/synthetic_motifs.py` (MIT) or ask the organizers.

The 20 prompts in `motifforge/eval_prompts_motifs.jsonl` were written for this project, so they are also safe to publish.

## Training

Open `motifforge/train_kaggle.ipynb`.

1. Kaggle: enable GPU T4 x2 and Internet.
2. Set `REPO_URL` to your fork of this repository.
3. Choose `DATA_MODE = "prepare"` to build the data on Kaggle, or `"attach"` to use an uploaded Kaggle dataset at `/kaggle/input/motifforge-data`.
4. Run the cells in order.

Default settings in `motifforge/config.yaml`:

| Setting | Value |
| --- | --- |
| LoRA rank / alpha | 32 / 64 |
| Learning rate | 0.0002 |
| Epochs | 2 |
| Max sequence length | 2048 |
| Microbatch / accumulation | 1 / 8 |
| Seed | 42 |
| Max new tokens at inference | 1800 |
| Quantization | Q4_K_M |

The notebook records base answers before training, trains, records fine-tuned answers on the same held-out prompts, saves an adapter, scores both, and exports a Q4_K_M GGUF.

Replace these placeholders after your run:

| Report item | Your value |
| --- | --- |
| Kaggle GPU | `<T4 x2>` |
| Training time | `<minutes>` |
| Peak VRAM | `<GiB>` |
| Training examples | `<count>` |
| Software versions | `<from run.json>` |

## Before and after

The notebook writes `comparison.json` with prompts, reference SVGs, base SVGs and fine-tuned SVGs. Score it again anywhere with:

```bash
.venv/bin/python motifforge/evaluate.py \
  --records runs/motifforge-YYYYMMDD-HHMMSS/comparison.json \
  --out outputs/motifforge-results --size 256
```

The report contains:

| Metric | What it shows |
| --- | --- |
| Valid SVG rate | well-formed and self-contained output |
| Render rate | can be rasterized without error |
| Blank rate | output was valid but effectively empty |
| Mean and median SSIM | secondary pixel similarity to the reference |
| Per-example table | base versus tuned on the same prompt |

SSIM is a secondary signal. Two different but valid designs can score low, so lead with valid and render rates, then show the rendered images and a short human rating (1 to 5) from the team.

Fill this summary table from `report.md`:

| Metric | Base | Fine-tuned |
| --- | ---: | ---: |
| Valid SVG rate | `<value>` | `<value>` |
| Render rate | `<value>` | `<value>` |
| Mean SSIM | `<value>` | `<value>` |
| Human rating (1 to 5) | `<value>` | `<value>` |

Add one real failure case. Judges reward honest reporting.

## Run locally with Ollama

Download the Q4_K_M GGUF from the notebook or the model hub, then:

```bash
cd motifforge
# put motifforge-q4_k_m.gguf next to the Modelfile
ollama create motifforge -f Modelfile
ollama run motifforge
```

Local hardware used for this project:

| Item | Value |
| --- | --- |
| Laptop | Lenovo ThinkBook 14 G6 |
| RAM | 16 GB DDR5 |
| OS | Ubuntu |
| GPU | none, CPU inference |

Record real local numbers:

| Measurement | Your value |
| --- | --- |
| Model file size | `<GB>` |
| Load time | `<seconds>` |
| Generation speed | `<tokens per second>` |
| Peak RAM | `<GB>` |
| Typical SVG time | `<seconds>` |

For a local before-and-after against the base model, pull a base Ollama tag and run:

```bash
ollama pull qwen3:1.7b
.venv/bin/python motifforge/evaluate.py \
  --eval-jsonl motifforge/eval_prompts_motifs.jsonl \
  --base-model qwen3:1.7b --tuned-model motifforge \
  --out outputs/motifforge-local --limit 20
```

## Demo

```bash
.venv/bin/python motifforge/app.py --model motifforge
```

Open http://127.0.0.1:7860. The app sends the prompt to local Ollama, extracts the SVG, validates it, renders a preview at the chosen size, and offers the SVG for download. A scale slider proves the vector stays sharp at poster sizes.

Suggested 2 to 3 minute video:

1. Show the base model failing or returning poor SVG for one prompt.
2. Show MotifForge returning a clean SVG for the same prompt.
3. Scale the preview to large size, then open the SVG in Inkscape.
4. Show the held-out metrics and one failure.

## Publish and submit

1. Push the adapter and GGUF to Hugging Face and ModelScope.
2. Copy `motifforge/submission-template/` to `showcase/qwen-local-hackathon-2026/motifforge/` and fill every field.
3. Open one pull request that touches only that folder.
4. Keep files under 5 MB and never commit weights.

Model card must state: base model and revision, data source and license, filters, split method, hyperparameters, hardware, training time, quantization, local runtime and measured speed, real before and after examples, and limitations.

## Mapping to the rubric

| Criterion | Evidence in this kit |
| --- | --- |
| Usefulness | vector-first motifs for print and logos, local and private |
| Proof it helped | held-out valid, render and SSIM table plus rendered pairs |
| Runs locally | Q4_K_M GGUF, Ollama on 16 GB CPU laptop, measured speed |
| Demo and write-up | Gradio app, video script, reproducible commands, model card |

## Team

- Maaz: training, evaluation, submission
- Ahmad: data pipeline and validation
- Waleed: local inference, demo, video, documentation

## Attribution

- Base model: Qwen3, Apache-2.0, from Qwen (https://huggingface.co/Qwen).
- Training: Unsloth (https://unsloth.ai/docs).
- Data: SVG-Stack captions derived from StarVector (https://github.com/joanrod/star-vector).
- Runtime: Ollama (https://ollama.com).
