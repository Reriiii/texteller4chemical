# AGENTS.md

Guidance for coding agents working on this repository.

## Project Goal

This project fine-tunes and domain-adapts pretrained TexTeller for handwritten chemical formula and chemical structure recognition on EDU-CHEMC / EDU-CHMEC_MM23.

The intended pipeline is:

```text
handwritten chemical image -> pretrained TexTeller -> chemical markup sequence
```

## Non-Negotiable Constraints

- Do not require Tex80M.
- Do not train TexTeller from scratch by default.
- Treat this as fine-tuning / domain adaptation from pretrained TexTeller.
- Use `OleehyO/TexTeller` as the default model.
- Use `ssml_sd` as the default target.
- Do not use `ssml_rcgd` for the baseline sequence decoder unless explicitly requested.
- EDU-CHEMC targets are chemical markup, not ordinary math LaTeX.

## Model Facts

`OleehyO/TexTeller` is loaded through Hugging Face-compatible APIs when possible.

Observed default properties:

- encoder: ViT
- decoder: TrOCR-like
- input image: grayscale
- channels: 1
- image size: `448x448`
- default config: `configs/train_edu_chemc.yaml`

Keep preprocessing aligned with:

```yaml
image_size:
  height: 448
  width: 448
  channels: 1
```

## Repository Structure

```text
configs/train_edu_chemc.yaml
scripts/prepare_edu_chemc.py
scripts/analyze_targets.py
scripts/analyze_tokenizer_coverage.py
scripts/train.py
scripts/evaluate.py
scripts/predict.py
src/chemtexteller/data.py
src/chemtexteller/transforms.py
src/chemtexteller/tokenizer_utils.py
src/chemtexteller/model_loader.py
src/chemtexteller/metrics.py
src/chemtexteller/utils.py
```

## Standard Workflow

Prepare data:

```bash
uv run python scripts/prepare_edu_chemc.py \
  --src_dir /path/to/EDU-CHMEC_MM23 \
  --out_dir data/processed/edu_chemc \
  --target_field ssml_sd
```

Analyze targets:

```bash
uv run python scripts/analyze_targets.py \
  --metadata data/processed/edu_chemc/train/metadata.jsonl
```

Analyze tokenizer coverage:

```bash
uv run python scripts/analyze_tokenizer_coverage.py \
  --metadata data/processed/edu_chemc/train/metadata.jsonl \
  --pretrained_model_name_or_path OleehyO/TexTeller
```

Fine-tune:

```bash
uv run accelerate launch scripts/train.py \
  --config configs/train_edu_chemc.yaml \
  --dataset_dir data/processed/edu_chemc \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --output_dir outputs/runs/edu_chemc_texteller
```

Evaluate:

```bash
uv run python scripts/evaluate.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --dataset_dir data/processed/edu_chemc \
  --split test
```

Predict one image:

```bash
uv run python scripts/predict.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --image_path /path/to/image.png
```

## Environment Notes

The runtime may install CPU-only PyTorch unless CUDA wheels are selected explicitly. Check CUDA with:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

When VRAM is limited, prefer:

- small per-device batch size
- gradient accumulation
- encoder freezing
- LoRA
- shorter target length only after checking truncation risk

## Managed GPU Environments

Avoid reinstalling PyTorch accidentally in managed notebook environments.

Use:

```bash
pip install -r requirements-kaggle.txt
pip install -e . --no-deps
```

For read-only dataset mounts:

```bash
uv run python scripts/prepare_edu_chemc.py \
  --src_dir /path/to/EDU-CHMEC_MM23 \
  --out_dir data/processed/edu_chemc \
  --target_field ssml_sd \
  --copy_mode reference
```

Use:

```text
configs/train_edu_chemc_kaggle.yaml
configs/train_edu_chemc_kaggle_fast.yaml
```

## Tokenizer And Target Guidance

Default target:

```text
ssml_sd
```

Why:

- It is already a sequence target.
- It fits a standard encoder-decoder baseline.
- It avoids the conditional/reconnection mechanism required by `ssml_rcgd`.

Tokenizer risks:

- Chemical tokens such as `branch`, `?[a]`, `-[:30]`, `=[:180]`, `<:[:270]`, atoms, rings, and ChemFig-like macros may be poorly covered by the original math tokenizer.
- Always run tokenizer coverage before serious training.
- If extending the tokenizer, make sure decoder embeddings and output projection are resized.

## Model Loader Guidance

Model loading is intentionally defensive in:

```text
src/chemtexteller/model_loader.py
```

It tries:

1. Hugging Face tokenizer / processor
2. `AutoModelForVision2Seq` when available
3. `VisionEncoderDecoderModel`
4. optional package `texteller`
5. PEFT adapters when loading LoRA checkpoints

Do not hard-code an unverified TexTeller internal API. If the Hugging Face model loads, prefer the standard Hugging Face path.

## Code Quality Rules

- Use `pathlib`.
- Keep scripts runnable with `uv run python ...`.
- Do not hard-code local paths except generic examples in docs.
- Keep `ssml_sd` as the default.
- Do not silently swallow dataset/model loading errors.
- Preserve clear error messages for missing checkpoints, tokenizer resize failures, and bad dataset rows.
- Prefer small, focused changes.
- Avoid notebook-only workflows.

## Common Failure Modes

- CPU-only PyTorch installed even though CUDA is available.
- Image channel mismatch: TexTeller expects 1-channel grayscale input.
- Tokenizer extended but decoder embedding/output projection not resized.
- `ssml_rcgd` used with a normal sequence decoder.
- Chemical target truncated by too-small `max_target_length`.
- Augmentation too strong and changes bond topology.
- Full fine-tuning OOM on limited VRAM.
