# AGENTS.md

Guidance for coding agents working on this repository.

## Project Goal

This project fine-tunes/domain-adapts pretrained TexTeller for handwritten chemical formula / chemical structure recognition on EDU-CHEMC / EDU-CHMEC_MM23.

The intended task is:

```text
handwritten chemical image -> pretrained TexTeller -> chemical markup sequence
```

Default dataset location on the user's machine:

```text
F:\dataset\EDU-CHMEC_MM23
```

## Non-Negotiable Constraints

- Do not ask the user to download or use Tex80M.
- Do not train TexTeller from scratch by default.
- Treat this as fine-tuning / domain adaptation from pretrained TexTeller.
- The default model is `OleehyO/TexTeller` from Hugging Face.
- The default target is `ssml_sd`.
- Do not use `ssml_rcgd` for the baseline sequence decoder unless explicitly requested.
- EDU-CHEMC targets are chemical markup, not normal math LaTeX.

## Important Model Facts

`OleehyO/TexTeller` is a Hugging Face `VisionEncoderDecoderModel`.

Observed model properties:

- encoder: ViT
- decoder: TrOCR-like
- input image: grayscale
- channels: 1
- image size: 448 x 448
- default config file: `configs/train_edu_chemc.yaml`

Keep the default preprocessing aligned with this:

```yaml
image_size:
  height: 448
  width: 448
  channels: 1
```

## Project Structure

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

## Usual Workflow

Prepare data:

```bash
uv run python scripts/prepare_edu_chemc.py \
  --src_dir F:/dataset/EDU-CHMEC_MM23 \
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

## CUDA Notes

The user's machine has an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM. `nvidia-smi` reports driver CUDA support 13.0 and `nvcc` 13.0.

The uv environment may still install CPU-only PyTorch unless explicitly changed.

Check CUDA:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Preferred PyTorch CUDA wheel for this machine:

```bash
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Fallback:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Because VRAM is only 4 GB, prefer:

- batch size 1
- gradient accumulation
- freeze encoder
- LoRA where possible
- no fp16 unless verified stable
- reduced max target length if needed

## Kaggle Notes

Kaggle is a good target for longer runs because it usually provides larger GPUs than the user's local 4 GB RTX 3050 Laptop GPU.

Do not install project dependencies with plain `pip install -e .` on Kaggle unless you intentionally want pip to resolve all dependencies again. It may reinstall `torch` and break CUDA.

Use:

```bash
pip install -r requirements-kaggle.txt
pip install -e . --no-deps
```

Use Kaggle config:

```text
configs/train_edu_chemc_kaggle.yaml
```

For the first run, prefer:

```text
configs/train_edu_chemc_kaggle_fast.yaml
```

The normal Kaggle config is intentionally longer; the fast config is the safer notebook-time baseline.

Kaggle paths usually look like:

```text
/kaggle/input/<dataset-name>
/kaggle/working
```

When preparing data on Kaggle, use:

```bash
--copy_mode reference
```

This avoids copying all images from `/kaggle/input` into `/kaggle/working`. The metadata will contain absolute image paths and the dataset loader supports those paths.

## Tokenizer And Target Guidance

Default target:

```text
ssml_sd
```

Why:

- It is already a sequence target.
- It is appropriate for a normal encoder-decoder baseline.
- It avoids the conditional/reconnection mechanism required by `ssml_rcgd`.

Tokenizer risks:

- Chemical tokens such as `branch`, `?[a]`, `-[:30]`, `=[:180]`, `<:[:270]`, atoms, rings, and ChemFig-like macros may be poorly covered by the original math tokenizer.
- Always run tokenizer coverage before serious training.
- If extending the tokenizer, make sure decoder embeddings and output head are resized.

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

Do not hard-code an unverified TexTeller internal API. If the HF model loads, prefer the standard Hugging Face path.

## Code Quality Rules

- Use pathlib.
- Keep scripts runnable with `uv run python ...`.
- Do not hard-code local paths except examples in docs.
- Keep `ssml_sd` as the default.
- Do not silently swallow dataset/model loading errors.
- Preserve clear error messages for missing checkpoints, tokenizer resize failures, and bad dataset rows.
- Prefer small, focused changes.
- Avoid notebook-only workflows.

## Common Failure Modes

- CPU-only PyTorch installed even though CUDA exists.
- Image channel mismatch: TexTeller expects 1-channel grayscale input.
- Tokenizer extended but decoder embedding/output head not resized.
- `ssml_rcgd` used with a normal sequence decoder.
- Chemical target truncated by too-small `max_target_length`.
- Augmentation too strong and changes bond topology.
- 4 GB VRAM OOM during full fine-tuning.
