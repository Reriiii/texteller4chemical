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
- Use `ssml_normed` for current paper-style graph-based EDU-CHEMC experiments.
- Treat `ssml_sd` as a legacy/simple sequence baseline target, not the current best target.
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
- current strongest local config: `configs/train_edu_chemc_normed_20ep_local_16gb.yaml`

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
configs/train_edu_chemc_normed_20ep_local_16gb.yaml
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

Prepare data for the current graph-based experiment:

```bash
uv run python scripts/prepare_edu_chemc.py \
  --src_dir /path/to/EDU-CHMEC_MM23 \
  --out_dir data/processed/edu_chemc_normed \
  --target_field ssml_normed
```

Analyze targets:

```bash
uv run python scripts/analyze_targets.py \
  --metadata data/processed/edu_chemc_normed/train/metadata.jsonl
```

Analyze tokenizer coverage:

```bash
uv run python scripts/analyze_tokenizer_coverage.py \
  --metadata data/processed/edu_chemc_normed/train/metadata.jsonl \
  --pretrained_model_name_or_path OleehyO/TexTeller
```

Fine-tune:

```bash
uv run accelerate launch \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train.py \
  --config configs/train_edu_chemc_normed_20ep_local_16gb.yaml \
  --dataset_dir data/processed/edu_chemc_normed \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --output_dir outputs/runs/edu_chemc_texteller_normed_lora16_bf16_20ep
```

Evaluate with paper-style graph matching:

```bash
uv run python scripts/evaluate.py \
  --model_ckpt outputs/runs/edu_chemc_texteller_normed_lora16_bf16_20ep/best \
  --dataset_dir data/processed/edu_chemc_normed \
  --split test \
  --batch_size 8 \
  --num_beams 1 \
  --max_new_tokens 768 \
  --dtype bf16 \
  --output_csv outputs/eval_normed_lora16_20ep_test_greedy.csv \
  --graph_eval \
  --graph_matching_tool_dir external/GraphMatchingTool \
  --graph_label_key ssml_normed \
  --graph_num_workers 0 \
  --graph_keep_temp
```

Predict one image:

```bash
uv run python scripts/predict.py \
  --model_ckpt outputs/runs/edu_chemc_texteller_normed_lora16_bf16_20ep/best \
  --image_path /path/to/image.png
```

Legacy `ssml_sd` baseline workflow is still useful for debugging a simpler sequence target, but it is not the current best path for graph-based EM.

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

Current recommended target for graph-based evaluation:

```text
ssml_normed
```

Why:

- GraphMatchingTool expects labels in `ssml_normed`.
- Training directly on `ssml_normed` greatly improved graph-based metrics.
- It still fits the standard encoder-decoder baseline and avoids the conditional/reconnection mechanism required by `ssml_rcgd`.

Known local results:

```text
ssml_sd 10ep LoRA r16 full test:
  Graph EM:      4.74%
  Structure EM:  6.11%

ssml_normed 20ep LoRA r16 full test, greedy:
  Graph EM:     30.08%  (1594/5299)
  Structure EM: 34.97%  (1853/5299)
  String EM:     6.61%  (350/5299)

ssml_normed 20ep LoRA r16 test500, beam3:
  Graph EM:     32.40%
  Structure EM: 37.00%
```

Tradeoffs:

- `ssml_normed` sequences are longer than `ssml_sd`.
- With TexTeller tokenizer, `ssml_normed` train targets averaged about 268 tokens vs about 182 for `ssml_sd`.
- `max_target_length: 512` truncates more `ssml_normed` samples than `ssml_sd`; consider `768` for a future experiment if runtime/VRAM allow it.

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

PEFT checkpoints saved in `best/` are adapter-only directories. Loading `outputs/runs/.../best` should load the base model recorded in `adapter_config.json` and then attach the LoRA adapter.

## Evaluation Guidance

Use `scripts/evaluate.py --graph_eval` for paper-style metrics. GraphMatchingTool reports:

```text
struct.line -> graph_em
struct      -> graph_structure_em
base        -> string sentence accuracy reference
```

GraphMatchingTool is kept as a local external clone under `external/GraphMatchingTool`, which is intentionally gitignored. Install its dependencies locally as needed, for example `python-Levenshtein`.

On Windows, prefer `--graph_num_workers 0`. This avoids multiprocessing/Manager permission issues and was fast enough for the full EDU-CHEMC test set.

Greedy decoding is the full-test baseline. Beam search can help; `num_beams 3` improved test500 graph EM from `30.2%` to `32.4%` for the 20ep `ssml_normed` model.

## Logging And Artifacts

Training writes run logs to `logs/`:

```text
logs/<run_name>_<timestamp>.log
logs/<run_name>_<timestamp>.trainer_events.jsonl
```

The 20ep local config logs/evaluates/saves by epoch. Keep large generated artifacts out of git:

```text
data/
outputs/
logs/
external/
.uv-cache/
.uv-python/
.venv/
```

For a clean handoff, keep only the important adapter directory and final eval metrics, for example:

```text
outputs/runs/edu_chemc_texteller_normed_lora16_bf16_20ep/best
outputs/eval_normed_lora16_20ep_test_greedy.metrics.json
outputs/eval_normed_lora16_20ep_test_greedy.graph_result.txt
```

## Code Quality Rules

- Use `pathlib`.
- Keep scripts runnable with `uv run python ...`.
- Do not hard-code local paths except generic examples in docs.
- Do not silently switch experiments back to `ssml_sd`; use `ssml_normed` for paper-style graph EM unless the user explicitly asks for the legacy baseline.
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
