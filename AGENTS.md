# AGENTS.md

## Defaults
- This repo fine-tunes/domain-adapts pretrained TexTeller for EDU-CHEMC / EDU-CHMEC_MM23 handwritten chemical images to chemical markup.
- Default base model is `OleehyO/TexTeller`; do not require Tex80M or train from scratch unless the user explicitly changes scope.
- Current graph-eval target is `ssml_normed`; `ssml_sd` is only a legacy/simple sequence baseline, and `ssml_rcgd` is not suitable for the normal sequence decoder.
- Keep TexTeller preprocessing at grayscale `448x448x1`; configs set `image_size.height: 448`, `width: 448`, `channels: 1`.
- Use `configs/train_edu_chemc.yaml` for the active experiment: `max_target_length: 768`, bf16, encoder frozen, LoRA r32 over encoder+decoder, length-balanced sampling, 30 epochs.
- `configs/train_edu_chemc_baseline.yaml` is the older 20-epoch decoder-only LoRA r16 baseline for comparison.

## Data Pipeline
- User target runtime is a Linux GPU server with this flow: download Hugging Face data -> prepare data -> train -> evaluate with GraphMatchingTool.
- Hugging Face dataset id is `ConstantHao/EDU-CHEMC_MM23`; use `datasets.load_dataset("ConstantHao/EDU-CHEMC_MM23")` when adding/using a download stage.
- The HF dataset card exposes splits `train`, `val`, `test` and fields `image`, `chemfig`, `ssml_sd`, `ssml_normed`, `ssml_rcgd`, `image_path`; total size is about 53k rows / 28.9 GB.
- `scripts/run_edu_chemc_pipeline.py` is the end-to-end Linux GPU launcher for download -> materialize -> analyze -> train -> graph-evaluate.
- `scripts/materialize_hf_edu_chemc.py` uses `load_dataset`, preserves official HF splits, and maps HF `val` to repo split name `validation`.
- Cache-only HF download command: `uv run python -c "from datasets import load_dataset; load_dataset('ConstantHao/EDU-CHEMC_MM23')"`; this does not create the prepared imagefolder metadata used by training.
- `scripts/prepare_edu_chemc.py` is only for a local image tree with same-stem `.json` annotations; it creates its own random split, so do not use it when official HF splits matter.
- Graph eval needs `targets.ssml_normed` in prepared `metadata.jsonl`; re-run prepare with `--target_field ssml_normed` if `evaluate.py --graph_eval` complains about missing graph labels.
- For local-tree prepare on read-only mounts, `--copy_mode reference` writes absolute source image paths instead of copying images.

## Setup And Checks
- Standard local setup is `uv sync`; `pyproject.toml` routes `torch`/`torchvision` to the explicit `pytorch-cu130` uv index.
- On managed GPU servers, avoid accidentally replacing the system CUDA PyTorch; install through the server workflow, then use `pip install -e . --no-deps` for this repo if needed.
- Check CUDA before training: `uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"`.
- `training.bf16: true` requires a bf16-capable CUDA GPU; otherwise switch the config to fp16 or fp32 before launch.

## Core Commands
Full server pipeline:

```bash
uv run python scripts/run_edu_chemc_pipeline.py \
  --stages all \
  --graph_matching_tool_dir external/GraphMatchingTool
```

Materialize current graph target from Hugging Face:

```bash
uv run python scripts/materialize_hf_edu_chemc.py \
  --dataset_id ConstantHao/EDU-CHEMC_MM23 \
  --out_dir data/processed/edu_chemc_normed \
  --target_field ssml_normed
```

Analyze before serious training:

```bash
uv run python scripts/analyze_targets.py \
  --metadata data/processed/edu_chemc_normed/train/metadata.jsonl

uv run python scripts/analyze_tokenizer_coverage.py \
  --metadata data/processed/edu_chemc_normed/train/metadata.jsonl \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --max_decoder_length 768
```

Train on a single Linux GPU node:

```bash
uv run accelerate launch \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train.py \
  --config configs/train_edu_chemc.yaml \
  --dataset_dir data/processed/edu_chemc_normed \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --output_dir outputs/runs/edu_chemc_texteller_normed_len768_r32_all_lora_balanced_30ep
```

Evaluate with paper-style graph matching:

```bash
uv run python scripts/evaluate.py \
  --model_ckpt outputs/runs/edu_chemc_texteller_normed_len768_r32_all_lora_balanced_30ep/best \
  --dataset_dir data/processed/edu_chemc_normed \
  --split test \
  --batch_size 8 \
  --num_beams 1 \
  --max_new_tokens 768 \
  --dtype bf16 \
  --output_csv outputs/eval_normed_len768_r32_all_lora_balanced_30ep_test_greedy.csv \
  --graph_eval \
  --graph_matching_tool_dir external/GraphMatchingTool \
  --graph_label_key ssml_normed \
  --graph_num_workers 8
```

## GraphMatchingTool
- `scripts/evaluate.py --graph_eval` requires `external/GraphMatchingTool/eval.py`; `external/` is gitignored, so clone/install it locally on the server.
- GraphMatchingTool is invoked with the same Python executable running `evaluate.py`; install its dependencies in that environment, for example `python-Levenshtein` if the tool needs it.
- Metric mapping in `src/chemtexteller/graph_matching_eval.py`: `struct.line -> graph_em`, `struct -> graph_structure_em`, `base -> graph_base_sent_acc`.
- On Windows use `--graph_num_workers 0` to avoid multiprocessing issues; on Linux GPU servers the default/`8` workers is the intended path.

## Model And Tokenizer Quirks
- `src/chemtexteller/model_loader.py` intentionally tries Hugging Face tokenizer/processor/model classes, optional `texteller`, then PEFT adapters; do not hard-code an unverified TexTeller internal API.
- PEFT `best/` checkpoints can be adapter-only; evaluation loads the base model from `adapter_config.json` and merges LoRA unless `--no_merge_lora` is passed.
- Run tokenizer coverage before long runs; chemical tokens such as `branch`, `?[a]`, angle bonds, atoms, rings, and ChemFig-like macros may be poorly covered by the math tokenizer.
- If extending the tokenizer, the decoder embeddings/output projection must resize successfully; `resize_token_embeddings_if_needed` raises if that cannot happen.
- `data.target_length_policy: filter` drops samples above `max_target_length`; do not lower `max_target_length` without checking truncation/coverage stats.

## Verification And Artifacts
- No test suite, lint config, formatter config, CI, or pre-commit config is present; verify changes with the smallest relevant script command, often `evaluate.py --max_samples N` after a checkpoint exists.
- Training writes logs to `logs/<run_name>_<timestamp>.log` and trainer events to `logs/<run_name>_<timestamp>.trainer_events.jsonl`.
- Keep generated `data/`, `outputs/`, `logs/`, `external/`, `.uv-cache/`, `.uv-python/`, and `.venv/` out of git.
- Keep scripts runnable as `uv run python ...`, use `pathlib`, and avoid hard-coded local paths except generic examples in docs.
