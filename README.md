# texteller4chemical

Fine-tune and domain-adapt pretrained TexTeller for handwritten chemical formula and chemical structure recognition on EDU-CHEMC / EDU-CHMEC_MM23.

```text
handwritten chemical image -> pretrained TexTeller -> chemical markup sequence
```

## Scope

- Default model: `OleehyO/TexTeller`.
- Current graph-eval target: `ssml_graph_norm`, generated from `ssml_normed` by graph-preserving bond geometry rounding.
- Legacy/simple sequence target: `ssml_sd`.
- Image format: grayscale `448x448x1`.
- The project does not require Tex80M and does not train TexTeller from scratch by default.
- `ssml_rcgd` is not used for the normal sequence-decoder baseline.

## Configs

The stable configs are:

```text
configs/train_edu_chemc.yaml
configs/train_edu_chemc_baseline.yaml
```

`configs/train_edu_chemc.yaml` is the active experiment: `max_target_length: 1024`, TexTeller-aligned preprocessing, TexTeller OCR/Augraphy augmentation, full-model fine-tuning with encoder unfrozen, bf16, and length-balanced sampling.

`configs/train_edu_chemc_baseline.yaml` preserves the earlier 20-epoch decoder-only LoRA r16 baseline for comparison.

Additional config files in `configs/` are experiment snapshots. Treat
`configs/train_edu_chemc.yaml` as the active default unless an experiment
explicitly names a different file.

## Research Pipeline Notes

The graph-aware RFL branch is documented in
`docs/TEXTELLER_RFL_PIPELINE.md`. It creates a separate RFL target dataset and
does not mutate `data/processed/edu_chemc_graph_norm`.

## Setup

```bash
uv sync
```

For managed GPU environments, avoid reinstalling PyTorch unless needed. Install project dependencies from the environment's preferred package workflow, then use `pip install -e . --no-deps` for this repo.

## Train Then Evaluate

On a Linux GPU server with data already materialized, this trains the active full-model config and then evaluates the best checkpoint with GraphMatchingTool.

```bash
uv run python scripts/run_edu_chemc_pipeline.py \
  --stages train_eval \
  --graph_matching_tool_dir external/GraphMatchingTool
```

Use `--stages all` when you also want Hugging Face download, materialization, and analysis before training. For a smoke run, add `--max_samples_per_split N --eval_max_samples N` and omit `train` from `--stages` unless a checkpoint already exists.

## Prepare Data

Preferred Hugging Face path, preserving official `train`/`val`/`test` splits and mapping `val` to repo split `validation`:

```bash
uv run python scripts/materialize_hf_edu_chemc.py \
  --dataset_id ConstantHao/EDU-CHEMC_MM23 \
  --out_dir data/processed/edu_chemc_graph_norm \
  --target_field ssml_graph_norm
```

Legacy local image + same-stem JSON tree path:

```bash
uv run python scripts/prepare_edu_chemc.py \
  --src_dir /path/to/EDU-CHMEC_MM23 \
  --out_dir data/processed/edu_chemc_graph_norm \
  --target_field ssml_graph_norm
```

For read-only local-tree dataset mounts, add:

```bash
--copy_mode reference
```

## Analyze

```bash
uv run python scripts/analyze_targets.py \
  --metadata data/processed/edu_chemc_graph_norm/train/metadata.jsonl

uv run python scripts/analyze_tokenizer_coverage.py \
  --metadata data/processed/edu_chemc_graph_norm/train/metadata.jsonl \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --max_decoder_length 1024
```

Validate that `ssml_graph_norm` still preserves the original `ssml_normed` graph semantics before a serious run:

```bash
uv run python scripts/validate_graph_norm.py \
  --dataset_dir data/processed/edu_chemc_graph_norm \
  --splits train validation test \
  --source_key ssml_normed \
  --normalized_key ssml_graph_norm \
  --graph_matching_tool_dir external/GraphMatchingTool \
  --graph_num_workers 8
```

## Train

```bash
uv run accelerate launch \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  scripts/train.py \
  --config configs/train_edu_chemc.yaml \
  --dataset_dir data/processed/edu_chemc_graph_norm \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --output_dir outputs/runs/edu_chemc_texteller_graph_norm_full_model_bf16_30ep
```

To reproduce the previous baseline, swap the config and output directory:

```bash
--config configs/train_edu_chemc_baseline.yaml
--output_dir outputs/runs/edu_chemc_texteller_normed_lora16_bf16_20ep
```

## Evaluate

```bash
uv run python scripts/evaluate.py \
  --model_ckpt outputs/runs/edu_chemc_texteller_graph_norm_full_model_bf16_30ep/best \
  --dataset_dir data/processed/edu_chemc_graph_norm \
  --split test \
  --batch_size 8 \
  --num_beams 1 \
  --max_new_tokens 1024 \
  --dtype bf16 \
  --output_csv outputs/eval_graph_norm_full_model_bf16_30ep_test_greedy.csv \
  --graph_eval \
  --graph_matching_tool_dir external/GraphMatchingTool \
  --graph_label_key ssml_graph_norm \
  --graph_num_workers 8
```

Built-in metrics include exact match, normalized exact match, token edit distance, normalized token edit distance, character edit distance, and length stats. With `--graph_eval`, GraphMatchingTool also reports graph EM and structure EM.

## Predict

```bash
uv run python scripts/predict.py \
  --model_ckpt outputs/runs/edu_chemc_texteller_graph_norm_full_model_bf16_30ep/best \
  --image_path /path/to/image.png \
  --max_new_tokens 1024
```

## Outputs

Generated artifacts stay out of git:

```text
data/
outputs/
logs/
external/
.uv-cache/
.uv-python/
.venv/
```
