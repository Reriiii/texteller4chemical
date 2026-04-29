# texteller4chemical

Fine-tune and domain-adapt pretrained TexTeller for handwritten chemical formula and chemical structure recognition on EDU-CHEMC / EDU-CHMEC_MM23.

The project baseline is:

```text
handwritten chemical image -> pretrained TexTeller -> chemical markup sequence
```

## Scope

- Default model: `OleehyO/TexTeller`.
- Default target: `ssml_sd`.
- Default image format: grayscale `448x448x1`.
- The baseline does not require Tex80M.
- The baseline does not train TexTeller from scratch.
- `ssml_rcgd` is not used for the baseline sequence decoder.

## Setup

```bash
uv sync
```

For Kaggle or other managed GPU environments, avoid reinstalling PyTorch unless necessary:

```bash
pip install -r requirements-kaggle.txt
pip install -e . --no-deps
```

## Prepare Data

```bash
uv run python scripts/prepare_edu_chemc.py \
  --src_dir /path/to/EDU-CHMEC_MM23 \
  --out_dir data/processed/edu_chemc \
  --target_field ssml_sd
```

For read-only dataset mounts, use references instead of copying images:

```bash
uv run python scripts/prepare_edu_chemc.py \
  --src_dir /path/to/EDU-CHMEC_MM23 \
  --out_dir data/processed/edu_chemc \
  --target_field ssml_sd \
  --copy_mode reference
```

## Analyze Targets

```bash
uv run python scripts/analyze_targets.py \
  --metadata data/processed/edu_chemc/train/metadata.jsonl
```

## Check Tokenizer Coverage

```bash
uv run python scripts/analyze_tokenizer_coverage.py \
  --metadata data/processed/edu_chemc/train/metadata.jsonl \
  --pretrained_model_name_or_path OleehyO/TexTeller
```

To extend the tokenizer:

```bash
uv run python scripts/analyze_tokenizer_coverage.py \
  --metadata data/processed/edu_chemc/train/metadata.jsonl \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --extend_tokenizer \
  --vocab_file /path/to/EDU-CHEMC.vocab \
  --output_tokenizer_dir outputs/tokenizer_edu_chemc
```

## Train

```bash
uv run accelerate launch scripts/train.py \
  --config configs/train_edu_chemc.yaml \
  --dataset_dir data/processed/edu_chemc \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --output_dir outputs/runs/edu_chemc_texteller
```

Kaggle configs:

```text
configs/train_edu_chemc_kaggle.yaml
configs/train_edu_chemc_kaggle_fast.yaml
```

To train with an extended tokenizer:

```bash
uv run accelerate launch scripts/train.py \
  --config configs/train_edu_chemc.yaml \
  --dataset_dir data/processed/edu_chemc \
  --pretrained_model_name_or_path OleehyO/TexTeller \
  --tokenizer_path outputs/tokenizer_edu_chemc \
  --output_dir outputs/runs/edu_chemc_texteller
```

## Evaluate

```bash
uv run python scripts/evaluate.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --dataset_dir data/processed/edu_chemc \
  --split test \
  --output_csv outputs/eval_predictions.csv
```

Current built-in metrics:

- exact match
- whitespace-normalized exact match
- token edit distance
- normalized token edit distance
- character edit distance
- average target length
- average prediction length

For paper-style graph matching evaluation, see `GRAPH_MATCHING_EVALUATION_AGENT_GUIDE.md`.

If GraphMatchingTool is available locally:

```bash
uv run python scripts/evaluate.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --dataset_dir data/processed/edu_chemc \
  --split test \
  --graph_eval \
  --graph_matching_tool_dir external/GraphMatchingTool \
  --graph_label_key ssml_normed
```

## Predict

```bash
uv run python scripts/predict.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --image_path /path/to/image.png
```

With beam search:

```bash
uv run python scripts/predict.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --image_path /path/to/image.png \
  --num_beams 4 \
  --max_new_tokens 512
```

## Outputs

Common output locations:

```text
data/processed/edu_chemc/
outputs/reports/
outputs/runs/edu_chemc_texteller/
outputs/eval_predictions.csv
```

## Notes

- Keep `ssml_sd` as the default training target.
- Use `ssml_normed` only when required by graph matching evaluation.
- Preserve grayscale `448x448x1` preprocessing unless the model checkpoint requires a different format.
- If the tokenizer is extended, decoder embeddings and the output projection must be resized.
- If GPU memory is limited, use a smaller per-device batch size, gradient accumulation, encoder freezing, and LoRA.
