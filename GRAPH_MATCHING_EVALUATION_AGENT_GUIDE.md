# Agent Guide: Paper-Style Graph Matching Evaluation

This guide describes how to implement paper-style evaluation for EDU-CHEMC/CROCS using the official graph matching tool.

References:

- Graph Matching Tool: https://github.com/crocs-ifly-ustc/GraphMatchingTool
- CROCS task instructions: https://crocs-ifly-ustc.github.io/crocs/task1_instructions.html
- EDU-CHEMC repository: https://github.com/iFLYTEK-CV/EDU-CHEMC

## Goal

Add optional graph matching evaluation to `scripts/evaluate.py` while preserving the current string-level metrics.

Required graph metrics:

- `graph_em`: Exact Match from GraphMatchingTool metric `struct.line`.
- `graph_structure_em`: Structure Exact Match from metric `struct`.
- `graph_base_sent_acc`: string edit-distance sentence accuracy from metric `base`, for reference only.

## Constraints

- Keep `OleehyO/TexTeller` as the default pretrained model.
- Keep `ssml_sd` as the default training target.
- Do not switch the baseline to `ssml_rcgd`.
- Do not require Tex80M.
- Do not train from scratch by default.
- Preserve grayscale `448x448x1` preprocessing.
- Do not hard-code local paths.
- Keep scripts runnable with `uv run python ...`.

## Important Target Difference

The project baseline trains on `ssml_sd`, but GraphMatchingTool expects label strings in `ssml_normed`.

Do not silently change the project default from `ssml_sd` to `ssml_normed`. Instead:

1. Make graph evaluation optional.
2. Use `targets.ssml_normed` when it is present in metadata.
3. Raise a clear error when graph evaluation is requested but `ssml_normed` is missing.
4. Allow the user to prepare a dataset with `--target_field ssml_normed` when they want official-style evaluation.

## GraphMatchingTool Input Format

The tool expects two text files:

```text
rec: <img_name>\t<prediction_string>
lab: <img_name>\t<label_string>
```

The label file must use `ssml_normed`.

Terminal output includes:

- `base`: direct string edit-distance metric, reference only.
- `struct`: Structure EM.
- `struct.line`: full Exact Match.

The per-sample output file contains five tab-separated fields:

```text
image_name    structure_error    em_error    label_string    prediction_string
```

For `structure_error` and `em_error`, `0` means correct and `1` means incorrect.

## Recommended CLI

Add these arguments to `scripts/evaluate.py`:

```text
--graph_eval
--graph_matching_tool_dir PATH
--graph_label_key ssml_normed
--graph_num_workers 8
--graph_output_txt PATH
--graph_keep_temp
```

Example:

```bash
uv run python scripts/evaluate.py \
  --model_ckpt outputs/runs/edu_chemc_texteller/best \
  --dataset_dir data/processed/edu_chemc \
  --split test \
  --graph_eval \
  --graph_matching_tool_dir external/GraphMatchingTool \
  --graph_label_key ssml_normed
```

Do not clone or vendor GraphMatchingTool automatically from code. Accept it as an external directory.

## Files To Update

### `scripts/prepare_edu_chemc.py`

Metadata should keep the selected training target and the original available targets.

Recommended metadata shape:

```json
{
  "image_path": "...",
  "target": "<selected target_field>",
  "target_field": "ssml_sd",
  "targets": {
    "ssml_sd": "...",
    "ssml_normed": "...",
    "chemfig": "..."
  }
}
```

Requirements:

- Keep `--target_field ssml_sd` as the default.
- Fail clearly if the selected training target is missing.
- Keep preparing data even if `ssml_normed` is missing.
- Let graph evaluation report the missing graph label later.

### `src/chemtexteller/data.py`

Evaluation needs access to:

- `image_path`
- `image_name`
- `target`
- `targets`

Keep training batches compatible with `Seq2SeqTrainer`. If string metadata cannot be collated into training tensors, only preserve it in evaluation-specific paths.

### `scripts/evaluate.py`

When `--graph_eval` is disabled, behavior should remain unchanged.

When `--graph_eval` is enabled:

1. Validate that `graph_matching_tool_dir/eval.py` exists.
2. Validate that each sample has the requested graph label.
3. Generate predictions as usual.
4. Combine all distributed evaluation rows on the main process.
5. Write temporary GraphMatchingTool input files:
   - `<output_stem>.graph_rec.txt`
   - `<output_stem>.graph_lab.txt`
   - `<output_stem>.graph_result.txt`
6. Run GraphMatchingTool with `subprocess.run`.
7. Parse stdout.
8. Add graph metrics to the metrics JSON.

Use list-style subprocess arguments:

```python
cmd = [
    sys.executable,
    str(graph_tool_dir / "eval.py"),
    "-rec",
    str(rec_path),
    "-lab",
    str(lab_path),
    "-output",
    str(result_path),
    "-num_workers",
    str(args.graph_num_workers),
]
```

Do not use shell strings or shell pipelines.

### `src/chemtexteller/graph_matching_eval.py`

Prefer a small helper module instead of putting all graph-evaluation logic in `scripts/evaluate.py`.

Suggested helpers:

- `write_graph_matching_files(...)`
- `run_graph_matching_tool(...)`
- `parse_graph_matching_stdout(...)`
- `GraphMatchingResult`

## Metrics JSON

Extend the existing metrics JSON with:

```json
{
  "graph_em": 0.156804,
  "graph_structure_em": 0.330993,
  "graph_base_sent_acc": 0.0,
  "graph_matching_tool_dir": "external/GraphMatchingTool",
  "graph_label_key": "ssml_normed"
}
```

## Distributed Evaluation

If `accelerate launch scripts/evaluate.py` is used:

- Each rank may generate predictions for its shard.
- The main rank must combine all rows.
- GraphMatchingTool must run only once on the main rank after combination.
- Do not average per-rank graph metrics.

## Dependencies

GraphMatchingTool may require:

```bash
pip install python-Levenshtein six opencv-python-headless
```

In managed GPU environments, avoid dependency commands that reinstall PyTorch unless that is intentional.

## Validation Checklist

- `python -m py_compile` passes for modified Python files.
- `scripts/evaluate.py` still works without `--graph_eval`.
- Missing `--graph_matching_tool_dir` produces a clear error.
- Missing `ssml_normed` labels produce a clear error.
- Generated `rec` and `lab` files use `<img_name>\t<string>`.
- GraphMatchingTool stdout is parsed into `graph_em` and `graph_structure_em`.
- Multi-GPU evaluation runs graph matching only on the combined prediction set.
- Default target remains `ssml_sd`.
- `scripts/predict.py` behavior is unchanged.

## Parser Smoke Test

The parser should handle output like:

```text
------ metric struct ------
sent acc = 33.0993 %( 990/2991 )
------ metric struct.line ------
sent acc = 15.6804 %( 469/2991 )
```

Expected values:

```text
graph_structure_em = 0.330993
graph_em = 0.156804
```

## Common Pitfalls

- Plain string exact match underestimates valid Chemfig/SSML predictions because one structure can have multiple equivalent serializations.
- GraphMatchingTool expects `ssml_normed`; using `ssml_sd` directly can produce invalid or misleading scores.
- Structure EM is not a replacement for EM. The ranking protocol prioritizes EM and uses Structure EM as an auxiliary metric.
- Predictions without the expected `\chemfig { ... }` wrapper may fail graph parsing.
- Newlines inside predictions or labels must not corrupt the tab-separated input files.
- Absolute image paths are unnecessary for metric computation unless visualization is enabled.

## Definition Of Done

The task is complete when one evaluation command can produce:

- The existing prediction CSV.
- The existing string-level metrics.
- GraphMatchingTool per-sample output.
- `graph_em` and `graph_structure_em` in the metrics JSON.
- Clear logs showing `EM(struct.line)` and `Structure EM(struct)`.
