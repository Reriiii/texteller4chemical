# TexTeller-RFL Research Pipeline

This document separates the stable TexTeller sequence baseline from the
graph-aware RFL research branch. The goal is to reuse verified code from
TexTeller and RFL-MSD where possible, and only add local glue code at the
boundaries.

## Step 0: Repo Cleanup Boundary

The active production-style baseline remains:

```text
EDU-CHEMC image -> TexTeller full fine-tune -> ssml_graph_norm -> GraphMatchingTool
```

Keep this path on `configs/train_edu_chemc.yaml` and the prepared dataset
`data/processed/edu_chemc_graph_norm`.

The research branch is:

```text
EDU-CHEMC ssml/chemfig label -> RFL-MSD converter -> RFL target + branch/ring labels
```

The branch writes a new prepared dataset instead of mutating the graph-norm
dataset. This makes experiments reversible and avoids silently replacing labels.

## Step 1: RFL Label Materialization

We reuse the official RFL-MSD converter in `external/RFL-MSD/RFL/RFL.py` through
`src/chemtexteller/rfl_adapter.py`. The adapter calls `cs_main(...)`, keeps the
serialized RFL token sequence, and also preserves the auxiliary arrays used by
RFL-MSD for branch/ring reconstruction.

Recommended smoke command:

```bash
uv run python scripts/create_rfl_target_dataset.py \
  --dataset_dir data/processed/edu_chemc_graph_norm \
  --out_dir outputs/smoke_edu_chemc_rfl \
  --splits validation \
  --max_samples_per_split 20 \
  --source_key ssml_sd \
  --target_field ssml_rfl \
  --rfl_tool_dir external/RFL-MSD \
  --overwrite \
  --on_error raise
```

Full materialization:

```bash
uv run python scripts/create_rfl_target_dataset.py \
  --dataset_dir data/processed/edu_chemc_graph_norm \
  --out_dir data/processed/edu_chemc_rfl \
  --splits train validation test \
  --source_key ssml_sd \
  --target_field ssml_rfl \
  --rfl_tool_dir external/RFL-MSD \
  --overwrite \
  --on_error raise
```

Output rows keep the original metadata and set:

```text
target = serialized RFL token sequence
target_field = ssml_rfl
targets.ssml_rfl = serialized RFL token sequence
rfl.tokens = RFL-MSD token list
rfl.branch_info = RFL-MSD branch labels
rfl.ring_branch_info = RFL-MSD ring/branch labels
rfl.cond_data = RFL-MSD condition/non-structure flags
rfl.ring_count = number of rings found by RFL-MSD, when the cloned converter supports it
```

The script defaults to `--on_error raise` because a fallback target would mix
RFL and non-RFL labels in one dataset. Use `--on_error skip` only for conversion
coverage analysis.

## Next Steps

1. Add a validation script that reconstructs RFL back to ChemFig/SSML with
   RFL-MSD and checks graph equivalence with GraphMatchingTool.
2. Train a low-risk RFL-lite TexTeller seq2seq run on `targets.ssml_rfl`.
3. If RFL-lite helps, extract TexTeller's visual encoder and add a chemical/RFL
   decoder.
4. Only after the decoder is stable, add the branch-connection head using
   `rfl.branch_info`, `rfl.ring_branch_info`, and `rfl.cond_data`.
