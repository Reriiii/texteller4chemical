# Error Visualization Report

- Samples: 2991
- String EM: 10.23%
- Graph EM: 48.71%
- Structure EM: 55.30%
- Mean token edit distance: 13.46
- Mean normalized token edit distance: 0.190
- Long samples >128 tokens: 364
- Long-sample mean token edit distance: 29.85

## Generated Figures

- `01_error_overview.png`
- `02_token_error_by_position.png`
- `03_top_token_error_types.png`
- `04_error_by_target_length.png`
- `05_error_by_graph_depth.png`
- `06_graph_em_heatmap_length_branch.png`
- `07_graph_outcome_by_length.png`
- `08_compare_csv_improvement.png`

Interpretation tip: if token edit error and graph failure rise with target length, the bottleneck is likely long-reaction representation/decoding rather than only local OCR quality.
