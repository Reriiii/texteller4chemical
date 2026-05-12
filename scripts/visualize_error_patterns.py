from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOKEN_RE = re.compile(
    r"\\[A-Za-z]+|\?\[[^\]]+\]|[-=~<>|_:]*\[:\s*-?\d+\]|[A-Za-z]+|\d+|[^\s]"
)
ANCHOR_RE = re.compile(r"\?\[[^\]]+\]")
ANGLE_BOND_RE = re.compile(r"[-=~<>|_:]*\[:\s*-?\d+\]")
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize EDU-CHEMC sequence and graph-structure error patterns from "
            "evaluate.py CSV outputs."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Evaluation CSV.")
    parser.add_argument(
        "--graph_result",
        type=Path,
        default=None,
        help="Optional GraphMatchingTool result txt produced beside the CSV.",
    )
    parser.add_argument(
        "--compare_csv",
        type=Path,
        default=None,
        help="Optional older/baseline CSV to compare token edit distance against.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("visualize"))
    parser.add_argument("--position_bins", type=int, default=40)
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def token_list(text: Any) -> list[str]:
    return TOKEN_RE.findall("" if text is None else str(text))


def clean_key(value: Any) -> str:
    key = Path(str(value)).name
    return re.sub(r"\.jpg$", "", key)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def exact_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "1.0", "true", "yes"}


def max_balanced_depth(tokens: list[str], open_tok: str, close_tok: str) -> int:
    depth = 0
    max_depth = 0
    for tok in tokens:
        if tok == open_tok:
            depth += 1
            max_depth = max(max_depth, depth)
        elif tok == close_tok:
            depth = max(0, depth - 1)
    return max_depth


def structural_features(text: Any) -> dict[str, int]:
    value = "" if text is None else str(text)
    toks = token_list(value)
    return {
        "target_len": len(toks),
        "chars": len(value),
        "chemfig_count": value.count(r"\chemfig"),
        "arrow_count": value.count(r"\rightarrow")
        + value.count(r"\leftarrow")
        + value.count(r"\leftrightarrow"),
        "branch_count": toks.count("branch"),
        "ring_anchor_count": len(ANCHOR_RE.findall(value)),
        "angle_bond_count": len(ANGLE_BOND_RE.findall(value)),
        "max_brace_depth": max_balanced_depth(toks, "{", "}"),
        "max_paren_depth": max_balanced_depth(toks, "(", ")"),
        "cjk_count": len(CJK_RE.findall(value)),
        "digit_token_count": sum(tok.isdigit() for tok in toks),
        "brace_delta": value.count("{") - value.count("}"),
        "paren_delta": value.count("(") - value.count(")"),
    }


def parse_graph_result(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) != 5:
                continue
            key, struct_flag, struct_line_flag, _label, _prediction = parts
            try:
                # GraphMatchingTool writes 0 for correct and 1 for incorrect.
                struct_wrong = int(struct_flag)
                struct_line_wrong = int(struct_line_flag)
            except ValueError:
                continue
            rows.append(
                {
                    "key": key,
                    "graph_structure_ok": struct_wrong == 0,
                    "graph_em_ok": struct_line_wrong == 0,
                    "graph_structure_wrong": struct_wrong,
                    "graph_em_wrong": struct_line_wrong,
                }
            )
    return pd.DataFrame(rows)


def load_eval_frame(csv_path: Path, graph_result_path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path).fillna("")
    df["key"] = df["image_name"].map(clean_key)
    for col in [
        "token_edit_distance",
        "normalized_token_edit_distance",
        "char_edit_distance",
    ]:
        df[col] = df[col].map(to_float)
    df["exact_match_bool"] = df["exact_match"].map(exact_bool)
    feature_rows = [structural_features(value) for value in df["ground_truth"]]
    feature_df = pd.DataFrame(feature_rows)
    df = pd.concat([df, feature_df], axis=1)
    df["prediction_len"] = df["prediction"].map(lambda value: len(token_list(value)))
    df["length_ratio"] = df["prediction_len"] / df["target_len"].clip(lower=1)
    df["token_edit_rate"] = df["token_edit_distance"] / df["target_len"].clip(lower=1)
    df["structure_score"] = (
        df["target_len"]
        + 3 * df["branch_count"]
        + 3 * df["ring_anchor_count"]
        + 2 * df["angle_bond_count"]
        + 8 * df["chemfig_count"]
        + 6 * df["arrow_count"]
        + 8 * df["max_brace_depth"]
    )

    if graph_result_path is not None and graph_result_path.exists():
        graph_df = parse_graph_result(graph_result_path)
        df = df.merge(graph_df, on="key", how="left")
    else:
        df["graph_structure_ok"] = np.nan
        df["graph_em_ok"] = np.nan
    return df


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def bar_label(ax: plt.Axes, values: list[float], fmt: str = "{:.1f}") -> None:
    for idx, value in enumerate(values):
        ax.text(idx, value, fmt.format(value), ha="center", va="bottom", fontsize=9)


def plot_dashboard(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EDU-CHEMC Evaluation Error Overview", fontsize=15, fontweight="bold")

    metric_names = ["String EM", "Graph EM", "Structure EM"]
    metric_values = [
        100 * df["exact_match_bool"].mean(),
        100 * df["graph_em_ok"].mean() if "graph_em_ok" in df else np.nan,
        100 * df["graph_structure_ok"].mean() if "graph_structure_ok" in df else np.nan,
    ]
    ax = axes[0, 0]
    ax.bar(metric_names, metric_values, color=["#536dfe", "#00897b", "#f9a825"])
    ax.set_ylim(0, max(100, np.nanmax(metric_values) + 8))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Metric gap: string vs graph")
    bar_label(ax, [0 if np.isnan(v) else v for v in metric_values])

    ax = axes[0, 1]
    ax.hist(df["target_len"], bins=40, color="#455a64", alpha=0.82)
    ax.axvline(df["target_len"].median(), color="#c62828", linestyle="--", label="median")
    ax.set_title("Target token length distribution")
    ax.set_xlabel("Target tokens")
    ax.set_ylabel("Samples")
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(
        df["target_len"],
        df["normalized_token_edit_distance"],
        s=10,
        alpha=0.35,
        color="#5e35b1",
        edgecolors="none",
    )
    ax.set_title("Longer targets create larger normalized edit error")
    ax.set_xlabel("Target tokens")
    ax.set_ylabel("Normalized token edit distance")

    ax = axes[1, 1]
    ax.scatter(
        df["target_len"],
        df["length_ratio"],
        s=10,
        alpha=0.35,
        color="#ef6c00",
        edgecolors="none",
    )
    ax.axhline(1.0, color="#263238", linestyle="--", linewidth=1)
    ax.set_title("Prediction length ratio")
    ax.set_xlabel("Target tokens")
    ax.set_ylabel("Prediction tokens / target tokens")
    savefig(fig, out_dir / "01_error_overview.png")


def alignment_counters(
    df: pd.DataFrame,
    bins: int,
    top_k: int,
) -> dict[str, Any]:
    denominator = np.zeros(bins, dtype=float)
    insert_counts = np.zeros(bins, dtype=float)
    delete_counts = np.zeros(bins, dtype=float)
    replace_counts = np.zeros(bins, dtype=float)
    deleted: Counter[str] = Counter()
    inserted: Counter[str] = Counter()
    substitutions: Counter[tuple[str, str]] = Counter()

    for row in df.itertuples(index=False):
        ref = token_list(row.ground_truth)
        pred = token_list(row.prediction)
        if not ref:
            continue
        for idx in range(len(ref)):
            bin_idx = min(bins - 1, int(idx / max(1, len(ref) - 1) * (bins - 1)))
            denominator[bin_idx] += 1
        matcher = SequenceMatcher(a=ref, b=pred, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "insert":
                pos = min(len(ref) - 1, max(0, i1))
                bin_idx = min(bins - 1, int(pos / max(1, len(ref) - 1) * (bins - 1)))
                count = max(1, j2 - j1)
                insert_counts[bin_idx] += count
                inserted.update(pred[j1:j2])
                continue
            span = max(1, i2 - i1)
            for local_idx, ref_idx in enumerate(range(i1, i2)):
                bin_idx = min(bins - 1, int(ref_idx / max(1, len(ref) - 1) * (bins - 1)))
                if tag == "delete":
                    delete_counts[bin_idx] += 1
                    deleted[ref_idx and ref[ref_idx] or ref[ref_idx]] += 1
                elif tag == "replace":
                    pred_idx = j1 + min(local_idx, max(0, j2 - j1 - 1))
                    if pred_idx < j2:
                        replace_counts[bin_idx] += 1
                        substitutions[(ref[ref_idx], pred[pred_idx])] += 1
                    else:
                        delete_counts[bin_idx] += 1
                        deleted[ref[ref_idx]] += 1
            if tag == "replace" and (j2 - j1) > span:
                pos = min(len(ref) - 1, max(0, i2 - 1))
                bin_idx = min(bins - 1, int(pos / max(1, len(ref) - 1) * (bins - 1)))
                extras = pred[j1 + span : j2]
                insert_counts[bin_idx] += len(extras)
                inserted.update(extras)

    return {
        "denominator": denominator,
        "insert_counts": insert_counts,
        "delete_counts": delete_counts,
        "replace_counts": replace_counts,
        "deleted": deleted.most_common(top_k),
        "inserted": inserted.most_common(top_k),
        "substitutions": substitutions.most_common(top_k),
    }


def plot_token_position_errors(df: pd.DataFrame, out_dir: Path, bins: int, top_k: int) -> dict[str, Any]:
    counters = alignment_counters(df, bins=bins, top_k=top_k)
    denominator = np.maximum(counters["denominator"], 1.0)
    x = np.linspace(0, 100, bins)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, counters["replace_counts"] / denominator, label="substitution", color="#c62828")
    ax.plot(x, counters["delete_counts"] / denominator, label="deletion", color="#1565c0")
    ax.plot(x, counters["insert_counts"] / denominator, label="insertion", color="#2e7d32")
    ax.set_title("Token edit operations by normalized target position")
    ax.set_xlabel("Normalized target position (%)")
    ax.set_ylabel("Edit operations per reference token")
    ax.legend()
    ax.grid(alpha=0.2)
    savefig(fig, out_dir / "02_token_error_by_position.png")
    return counters


def plot_top_token_errors(counters: dict[str, Any], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    panels = [
        ("Most deleted target tokens", counters["deleted"], "#1565c0"),
        ("Most inserted prediction tokens", counters["inserted"], "#2e7d32"),
        (
            "Most common substitutions",
            [(f"{a} -> {b}", count) for (a, b), count in counters["substitutions"]],
            "#c62828",
        ),
    ]
    for ax, (title, items, color) in zip(axes, panels):
        labels = [str(item[0]) for item in items][::-1]
        values = [int(item[1]) for item in items][::-1]
        ax.barh(labels, values, color=color, alpha=0.82)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.tick_params(axis="y", labelsize=8)
    savefig(fig, out_dir / "03_top_token_error_types.png")


def group_by_bins(
    df: pd.DataFrame,
    value_col: str,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    tmp = df.copy()
    tmp["bucket"] = pd.cut(tmp[value_col], bins=bins, labels=labels, include_lowest=True)
    return tmp.groupby("bucket", observed=False)


def plot_length_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    labels = ["<=32", "33-64", "65-96", "97-128", "129-192", ">192"]
    grouped = group_by_bins(df, "target_len", [0, 32, 64, 96, 128, 192, np.inf], labels)
    summary = grouped.agg(
        samples=("key", "count"),
        token_ed=("token_edit_distance", "mean"),
        norm_ed=("normalized_token_edit_distance", "mean"),
        length_ratio=("length_ratio", "mean"),
        graph_em=("graph_em_ok", "mean"),
        structure_em=("graph_structure_ok", "mean"),
    ).reset_index()

    x = np.arange(len(summary))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].bar(x - 0.2, 100 * summary["graph_em"], width=0.4, label="Graph EM", color="#00897b")
    axes[0].bar(
        x + 0.2,
        100 * summary["structure_em"],
        width=0.4,
        label="Structure EM",
        color="#f9a825",
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Graph metrics collapse as target length grows")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(x, summary["token_ed"], marker="o", label="Token edit distance", color="#c62828")
    axes[1].plot(x, summary["length_ratio"] * 50, marker="s", label="Length ratio x50", color="#5e35b1")
    axes[1].set_ylabel("Distance / scaled ratio")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["bucket"])
    axes[1].set_xlabel("Target length bucket")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    for idx, n in enumerate(summary["samples"]):
        axes[0].text(idx, 2, f"n={int(n)}", ha="center", va="bottom", fontsize=8)
    savefig(fig, out_dir / "04_error_by_target_length.png")


def plot_graph_depth(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    depth_group = df.groupby("max_brace_depth").agg(
        samples=("key", "count"),
        graph_em=("graph_em_ok", "mean"),
        structure_em=("graph_structure_ok", "mean"),
        token_ed=("token_edit_rate", "mean"),
    )
    depth_group = depth_group[depth_group["samples"] >= 10]
    axes[0].plot(depth_group.index, 100 * depth_group["graph_em"], marker="o", label="Graph EM")
    axes[0].plot(
        depth_group.index,
        100 * depth_group["structure_em"],
        marker="s",
        label="Structure EM",
    )
    axes[0].set_title("Metric by brace/nesting depth")
    axes[0].set_xlabel("Max brace depth in target")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    branch_labels = ["<=20", "21-40", "41-80", "81-120", ">120"]
    branch_group = group_by_bins(
        df,
        "branch_count",
        [0, 20, 40, 80, 120, np.inf],
        branch_labels,
    ).agg(
        samples=("key", "count"),
        graph_em=("graph_em_ok", "mean"),
        structure_em=("graph_structure_ok", "mean"),
        token_ed=("token_edit_rate", "mean"),
    )
    x = np.arange(len(branch_group))
    axes[1].bar(x - 0.2, 100 * branch_group["graph_em"], width=0.4, label="Graph EM")
    axes[1].bar(
        x + 0.2,
        100 * branch_group["structure_em"],
        width=0.4,
        label="Structure EM",
    )
    axes[1].set_title("Metric by branch count")
    axes[1].set_xlabel("branch token count")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(branch_labels)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    savefig(fig, out_dir / "05_error_by_graph_depth.png")


def plot_feature_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    tmp = df.copy()
    length_labels = ["<=64", "65-128", ">128"]
    branch_labels = ["<=20", "21-60", ">60"]
    tmp["length_bucket"] = pd.cut(
        tmp["target_len"],
        [0, 64, 128, np.inf],
        labels=length_labels,
        include_lowest=True,
    )
    tmp["branch_bucket"] = pd.cut(
        tmp["branch_count"],
        [0, 20, 60, np.inf],
        labels=branch_labels,
        include_lowest=True,
    )
    pivot = tmp.pivot_table(
        index="length_bucket",
        columns="branch_bucket",
        values="graph_em_ok",
        aggfunc="mean",
        observed=False,
    )
    count = tmp.pivot_table(
        index="length_bucket",
        columns="branch_bucket",
        values="key",
        aggfunc="count",
        observed=False,
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(branch_labels)))
    ax.set_xticklabels(branch_labels)
    ax.set_yticks(range(len(length_labels)))
    ax.set_yticklabels(length_labels)
    ax.set_xlabel("Branch count bucket")
    ax.set_ylabel("Target length bucket")
    ax.set_title("Graph EM heatmap by length and graph complexity")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            n = int(count.iloc[i, j]) if i < count.shape[0] and j < count.shape[1] else 0
            label = "n=0" if np.isnan(value) else f"{100 * value:.1f}%\nn={n}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Graph EM")
    savefig(fig, out_dir / "06_graph_em_heatmap_length_branch.png")


def plot_graph_outcome_buckets(df: pd.DataFrame, out_dir: Path) -> None:
    if "graph_em_ok" not in df or df["graph_em_ok"].isna().all():
        return
    tmp = df.copy()
    tmp["graph_outcome"] = np.select(
        [
            tmp["graph_em_ok"] == True,
            (tmp["graph_structure_ok"] == True) & (tmp["graph_em_ok"] == False),
        ],
        ["graph_em_ok", "structure_ok_only"],
        default="structure_fail",
    )
    labels = ["<=64", "65-128", ">128"]
    tmp["length_bucket"] = pd.cut(
        tmp["target_len"],
        [0, 64, 128, np.inf],
        labels=labels,
        include_lowest=True,
    )
    counts = pd.crosstab(tmp["length_bucket"], tmp["graph_outcome"], normalize="index")
    counts = counts.reindex(columns=["graph_em_ok", "structure_ok_only", "structure_fail"]).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(counts))
    colors = {
        "graph_em_ok": "#00897b",
        "structure_ok_only": "#f9a825",
        "structure_fail": "#c62828",
    }
    for col in counts.columns:
        values = counts[col].to_numpy()
        ax.bar(counts.index.astype(str), values, bottom=bottom, label=col, color=colors[col])
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of samples")
    ax.set_title("Error outcome composition by length bucket")
    ax.legend()
    savefig(fig, out_dir / "07_graph_outcome_by_length.png")


def plot_compare_csv(current: pd.DataFrame, compare_csv: Path, out_dir: Path) -> None:
    old = pd.read_csv(compare_csv).fillna("")
    old["key"] = old["image_name"].map(clean_key)
    old["old_token_edit_distance"] = old["token_edit_distance"].map(to_float)
    old["old_prediction_len"] = old["prediction"].map(lambda value: len(token_list(value)))
    merged = current.merge(
        old[["key", "old_token_edit_distance", "old_prediction_len"]],
        on="key",
        how="inner",
    )
    merged["token_ed_improvement"] = (
        merged["old_token_edit_distance"] - merged["token_edit_distance"]
    )
    merged["prediction_len_delta"] = merged["prediction_len"] - merged["old_prediction_len"]

    labels = ["<=64", "65-128", ">128"]
    grouped = group_by_bins(
        merged,
        "target_len",
        [0, 64, 128, np.inf],
        labels,
    ).agg(
        samples=("key", "count"),
        token_ed_improvement=("token_ed_improvement", "mean"),
        prediction_len_delta=("prediction_len_delta", "mean"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(grouped))
    axes[0].bar(x, grouped["token_ed_improvement"], color="#00897b")
    axes[0].set_title("Token edit distance improvement vs compare CSV")
    axes[0].set_ylabel("Old ED - current ED")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, grouped["prediction_len_delta"], color="#5e35b1")
    axes[1].set_title("Prediction length change vs compare CSV")
    axes[1].set_ylabel("Current pred tokens - old pred tokens")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", alpha=0.2)
    for ax in axes:
        for idx, n in enumerate(grouped["samples"]):
            ax.text(idx, ax.get_ylim()[0], f"n={int(n)}", ha="center", va="bottom", fontsize=8)
    savefig(fig, out_dir / "08_compare_csv_improvement.png")


def write_feature_table(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "image_name",
        "target_len",
        "prediction_len",
        "length_ratio",
        "token_edit_distance",
        "normalized_token_edit_distance",
        "char_edit_distance",
        "exact_match_bool",
        "graph_em_ok",
        "graph_structure_ok",
        "chemfig_count",
        "arrow_count",
        "branch_count",
        "ring_anchor_count",
        "angle_bond_count",
        "max_brace_depth",
        "max_paren_depth",
        "structure_score",
    ]
    existing = [col for col in cols if col in df.columns]
    df[existing].to_csv(out_dir / "per_sample_error_features.csv", index=False)


def write_report(df: pd.DataFrame, out_dir: Path, image_paths: list[Path]) -> None:
    graph_em = df["graph_em_ok"].mean() if "graph_em_ok" in df else float("nan")
    structure_em = (
        df["graph_structure_ok"].mean() if "graph_structure_ok" in df else float("nan")
    )
    long_df = df[df["target_len"] > 128]
    report = [
        "# Error Visualization Report",
        "",
        f"- Samples: {len(df)}",
        f"- String EM: {100 * df['exact_match_bool'].mean():.2f}%",
        f"- Graph EM: {100 * graph_em:.2f}%",
        f"- Structure EM: {100 * structure_em:.2f}%",
        f"- Mean token edit distance: {df['token_edit_distance'].mean():.2f}",
        f"- Mean normalized token edit distance: {df['normalized_token_edit_distance'].mean():.3f}",
        f"- Long samples >128 tokens: {len(long_df)}",
        (
            f"- Long-sample mean token edit distance: "
            f"{long_df['token_edit_distance'].mean():.2f}"
            if len(long_df)
            else "- Long-sample mean token edit distance: n/a"
        ),
        "",
        "## Generated Figures",
        "",
    ]
    for path in image_paths:
        report.append(f"- `{path.name}`")
    report.append("")
    report.append(
        "Interpretation tip: if token edit error and graph failure rise with target length, "
        "the bottleneck is likely long-reaction representation/decoding rather than only "
        "local OCR quality."
    )
    (out_dir / "error_visualization_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_eval_frame(args.csv, args.graph_result)

    before = set(args.out_dir.glob("*.png"))
    plot_dashboard(df, args.out_dir)
    counters = plot_token_position_errors(
        df,
        args.out_dir,
        bins=args.position_bins,
        top_k=args.top_k,
    )
    plot_top_token_errors(counters, args.out_dir)
    plot_length_breakdown(df, args.out_dir)
    plot_graph_depth(df, args.out_dir)
    plot_feature_heatmap(df, args.out_dir)
    plot_graph_outcome_buckets(df, args.out_dir)
    if args.compare_csv is not None:
        plot_compare_csv(df, args.compare_csv, args.out_dir)
    write_feature_table(df, args.out_dir)
    images = sorted(set(args.out_dir.glob("*.png")) - before)
    if not images:
        images = sorted(args.out_dir.glob("*.png"))
    write_report(df, args.out_dir, images)

    print(f"Wrote {len(images)} figures to {args.out_dir}")
    for path in images:
        print(path)
    print(args.out_dir / "per_sample_error_features.csv")
    print(args.out_dir / "error_visualization_report.md")


if __name__ == "__main__":
    main()
