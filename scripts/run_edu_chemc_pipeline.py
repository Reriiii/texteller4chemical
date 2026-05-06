from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "ConstantHao/EDU-CHEMC_MM23"
DEFAULT_DATASET_DIR = Path("data/processed/edu_chemc_normed")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/runs/edu_chemc_texteller_normed_len768_r32_all_lora_balanced_30ep"
)
DEFAULT_EVAL_CSV = Path("outputs/eval_normed_len768_r32_all_lora_balanced_30ep_test_greedy.csv")
STAGE_ORDER = ("download", "prepare", "analyze", "train", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EDU-CHEMC Hugging Face -> prepare -> train -> graph-evaluate pipeline."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", *STAGE_ORDER],
        default=["all"],
        help="Stages to run; selected stages always execute in pipeline order.",
    )
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target_field", type=str, default="ssml_normed")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--overwrite_prepare", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/train_edu_chemc.yaml"))
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="OleehyO/TexTeller")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--dynamo_backend", type=str, default="no")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--eval_output_csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=Path("external/GraphMatchingTool"))
    parser.add_argument("--graph_num_workers", type=int, default=8)
    parser.add_argument("--no_graph_eval", action="store_true")
    parser.add_argument("--graph_keep_temp", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def selected_stages(values: list[str]) -> list[str]:
    if "all" in values:
        return list(STAGE_ORDER)
    requested = set(values)
    return [stage for stage in STAGE_ORDER if stage in requested]


def display_cmd(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(f"$ {display_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def load_dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.dataset_config:
        kwargs["name"] = args.dataset_config
    if args.revision:
        kwargs["revision"] = args.revision
    if args.cache_dir:
        kwargs["cache_dir"] = str(args.cache_dir)
    return kwargs


def download_dataset(args: argparse.Namespace) -> None:
    shown = [
        sys.executable,
        "-c",
        f"from datasets import load_dataset; load_dataset({args.dataset_id!r})",
    ]
    print(f"$ {display_cmd(shown)}", flush=True)
    if args.dry_run:
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("The 'datasets' package is required. Run `uv sync` or install repo deps.") from exc
    load_dataset(args.dataset_id, **load_dataset_kwargs(args))


def append_optional_hf_args(cmd: list[str], args: argparse.Namespace) -> None:
    if args.dataset_config:
        cmd.extend(["--dataset_config", args.dataset_config])
    if args.revision:
        cmd.extend(["--revision", args.revision])
    if args.cache_dir:
        cmd.extend(["--cache_dir", str(args.cache_dir)])


def prepare_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/materialize_hf_edu_chemc.py",
        "--dataset_id",
        args.dataset_id,
        "--out_dir",
        str(args.dataset_dir),
        "--target_field",
        args.target_field,
    ]
    append_optional_hf_args(cmd, args)
    if args.max_samples_per_split is not None:
        cmd.extend(["--max_samples_per_split", str(args.max_samples_per_split)])
    if args.overwrite_prepare:
        cmd.append("--overwrite")
    if args.target_field == "ssml_rcgd":
        cmd.append("--allow_rcgd")
    return cmd


def analyze_commands(args: argparse.Namespace) -> list[list[str]]:
    metadata = args.dataset_dir / "train" / "metadata.jsonl"
    return [
        [
            sys.executable,
            "scripts/analyze_targets.py",
            "--metadata",
            str(metadata),
        ],
        [
            sys.executable,
            "scripts/analyze_tokenizer_coverage.py",
            "--metadata",
            str(metadata),
            "--pretrained_model_name_or_path",
            args.pretrained_model_name_or_path,
            "--max_decoder_length",
            str(args.max_new_tokens),
        ],
    ]


def accelerate_command_prefix(args: argparse.Namespace) -> list[str]:
    accelerate_bin = shutil.which("accelerate")
    if accelerate_bin:
        cmd = [accelerate_bin, "launch"]
    else:
        cmd = [sys.executable, "-m", "accelerate.commands.launch"]
    cmd.extend(
        [
            "--num_machines",
            str(args.num_machines),
            "--mixed_precision",
            args.mixed_precision,
            "--dynamo_backend",
            args.dynamo_backend,
        ]
    )
    if args.num_processes is not None:
        cmd.extend(["--num_processes", str(args.num_processes)])
    return cmd


def train_command(args: argparse.Namespace) -> list[str]:
    cmd = accelerate_command_prefix(args)
    cmd.extend(
        [
            "scripts/train.py",
            "--config",
            str(args.config),
            "--dataset_dir",
            str(args.dataset_dir),
            "--pretrained_model_name_or_path",
            args.pretrained_model_name_or_path,
            "--output_dir",
            str(args.output_dir),
        ]
    )
    if args.tokenizer_path:
        cmd.extend(["--tokenizer_path", args.tokenizer_path])
    if args.resume_from_checkpoint:
        cmd.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    return cmd


def evaluate_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--model_ckpt",
        str(args.output_dir / "best"),
        "--dataset_dir",
        str(args.dataset_dir),
        "--split",
        args.eval_split,
        "--batch_size",
        str(args.eval_batch_size),
        "--num_beams",
        str(args.num_beams),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--output_csv",
        str(args.eval_output_csv),
        "--graph_label_key",
        args.target_field,
    ]
    if args.eval_max_samples is not None:
        cmd.extend(["--max_samples", str(args.eval_max_samples)])
    if args.tokenizer_path:
        cmd.extend(["--tokenizer_path", args.tokenizer_path])
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")
    if not args.no_graph_eval:
        cmd.extend(
            [
                "--graph_eval",
                "--graph_matching_tool_dir",
                str(args.graph_matching_tool_dir),
                "--graph_num_workers",
                str(args.graph_num_workers),
            ]
        )
        if args.graph_keep_temp:
            cmd.append("--graph_keep_temp")
    return cmd


def main() -> None:
    args = parse_args()
    stages = selected_stages(args.stages)
    for stage in stages:
        print(f"\n== {stage} ==", flush=True)
        if stage == "download":
            download_dataset(args)
        elif stage == "prepare":
            run_command(prepare_command(args), dry_run=args.dry_run)
        elif stage == "analyze":
            for cmd in analyze_commands(args):
                run_command(cmd, dry_run=args.dry_run)
        elif stage == "train":
            run_command(train_command(args), dry_run=args.dry_run)
        elif stage == "evaluate":
            run_command(evaluate_command(args), dry_run=args.dry_run)
        else:
            raise AssertionError(f"Unhandled stage: {stage}")


if __name__ == "__main__":
    main()
