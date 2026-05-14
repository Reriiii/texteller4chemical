from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "ConstantHao/EDU-CHEMC_MM23"
DEFAULT_TARGET_FIELD = "ssml_graph_norm"
DEFAULT_DATASET_DIR = Path("data/processed/edu_chemc_graph_norm")
DEFAULT_RFL_DATASET_DIR = Path("data/processed/edu_chemc_rfl_msd")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/runs/edu_chemc_texteller_graph_norm_full_model_bf16_30ep"
)
DEFAULT_RFL_OUTPUT_DIR = Path(
    "outputs/runs/edu_chemc_texteller_rfl_msd_full_model_bf16_30ep"
)
DEFAULT_EVAL_CSV = Path("outputs/eval_graph_norm_full_model_bf16_30ep_test_greedy.csv")
DEFAULT_RFL_EVAL_CSV = Path("outputs/eval_rfl_msd_full_model_bf16_30ep_test_greedy.csv")
DEFAULT_RFL_GRAPH_LABEL_FIELD = "ssml_rfl_graph_norm"
BASE_STAGE_ORDER = ("download", "prepare", "analyze", "train", "evaluate")
STAGE_ORDER = ("download", "prepare", "rfl_prepare", "rfl_check", "analyze", "train", "evaluate")
STAGE_ALIASES = ("all", "train_eval", "rfl_msd", "rfl_msd_prepare", "rfl_msd_train_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EDU-CHEMC Hugging Face -> prepare -> train -> graph-evaluate pipeline."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[*STAGE_ALIASES, *STAGE_ORDER],
        default=["all"],
        help=(
            "Stages to run; selected stages always execute in pipeline order. "
            "Use train_eval to train, then evaluate the resulting best checkpoint."
        ),
    )
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target_field", type=str, default=DEFAULT_TARGET_FIELD)
    parser.add_argument(
        "--use_rfl_msd",
        action="store_true",
        help="Use the RFL-MSD prepared dataset/config for analyze/train/evaluate stages.",
    )
    parser.add_argument("--rfl_dataset_dir", type=Path, default=DEFAULT_RFL_DATASET_DIR)
    parser.add_argument("--rfl_source_key", type=str, default="ssml_normed")
    parser.add_argument("--rfl_target_field", type=str, default="ssml_rfl")
    parser.add_argument("--rfl_graph_label_field", type=str, default=DEFAULT_RFL_GRAPH_LABEL_FIELD)
    parser.add_argument("--rfl_aux_field", type=str, default="rfl")
    parser.add_argument("--rfl_tool_dir", type=Path, default=Path("external/RFL-MSD"))
    parser.add_argument("--rfl_on_error", choices=["raise", "skip", "fallback"], default="raise")
    parser.add_argument("--overwrite_rfl_prepare", action="store_true")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--overwrite_prepare", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/train_edu_chemc.yaml"))
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="OleehyO/TexTeller")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help=(
            "Comma-separated physical GPU ids to expose to child processes, "
            "for example '3' or '0,1'. Prefer this over shell env prefixes "
            "when using the pipeline script."
        ),
    )
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--dynamo_backend", type=str, default="no")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--eval_output_csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument(
        "--eval_target_key",
        type=str,
        default=None,
        help="Target key used as the sequence reference in evaluate.py; defaults to prepared target.",
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--graph_matching_tool_dir", type=Path, default=Path("external/GraphMatchingTool"))
    parser.add_argument(
        "--graph_label_key",
        type=str,
        default=None,
        help="Metadata target used as the GraphMatchingTool label; defaults to --target_field.",
    )
    parser.add_argument(
        "--prediction_normalizer",
        type=str,
        default=None,
        help="Optional prediction normalizer for graph-safe evaluation, e.g. ssml_graph_sd.",
    )
    parser.add_argument("--graph_num_workers", type=int, default=8)
    parser.add_argument("--no_graph_eval", action="store_true")
    parser.add_argument("--graph_keep_temp", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def selected_stages(values: list[str]) -> list[str]:
    if "all" in values:
        return list(BASE_STAGE_ORDER)
    if "rfl_msd" in values:
        values = [*values, "prepare", "rfl_prepare", "rfl_check", "analyze", "train", "evaluate"]
    if "rfl_msd_prepare" in values:
        values = [*values, "prepare", "rfl_prepare", "rfl_check"]
    if "rfl_msd_train_eval" in values:
        values = [*values, "rfl_check", "train", "evaluate"]
    if "train_eval" in values:
        values = [*values, "train", "evaluate"]
    requested = set(values)
    return [stage for stage in STAGE_ORDER if stage in requested]


def active_dataset_dir(args: argparse.Namespace) -> Path:
    return args.rfl_dataset_dir if args.use_rfl_msd else args.dataset_dir


def display_cmd(cmd: list[str]) -> str:
    return shlex.join(cmd)


def child_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def env_prefix(args: argparse.Namespace) -> str:
    if args.cuda_visible_devices is None:
        return ""
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)} "


def run_command(cmd: list[str], dry_run: bool, args: argparse.Namespace) -> None:
    print(f"$ {env_prefix(args)}{display_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=child_env(args))


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
    ensure_hf_cache_dirs(args)
    try:
        load_dataset(args.dataset_id, **load_dataset_kwargs(args))
    except FileNotFoundError as exc:
        raise SystemExit(hf_download_error_message(args, exc)) from exc


def ensure_hf_cache_dirs(args: argparse.Namespace) -> None:
    if args.cache_dir:
        ensure_directory(args.cache_dir, "HF datasets cache")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    ensure_directory(hf_home, "HF_HOME")
    ensure_directory(hf_home / "hub", "HF hub cache")
    ensure_directory(hf_home / "datasets", "HF datasets cache")


def ensure_directory(path: Path, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError as exc:
        raise SystemExit(
            f"{label} path exists but is not a directory: {path}. "
            "Set HF_HOME and --cache_dir to a writable directory."
        ) from exc
    if not path.is_dir():
        raise SystemExit(
            f"{label} path exists but is not a directory: {path}. "
            "Set HF_HOME and --cache_dir to a writable directory."
        )


def hf_download_error_message(args: argparse.Namespace, exc: FileNotFoundError) -> str:
    offline_vars = [
        name
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
        if os.environ.get(name)
    ]
    hint = ""
    if offline_vars:
        hint = f"\nOffline env vars are set: {', '.join(offline_vars)}. Unset them to download."
    cache_hint = f"\nIf this server uses a custom disk, pass: --cache_dir /path/to/hf_cache"
    return (
        f"Could not download Hugging Face dataset {args.dataset_id!r}.\n"
        "Check internet/proxy access to https://huggingface.co and Hugging Face offline env vars."
        f"{hint}{cache_hint}\nOriginal error: {exc}"
    )


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
    metadata = active_dataset_dir(args) / "train" / "metadata.jsonl"
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


def rfl_prepare_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/create_rfl_target_dataset.py",
        "--dataset_dir",
        str(args.dataset_dir),
        "--out_dir",
        str(args.rfl_dataset_dir),
        "--source_key",
        args.rfl_source_key,
        "--target_field",
        args.rfl_target_field,
        "--graph_label_field",
        args.rfl_graph_label_field,
        "--aux_field",
        args.rfl_aux_field,
        "--rfl_tool_dir",
        str(args.rfl_tool_dir),
        "--on_error",
        args.rfl_on_error,
    ]
    if args.max_samples_per_split is not None:
        cmd.extend(["--max_samples_per_split", str(args.max_samples_per_split)])
    if args.overwrite_rfl_prepare:
        cmd.append("--overwrite")
    return cmd


def rfl_check_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/check_rfl_msd_loss_readiness.py",
        "--config",
        str(args.config),
        "--dataset_dir",
        str(args.rfl_dataset_dir),
        "--rfl_aux_field",
        args.rfl_aux_field,
        "--graph_label_key",
        args.rfl_graph_label_field,
        "--assume_rfl_decoder",
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
            str(active_dataset_dir(args)),
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
        str(active_dataset_dir(args)),
        "--config",
        str(args.config),
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
        args.graph_label_key or (args.rfl_graph_label_field if args.use_rfl_msd else args.target_field),
    ]
    if args.eval_target_key:
        cmd.extend(["--target_key", args.eval_target_key])
    if args.prediction_normalizer:
        cmd.extend(["--prediction_normalizer", args.prediction_normalizer])
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
        if args.use_rfl_msd:
            cmd.extend(
                [
                    "--rfl_graph_restore",
                    "--rfl_tool_dir",
                    str(args.rfl_tool_dir),
                ]
            )
        if args.graph_keep_temp:
            cmd.append("--graph_keep_temp")
    return cmd


def main() -> None:
    args = parse_args()
    if any(stage.startswith("rfl_") for stage in args.stages) or any(
        stage in {"rfl_msd", "rfl_msd_prepare", "rfl_msd_train_eval"} for stage in args.stages
    ):
        args.use_rfl_msd = True
    if args.use_rfl_msd and args.config == Path("configs/train_edu_chemc.yaml"):
        args.config = Path("configs/train_edu_chemc_rfl_msd.yaml")
    if args.use_rfl_msd and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_RFL_OUTPUT_DIR
    if args.use_rfl_msd and args.eval_output_csv == DEFAULT_EVAL_CSV:
        args.eval_output_csv = DEFAULT_RFL_EVAL_CSV
    stages = selected_stages(args.stages)
    if (
        "train" in stages
        and args.num_processes == 1
        and args.cuda_visible_devices is None
        and "CUDA_VISIBLE_DEVICES" not in os.environ
    ):
        print(
            "WARNING: --num_processes 1 only starts one training process, but all GPUs "
            "remain visible. Transformers Trainer may use torch.nn.DataParallel across "
            "all visible GPUs. Pass --cuda_visible_devices <gpu_id> to pin one GPU.",
            flush=True,
        )
    for stage in stages:
        print(f"\n== {stage} ==", flush=True)
        if stage == "download":
            download_dataset(args)
        elif stage == "prepare":
            run_command(prepare_command(args), dry_run=args.dry_run, args=args)
        elif stage == "rfl_prepare":
            run_command(rfl_prepare_command(args), dry_run=args.dry_run, args=args)
        elif stage == "rfl_check":
            run_command(rfl_check_command(args), dry_run=args.dry_run, args=args)
        elif stage == "analyze":
            for cmd in analyze_commands(args):
                run_command(cmd, dry_run=args.dry_run, args=args)
        elif stage == "train":
            run_command(train_command(args), dry_run=args.dry_run, args=args)
        elif stage == "evaluate":
            run_command(evaluate_command(args), dry_run=args.dry_run, args=args)
        else:
            raise AssertionError(f"Unhandled stage: {stage}")


if __name__ == "__main__":
    main()
