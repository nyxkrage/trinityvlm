from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import torch

from trinity_vlm.graft import TrinityMoondreamGraftConfig, TrinityMoondreamGraftModel
from trinity_vlm.tokenizer_utils import load_trinity_tokenizer, save_trinity_chat_template


DEFAULT_RUN_DIR = Path("checkpoints/bridge-mix-pixmo-nemotron-local-20260415-ddp-cce")
DEFAULT_CHECKPOINT = "step-003750"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a merged TrinityVLM model directory from a training checkpoint. "
            "The default source is the current DDP CCE run at step-003750."
        )
    )
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Training output directory containing checkpoints/, tokenizer/, and resolved_config.toml.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint directory name, absolute path, or 'latest'. Defaults to step-003750.",
    )
    parser.add_argument(
        "--output-dir",
        help="Merged model output directory. Defaults to exports/<run-name>-<checkpoint-name>-final.",
    )
    parser.add_argument("--device", default="cpu", help="Device used while merging, for example cpu or cuda.")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-shard-size", default="10GB")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve base repos from the local Hugging Face cache only.",
    )
    parser.add_argument("--trinity-revision")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device(device_name)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def resolve_checkpoint_dir(run_dir: Path, checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.exists():
        return checkpoint_path.resolve()

    if checkpoint == "latest":
        latest_path = run_dir / "latest_checkpoint.txt"
        if not latest_path.exists():
            raise FileNotFoundError(f"Could not find {latest_path}")
        latest_text = latest_path.read_text().strip()
        latest_candidate = Path(latest_text).expanduser()
        if latest_candidate.exists():
            return latest_candidate.resolve()
        run_relative_candidate = (run_dir / latest_text).resolve()
        if run_relative_candidate.exists():
            return run_relative_candidate
        raise FileNotFoundError(
            f"latest checkpoint path {latest_text!r} from {latest_path} does not exist."
        )

    candidate = (run_dir / "checkpoints" / checkpoint).resolve()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve checkpoint {checkpoint!r} under {run_dir / 'checkpoints'}."
    )


def resolve_output_dir(args: argparse.Namespace, run_dir: Path, checkpoint_dir: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (Path("exports") / f"{run_dir.name}-{checkpoint_dir.name}-final").resolve()


def load_resolved_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "resolved_config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def build_graft_config(
    resolved_config: dict[str, Any],
) -> tuple[TrinityMoondreamGraftConfig, str, str]:
    model_section = resolved_config.get("model", {})
    unfreeze_section = resolved_config.get("unfreeze", {})
    graft_config = TrinityMoondreamGraftConfig(
        trinity_model_name=model_section.get("trinity_model", "arcee-ai/Trinity-Nano-Preview"),
        moondream_model_name=model_section.get("moondream_model", "moondream/moondream3-preview"),
        projector_hidden_dim=model_section.get("projector_hidden_dim", 2048),
        use_cut_cross_entropy=model_section.get("use_cut_cross_entropy", False),
        cut_cross_entropy_impl=model_section.get("cut_cross_entropy_impl", "cce"),
        freeze_trinity=not unfreeze_section.get("train_trinity", False),
        freeze_vision=not unfreeze_section.get("train_vision", False),
        train_lm_head=unfreeze_section.get("train_lm_head", False),
        unfreeze_last_n_layers=unfreeze_section.get("unfreeze_last_n_layers", 0),
    )
    trinity_revision = model_section.get("trinity_revision")
    moondream_revision = model_section.get("moondream_revision")
    return graft_config, trinity_revision, moondream_revision


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_dir(run_dir, args.checkpoint)
    output_dir = resolve_output_dir(args, run_dir, checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = load_resolved_config(run_dir)
    graft_config, trinity_revision, moondream_revision = build_graft_config(resolved_config)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    model = TrinityMoondreamGraftModel(
        graft_config,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        revision=args.trinity_revision or trinity_revision,
    )
    model.to(device)
    model.eval()

    run_tokenizer_dir = run_dir / "tokenizer"
    tokenizer_source = (
        str(run_tokenizer_dir)
        if run_tokenizer_dir.exists()
        else graft_config.trinity_model_name
    )
    tokenizer = load_trinity_tokenizer(
        tokenizer_source,
        revision=None if run_tokenizer_dir.exists() else (args.trinity_revision or trinity_revision),
        local_files_only=True if run_tokenizer_dir.exists() else args.local_files_only,
    )
    token_info = model.ensure_image_special_tokens(tokenizer)
    model.load_delta(checkpoint_dir)

    model.export_pretrained(
        output_dir,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    save_trinity_chat_template(output_dir)

    print(f"exported merged TrinityVLM model to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
