from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trinity_vlm.graft import TrinityMoondreamGraftConfig, TrinityMoondreamGraftModel
from trinity_vlm.tokenizer_utils import load_trinity_tokenizer, save_trinity_chat_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TrinityVLM model and export a self-contained local model directory."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trinity-model", default="arcee-ai/Trinity-Nano-Preview")
    parser.add_argument("--moondream-model", default="moondream/moondream3-preview")
    parser.add_argument("--delta-dir")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--projector-hidden-dim", type=int, default=2048)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--train-trinity", action="store_true")
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    parser.add_argument("--max-shard-size", default="10GB")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu" and dtype_name != "float32":
        return torch.float32
    return getattr(torch, dtype_name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    model = TrinityMoondreamGraftModel(
        TrinityMoondreamGraftConfig(
            trinity_model_name=args.trinity_model,
            moondream_model_name=args.moondream_model,
            projector_hidden_dim=args.projector_hidden_dim,
            freeze_trinity=not args.train_trinity,
            freeze_vision=not args.train_vision,
            train_lm_head=args.train_lm_head,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        ),
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
        revision=args.revision,
    )
    model.to(device)
    model.eval()

    tokenizer = load_trinity_tokenizer(
        args.trinity_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    token_info = model.ensure_image_special_tokens(tokenizer)

    if args.delta_dir:
        model.load_delta(args.delta_dir)

    model.export_pretrained(
        output_dir,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    save_trinity_chat_template(output_dir)

    metadata = {
        "exported_from": "scripts/export_trinityvlm.py",
        "trinity_model": args.trinity_model,
        "moondream_model": args.moondream_model,
        "delta_dir": args.delta_dir,
        "token_info": token_info,
        "resolved_device": str(device),
        "resolved_dtype": str(dtype),
    }
    (output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"exported TrinityVLM model to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
