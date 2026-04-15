from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import CaptionCollator, PixMoCaptionIterable
from .graft import TrinityMoondreamGraftConfig, TrinityMoondreamGraftModel
from .tokenizer_utils import load_trinity_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Moondream-vision graft on Trinity.")
    parser.add_argument("--trinity-model", default="arcee-ai/Trinity-Nano-Preview")
    parser.add_argument("--moondream-model", default="moondream/moondream3-preview")
    parser.add_argument("--dataset", default="anthracite-org/pixmo-cap-images")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="checkpoints/trinity-moondream-graft")
    parser.add_argument("--prompt-text", default="Describe the image in detail.")
    parser.add_argument("--user-message-template", default="{image}\n{prompt}")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=64)
    parser.add_argument("--max-caption-tokens", type=int, default=384)
    parser.add_argument("--shuffle-buffer-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--train-trinity", action="store_true")
    parser.add_argument("--train-lm-head", action="store_true")
    parser.add_argument("--unfreeze-last-n-layers", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu" and dtype_name != "float32":
        return torch.float32
    return getattr(torch, dtype_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: TrinityMoondreamGraftModel,
    optimizer: AdamW,
    scheduler,
    args: argparse.Namespace,
    step: int,
) -> None:
    checkpoint_dir = Path(args.output_dir) / f"step-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_delta(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    (checkpoint_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer = load_trinity_tokenizer(
        args.trinity_model,
        local_files_only=args.local_files_only,
    )

    model = TrinityMoondreamGraftModel(
        TrinityMoondreamGraftConfig(
            trinity_model_name=args.trinity_model,
            moondream_model_name=args.moondream_model,
            freeze_trinity=not args.train_trinity,
            freeze_vision=not args.train_vision,
            train_lm_head=args.train_lm_head,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        ),
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.train()
    token_info = model.ensure_image_special_tokens(tokenizer)

    dataset = PixMoCaptionIterable(
        dataset_name=args.dataset,
        split=args.split,
        streaming=True,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        limit=args.limit,
    )
    collator = CaptionCollator(
        tokenizer=tokenizer,
        prompt_text=args.prompt_text,
        user_message_template=args.user_message_template,
        max_prompt_tokens=args.max_prompt_tokens,
        max_caption_tokens=args.max_caption_tokens,
        vision_start_token_id=token_info["vision_start_token_id"],
        image_token_id=token_info["image_token_id"],
        vision_end_token_id=token_info["vision_end_token_id"],
        image_seq_len=model.config.image_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=0,
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters were selected.")

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args.warmup_steps, args.max_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "resolved_device": str(device),
                "resolved_dtype": str(dtype),
                "trainable_parameters": model.trainable_parameter_count(),
            },
            indent=2,
        )
    )

    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    running_loss = 0.0
    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
        micro_step += 1
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            output = model(**batch)
            loss = output.loss / args.grad_accum_steps

        loss.backward()
        running_loss += loss.item() * args.grad_accum_steps

        if micro_step % args.grad_accum_steps != 0:
            continue

        torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        if global_step % args.log_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            average_loss = running_loss / float(args.log_every)
            print(
                f"step={global_step} loss={average_loss:.4f} lr={current_lr:.6g}",
                flush=True,
            )
            running_loss = 0.0

        if global_step % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, args, global_step)

        if global_step >= args.max_steps:
            break

    if global_step % args.save_every != 0:
        save_checkpoint(model, optimizer, scheduler, args, global_step)

    print(f"finished training at step={global_step}", flush=True)
