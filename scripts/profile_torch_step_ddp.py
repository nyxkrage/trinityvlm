#!/home/carsten/trinity-vlm/.venv/bin/python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from trinity_vlm.train_ddp import apply_ddp
from trinity_vlm.train_fsdp2 import (
    barrier_if_distributed,
    build_dataloader,
    build_model_and_tokenizer,
    create_dataloader_batches,
    init_distributed,
    load_config,
    maybe_compile_model,
    maybe_enable_activation_checkpointing,
    maybe_enable_tf32,
    maybe_relaunch_with_torchrun,
    rank0_print,
    resolve_dtype,
    resolve_train_data_section,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one real Trinity-VLM DDP training step with torch.profiler.")
    parser.add_argument("config", help="Path to a TOML config file.")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=40)
    return parser.parse_args()


def _run_step(
    *,
    model,
    batch,
    distributed,
    dtype: torch.dtype,
    autocast_enabled: bool,
) -> torch.Tensor:
    with torch.autocast(
        device_type=distributed.device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        output = model(
            images=batch["images"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            labels=batch["labels"],
        )
        loss = output.loss
    loss.backward()
    return loss.detach()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    if maybe_relaunch_with_torchrun(config_path, config):
        return

    distributed = init_distributed(config)
    try:
        set_seed(config.train.seed + distributed.rank)
        maybe_enable_tf32(config, distributed.device)
        dtype = resolve_dtype(config.train.dtype, distributed.device)
        autocast_enabled = distributed.device.type == "cuda" and dtype != torch.float32

        model, tokenizer, token_info = build_model_and_tokenizer(config, dtype)
        model.configure_profiling(enabled=False, synchronize_cuda=False)
        maybe_enable_activation_checkpointing(model, config.train.activation_checkpointing)
        model.to(distributed.device)
        model = maybe_compile_model(model, False)
        model = apply_ddp(model, config, distributed)
        model.train()

        train_data = resolve_train_data_section(config)
        train_dataloader = build_dataloader(
            train_data,
            tokenizer,
            token_info,
            image_seq_len=model.module.config.image_seq_len if hasattr(model, "module") else model.config.image_seq_len,
            sliding_window=(
                model.module.language_model.config.sliding_window
                if hasattr(model, "module")
                else model.language_model.config.sliding_window
            ),
            distributed=distributed,
        )
        batch_iterator = iter(
            create_dataloader_batches(
                train_dataloader,
                train_data,
                distributed,
                label="training",
            )
        )

        model.zero_grad(set_to_none=True)
        for _ in range(args.warmup_steps):
            batch = next(batch_iterator)
            loss = _run_step(
                model=model,
                batch=batch,
                distributed=distributed,
                dtype=dtype,
                autocast_enabled=autocast_enabled,
            )
            model.zero_grad(set_to_none=True)
            if distributed.device.type == "cuda":
                torch.cuda.synchronize(distributed.device)
            rank0_print(distributed, f"warmup_loss={float(loss.float().item()):.4f}")

        activities = [ProfilerActivity.CPU]
        if distributed.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(args.profile_steps):
                batch = next(batch_iterator)
                _run_step(
                    model=model,
                    batch=batch,
                    distributed=distributed,
                    dtype=dtype,
                    autocast_enabled=autocast_enabled,
                )
                prof.step()

        if distributed.device.type == "cuda":
            torch.cuda.synchronize(distributed.device)
        barrier_if_distributed(distributed)

        if distributed.is_main_process:
            print("=== self_cuda_time_total ===", flush=True)
            print(
                prof.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=args.row_limit,
                ),
                flush=True,
            )
            print("=== cuda_time_total ===", flush=True)
            print(
                prof.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=args.row_limit,
                ),
                flush=True,
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
