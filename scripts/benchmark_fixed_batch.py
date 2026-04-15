#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from trinity_vlm.data import PackedCaptionCollator
from trinity_vlm.train_fsdp2 import (
    Config,
    accumulate_timing,
    apply_fsdp2,
    assert_consistent_image_presence_across_ranks,
    barrier_if_distributed,
    build_model_and_tokenizer,
    build_optimizer,
    build_scheduler,
    init_distributed,
    load_config,
    maybe_compile_model,
    maybe_enable_activation_checkpointing,
    maybe_enable_tf32,
    maybe_set_gradient_sync,
    maybe_relaunch_with_torchrun,
    maybe_synchronize_timing,
    rank0_print,
    resolve_dtype,
    set_seed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TrinityVLM on a fixed synthetic multimodal batch.")
    parser.add_argument("config", help="Path to a TOML config file.")
    parser.add_argument("--steps", type=int, default=2, help="Number of optimizer steps to benchmark.")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="Number of optimizer steps to run before measurement.",
    )
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Tag used in printed benchmark lines.",
    )
    return parser.parse_args(argv)


def _make_repeated_text(prefix: str, repeat: int) -> str:
    return " ".join([prefix] * repeat)


def _build_fixed_batch(config: Config, tokenizer, token_info: dict[str, int], *, sliding_window: int):
    collator = PackedCaptionCollator(
        tokenizer=tokenizer,
        prompt_text=config.data.prompt_text,
        user_message_template=config.data.user_message_template,
        max_prompt_tokens=config.data.max_prompt_tokens,
        max_caption_tokens=config.data.max_caption_tokens,
        max_chat_input_tokens=config.data.max_chat_input_tokens,
        max_chat_output_tokens=config.data.max_chat_output_tokens,
        max_seq_len=config.data.max_seq_len,
        sliding_window=sliding_window,
        vision_start_token_id=int(token_info["vision_start_token_id"]),
        image_token_id=int(token_info["image_token_id"]),
        vision_end_token_id=int(token_info["vision_end_token_id"]),
        image_seq_len=729,
    )

    image_a = Image.new("RGB", (768, 768), color=(255, 255, 255))
    image_b = Image.new("RGB", (768, 768), color=(64, 96, 160))

    batch = [
        {
            "example_type": "cap_qa",
            "image": image_a,
            "prompt_text": _make_repeated_text(
                "Answer the question about the image and be precise about visible structure and texture.",
                48,
            ),
            "assistant_output": _make_repeated_text(
                "The answer should mention object identity, layout, spatial relations, colors, and notable fine details.",
                32,
            ),
        },
        {
            "example_type": "point_explanation",
            "image": image_b,
            "prompt_text": _make_repeated_text(
                "Explain the marked point in the image, grounding the explanation in nearby visual evidence.",
                48,
            ),
            "assistant_output": _make_repeated_text(
                "The explanation should connect the point to local patterns, edges, boundaries, and the surrounding region.",
                32,
            ),
        },
    ]
    return collator(batch)


def _count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def benchmark(args: argparse.Namespace, config: Config, config_path: Path) -> None:
    distributed = init_distributed(config)
    set_seed(config.train.seed + distributed.rank)
    maybe_enable_tf32(config, distributed.device)

    dtype = resolve_dtype(config.train.dtype, distributed.device)
    rank0_print(distributed, f"benchmark[{args.tag}]: loading tokenizer and model")
    model, tokenizer, token_info = build_model_and_tokenizer(config, dtype)
    maybe_enable_activation_checkpointing(model, config.train.activation_checkpointing)
    model.to(distributed.device)
    sliding_window = getattr(model.language_model.config, "sliding_window", config.data.max_seq_len)
    model = apply_fsdp2(model, config, distributed)
    model = maybe_compile_model(model, config.train.torch_compile)
    model.train()

    optimizer, trainable_parameters = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config.train.warmup_steps, max(1, args.warmup_steps + args.steps))
    optimizer.zero_grad(set_to_none=True)

    batch = _build_fixed_batch(
        config,
        tokenizer,
        token_info,
        sliding_window=sliding_window,
    )
    assert_consistent_image_presence_across_ranks(
        batch["images"],
        distributed,
        fsdp_enabled=config.fsdp.enabled,
    )

    autocast_enabled = distributed.device.type == "cuda" and dtype != torch.float32

    total_metrics: dict[str, float] = {}
    measured_steps = 0

    for optimizer_step in range(args.warmup_steps + args.steps):
        step_metrics: dict[str, float] = {}
        window_examples = 0
        window_input_tokens = 0
        window_target_tokens = 0
        running_loss_numerator = 0.0
        running_aux_loss_total = 0.0
        window_start_time = time.perf_counter()

        for micro_index in range(config.train.grad_accum_steps):
            should_sync = micro_index == config.train.grad_accum_steps - 1
            maybe_set_gradient_sync(model, should_sync)

            examples_in_batch = int(batch["example_counts"].sum().item())
            input_tokens = int(batch["sequence_lengths"].sum().item())
            target_tokens = int((batch["labels"] != -100).sum().item())

            if config.profiling.enabled and hasattr(model, "reset_profile_stats"):
                model.reset_profile_stats()

            maybe_synchronize_timing(distributed.device, config.profiling.synchronize_cuda)
            forward_start_time = time.perf_counter()
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
                loss_value = float(output.loss.detach().float().item())
                aux_loss_value = (
                    float(output.aux_loss.detach().float().item())
                    if getattr(output, "aux_loss", None) is not None
                    else 0.0
                )
                loss = output.loss / config.train.grad_accum_steps
            maybe_synchronize_timing(distributed.device, config.profiling.synchronize_cuda)
            forward_time = time.perf_counter() - forward_start_time

            model_profile_stats = {}
            if config.profiling.enabled and hasattr(model, "pop_profile_stats"):
                model_profile_stats = model.pop_profile_stats()

            maybe_synchronize_timing(distributed.device, config.profiling.synchronize_cuda)
            backward_start_time = time.perf_counter()
            loss.backward()
            maybe_synchronize_timing(distributed.device, config.profiling.synchronize_cuda)
            backward_time = time.perf_counter() - backward_start_time

            optimizer_time = 0.0
            if should_sync:
                maybe_synchronize_timing(distributed.device, config.profiling.synchronize_cuda)
                optimizer_start_time = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, config.train.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                maybe_synchronize_timing(distributed.device, config.profiling.synchronize_cuda)
                optimizer_time = time.perf_counter() - optimizer_start_time

            accumulate_timing(
                step_metrics,
                {
                    "profile/data_wait_s": 0.0,
                    "profile/forward_s": forward_time,
                    "profile/backward_s": backward_time,
                    "profile/optimizer_s": optimizer_time,
                    **model_profile_stats,
                },
            )
            window_examples += examples_in_batch
            window_input_tokens += input_tokens
            window_target_tokens += target_tokens
            running_loss_numerator += loss_value * target_tokens
            running_aux_loss_total += aux_loss_value

        elapsed = max(1e-6, time.perf_counter() - window_start_time)
        reduced = torch.tensor(
            [
                running_loss_numerator,
                float(window_examples),
                float(window_input_tokens),
                float(window_target_tokens),
                elapsed,
                running_aux_loss_total,
            ],
            dtype=torch.float64,
            device=distributed.device,
        )
        if distributed.world_size > 1:
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        global_examples = reduced[1].item()
        global_input_tokens = reduced[2].item()
        global_target_tokens = reduced[3].item()
        max_elapsed = reduced[4].item() / distributed.world_size
        loss_value = reduced[0].item() / max(1.0, global_target_tokens)
        aux_loss_value = reduced[5].item() / float(distributed.world_size * config.train.grad_accum_steps)

        if optimizer_step >= args.warmup_steps:
            measured_steps += 1
            accumulate_timing(total_metrics, step_metrics)
            rank0_print(
                distributed,
                f"benchmark[{args.tag}] step={optimizer_step - args.warmup_steps + 1} "
                f"loss={loss_value:.4f} aux={aux_loss_value:.4f} "
                f"examples={int(global_examples)} "
                f"input_toks/s={global_input_tokens / max_elapsed:.1f} "
                f"target_toks/s={global_target_tokens / max_elapsed:.1f}",
            )
            rank0_print(
                distributed,
                f"benchmark[{args.tag}] timing "
                f"fwd={step_metrics.get('profile/forward_s', 0.0):.3f}s "
                f"bwd={step_metrics.get('profile/backward_s', 0.0):.3f}s "
                f"opt={step_metrics.get('profile/optimizer_s', 0.0):.3f}s "
                f"embed={step_metrics.get('model/embed_input_ids_s', 0.0):.3f}s "
                f"vision={step_metrics.get('model/vision_encode_s', 0.0):.3f}s "
                f"bridge={step_metrics.get('model/bridge_project_s', 0.0):.3f}s "
                f"inject={step_metrics.get('model/inject_embeddings_s', 0.0):.3f}s "
                f"lm={step_metrics.get('model/language_model_forward_s', 0.0):.3f}s",
            )

    if measured_steps > 0:
        averaged = {key: value / float(measured_steps) for key, value in total_metrics.items()}
        rank0_print(
            distributed,
            f"benchmark[{args.tag}] summary "
            f"trainable_parameters={_count_trainable_parameters(model)} "
            f"grad_accum_steps={config.train.grad_accum_steps} "
            f"max_seq_len={config.data.max_seq_len} "
            f"packing_buffer_size={config.data.packing_buffer_size}",
        )
        rank0_print(
            distributed,
            f"benchmark[{args.tag}] avg_timing "
            f"fwd={averaged.get('profile/forward_s', 0.0):.3f}s "
            f"bwd={averaged.get('profile/backward_s', 0.0):.3f}s "
            f"opt={averaged.get('profile/optimizer_s', 0.0):.3f}s "
            f"embed={averaged.get('model/embed_input_ids_s', 0.0):.3f}s "
            f"vision={averaged.get('model/vision_encode_s', 0.0):.3f}s "
            f"bridge={averaged.get('model/bridge_project_s', 0.0):.3f}s "
            f"inject={averaged.get('model/inject_embeddings_s', 0.0):.3f}s "
            f"lm={averaged.get('model/language_model_forward_s', 0.0):.3f}s",
        )

    barrier_if_distributed(distributed)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if maybe_relaunch_with_torchrun(config_path, config):
        return
    benchmark(args, config, config_path)


if __name__ == "__main__":
    main()
