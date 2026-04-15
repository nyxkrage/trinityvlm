from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .tokenizer_utils import save_trinity_chat_template
from .train_fsdp2 import (
    Config,
    DistributedContext,
    TrainState,
    accumulate_timing,
    barrier_if_distributed,
    build_dataloader,
    build_model_and_tokenizer,
    build_optimizer,
    build_scheduler,
    create_dataloader_batches,
    init_distributed,
    load_config,
    maybe_compile_model,
    maybe_enable_activation_checkpointing,
    maybe_enable_tf32,
    maybe_init_wandb,
    maybe_log_wandb,
    maybe_relaunch_with_torchrun,
    maybe_synchronize_timing,
    parse_args,
    rank0_print,
    resolve_dtype,
    resolve_resume_checkpoint,
    resolve_train_data_section,
    resolve_validation_data_section,
    set_seed,
    _fast_forward_scheduler,
    _optimizer_kind,
    _scheduler_state_is_compatible,
)


def _unwrap_ddp_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _normalize_parameter_name(name: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def apply_ddp(model, config: Config, distributed: DistributedContext):
    if distributed.world_size <= 1:
        return model

    kwargs: dict[str, Any] = {
        "broadcast_buffers": config.ddp.broadcast_buffers,
        "find_unused_parameters": config.ddp.find_unused_parameters,
        "gradient_as_bucket_view": config.ddp.gradient_as_bucket_view,
        "static_graph": config.ddp.static_graph,
    }
    if config.ddp.bucket_cap_mb is not None:
        kwargs["bucket_cap_mb"] = config.ddp.bucket_cap_mb
    if distributed.device.type == "cuda":
        kwargs["device_ids"] = [distributed.local_rank]
        kwargs["output_device"] = distributed.local_rank
    return DistributedDataParallel(model, **kwargs)


def _ddp_sync_context(model, should_sync: bool):
    if isinstance(model, DistributedDataParallel) and not should_sync:
        return model.no_sync()
    return nullcontext()


def _gather_trainable_delta_state_dict(model) -> dict[str, torch.Tensor]:
    base_model = _unwrap_ddp_model(model)
    delta_state_dict: dict[str, torch.Tensor] = {}
    for name, parameter in base_model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta_state_dict[_normalize_parameter_name(name)] = parameter.detach().cpu().contiguous()
    return delta_state_dict


def _align_delta_state_dict_for_model(model, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_keys = list(model.state_dict().keys())
    if model_keys and all(key.startswith("_orig_mod.") for key in model_keys):
        return {f"_orig_mod.{key}" if not key.startswith("_orig_mod.") else key: value for key, value in state_dict.items()}
    return state_dict


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    train_state: TrainState,
    config: Config,
    distributed: DistributedContext,
) -> Path:
    checkpoint_dir = Path(config.train.output_dir) / "checkpoints" / f"step-{train_state.global_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    delta_state_dict = _gather_trainable_delta_state_dict(model)
    barrier_if_distributed(distributed)

    if distributed.is_main_process:
        save_file(delta_state_dict, checkpoint_dir / "trainable.safetensors")
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(
            {
                "global_step": train_state.global_step,
                "micro_step": train_state.micro_step,
                "packed_sequences": train_state.packed_sequences,
                "examples_seen": train_state.examples_seen,
                "input_tokens_seen": train_state.input_tokens_seen,
                "target_tokens_seen": train_state.target_tokens_seen,
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_dir / "trainer_state.pt",
        )
        metadata = {
            "format": "trinity_vlm_trainable_delta_v1",
            "trainable_parameter_names": sorted(delta_state_dict),
            "optimizer": {
                "kind": _optimizer_kind(optimizer),
                "config_value": config.train.optimizer,
            },
        }
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (Path(config.train.output_dir) / "latest_checkpoint.txt").write_text(str(checkpoint_dir))

    barrier_if_distributed(distributed)
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: Path,
    model,
    optimizer,
    scheduler,
    distributed: DistributedContext,
) -> TrainState:
    base_model = _unwrap_ddp_model(model)
    delta_state_dict = _align_delta_state_dict_for_model(
        base_model,
        load_file(checkpoint_dir / "trainable.safetensors"),
    )
    missing_keys, unexpected_keys = base_model.load_state_dict(delta_state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading checkpoint: {unexpected_keys}")
    trainable_missing = [
        _normalize_parameter_name(name)
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad and name in missing_keys
    ]
    if trainable_missing:
        raise RuntimeError(f"Missing trainable keys while loading checkpoint: {trainable_missing}")

    optimizer_state = torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=False)
    try:
        optimizer.load_state_dict(optimizer_state)
        optimizer_state_loaded = True
    except ValueError:
        optimizer_state_loaded = False

    trainer_state = torch.load(checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    scheduler_state = trainer_state.get("scheduler")
    if optimizer_state_loaded and _scheduler_state_is_compatible(scheduler, scheduler_state):
        scheduler.load_state_dict(scheduler_state)
    else:
        _fast_forward_scheduler(scheduler, int(trainer_state["global_step"]))

    barrier_if_distributed(distributed)
    return TrainState(
        global_step=int(trainer_state["global_step"]),
        micro_step=int(trainer_state["micro_step"]),
        packed_sequences=int(trainer_state["packed_sequences"]),
        examples_seen=int(trainer_state["examples_seen"]),
        input_tokens_seen=int(trainer_state["input_tokens_seen"]),
        target_tokens_seen=int(trainer_state["target_tokens_seen"]),
    )


def evaluate(
    model,
    dataloader,
    data_section,
    distributed: DistributedContext,
    *,
    dtype: torch.dtype,
    max_batches: int,
) -> dict[str, float]:
    if max_batches <= 0:
        raise ValueError("validation.max_batches must be positive.")

    autocast_enabled = distributed.device.type == "cuda" and dtype != torch.float32
    iterator = create_dataloader_batches(
        dataloader,
        data_section,
        distributed,
        label="validation",
    )

    model.eval()
    metrics = torch.zeros(5, dtype=torch.float64, device=distributed.device)
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(iterator):
            if batch_index >= max_batches:
                break

            examples_in_batch = int(batch["example_counts"].sum().item())
            input_tokens = int(batch["sequence_lengths"].sum().item())
            target_tokens = int((batch["labels"] != -100).sum().item())

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

            metrics[0] += float(output.loss.detach().float().item() * target_tokens)
            metrics[1] += float(target_tokens)
            metrics[2] += float(examples_in_batch)
            metrics[3] += float(input_tokens)
            metrics[4] += 1.0

    if distributed.world_size > 1:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    elapsed = max(1e-6, time.perf_counter() - start_time)
    model.train()

    total_target_tokens = max(1.0, metrics[1].item())
    total_batches = max(1.0, metrics[4].item())
    return {
        "validation/loss": metrics[0].item() / total_target_tokens,
        "validation/examples": metrics[2].item(),
        "validation/input_tokens": metrics[3].item(),
        "validation/target_tokens": metrics[1].item(),
        "validation/batches": metrics[4].item(),
        "validation/input_toks_per_s": metrics[3].item() / elapsed,
        "validation/target_toks_per_s": metrics[1].item() / elapsed,
        "validation/examples_per_batch": metrics[2].item() / total_batches,
    }


def train(config_path: Path, config: Config) -> None:
    if config.fsdp.enabled:
        raise RuntimeError("train_ddp.py requires [fsdp].enabled = false.")

    distributed = init_distributed(config)
    if config.validation.enabled and config.validation.every <= 0:
        raise ValueError("validation.every must be positive when validation is enabled.")

    set_seed(config.train.seed + distributed.rank)
    maybe_enable_tf32(config, distributed.device)

    train_data = resolve_train_data_section(config)
    validation_data = resolve_validation_data_section(config)
    dtype = resolve_dtype(config.train.dtype, distributed.device)
    rank0_print(distributed, "startup: loading tokenizer and model")
    base_model, tokenizer, token_info = build_model_and_tokenizer(config, dtype)
    rank0_print(distributed, "startup: model loaded")
    maybe_enable_activation_checkpointing(base_model, config.train.activation_checkpointing)
    base_model.to(distributed.device)
    rank0_print(distributed, f"startup: model moved to {distributed.device}")
    sliding_window = getattr(base_model.language_model.config, "sliding_window", train_data.max_seq_len)
    base_model = maybe_compile_model(base_model, config.train.torch_compile)
    model = apply_ddp(base_model, config, distributed)
    rank0_print(distributed, "startup: ddp setup complete")
    model.train()

    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if distributed.is_main_process:
        tokenizer.save_pretrained(output_dir / "tokenizer")
        save_trinity_chat_template(output_dir / "tokenizer")
        (output_dir / "resolved_config.toml").write_text(config_path.read_text())
        (output_dir / "token_info.txt").write_text(str(token_info))

    wandb_run = maybe_init_wandb(config, config_path, distributed)

    rank0_print(distributed, "startup: building training dataloader")
    train_dataloader = build_dataloader(
        train_data,
        tokenizer,
        token_info,
        image_seq_len=_unwrap_ddp_model(model).config.image_seq_len,
        sliding_window=sliding_window,
        distributed=distributed,
    )

    validation_dataloader = None
    if validation_data is not None:
        rank0_print(distributed, "startup: building validation dataloader")
        validation_dataloader = build_dataloader(
            validation_data,
            tokenizer,
            token_info,
            image_seq_len=_unwrap_ddp_model(model).config.image_seq_len,
            sliding_window=sliding_window,
            distributed=distributed,
        )

    rank0_print(
        distributed,
        "startup: training data "
        f"dataset={train_data.dataset} split={train_data.split} "
        f"shards={train_data.shard_start}:{train_data.shard_end}",
    )
    if validation_data is not None:
        rank0_print(
            distributed,
            "startup: validation data "
            f"dataset={validation_data.dataset} split={validation_data.split} "
            f"shards={validation_data.shard_start}:{validation_data.shard_end} "
            f"max_batches={config.validation.max_batches}",
        )

    rank0_print(distributed, "startup: building optimizer")
    optimizer, trainable_parameters = build_optimizer(_unwrap_ddp_model(model), config)
    scheduler = build_scheduler(optimizer, config.train.warmup_steps, config.train.max_steps)
    train_state = TrainState()

    resume_checkpoint = resolve_resume_checkpoint(config)
    if resume_checkpoint is not None:
        train_state = load_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            distributed,
        )
        rank0_print(distributed, f"resumed from {resume_checkpoint}")

    rank0_print(
        distributed,
        "trainable_parameters="
        f"{sum(parameter.numel() for parameter in trainable_parameters)} "
        f"world_size={distributed.world_size} "
        f"max_seq_len={train_data.max_seq_len}",
    )

    autocast_enabled = distributed.device.type == "cuda" and dtype != torch.float32
    running_loss_numerator = 0.0
    running_aux_loss_total = 0.0
    window_examples = 0
    window_target_tokens = 0
    window_input_tokens = 0
    window_timing_metrics: dict[str, float] = {}
    window_start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    batch_iterator = iter(
        create_dataloader_batches(
            train_dataloader,
            train_data,
            distributed,
            label="training",
        )
    )
    while True:
        data_wait_start_time = time.perf_counter()
        try:
            batch = next(batch_iterator)
        except StopIteration:
            break
        data_wait_time = time.perf_counter() - data_wait_start_time

        train_state.micro_step += 1
        should_sync = train_state.micro_step % config.train.grad_accum_steps == 0

        packed_sequences = len(batch["images"])
        examples_in_batch = int(batch["example_counts"].sum().item())
        input_tokens = int(batch["sequence_lengths"].sum().item())
        target_tokens = int((batch["labels"] != -100).sum().item())

        if config.profiling.enabled and hasattr(_unwrap_ddp_model(model), "reset_profile_stats"):
            _unwrap_ddp_model(model).reset_profile_stats()

        maybe_synchronize_timing(distributed.device, config.profiling.synchronize_cuda)
        forward_start_time = time.perf_counter()
        with _ddp_sync_context(model, should_sync):
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
            if config.profiling.enabled and hasattr(_unwrap_ddp_model(model), "pop_profile_stats"):
                model_profile_stats = _unwrap_ddp_model(model).pop_profile_stats()

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
            window_timing_metrics,
            {
                "profile/data_wait_s": data_wait_time,
                "profile/forward_s": forward_time,
                "profile/backward_s": backward_time,
                "profile/optimizer_s": optimizer_time,
                **model_profile_stats,
            },
        )

        running_loss_numerator += loss_value * target_tokens
        running_aux_loss_total += aux_loss_value
        window_examples += examples_in_batch
        window_input_tokens += input_tokens
        window_target_tokens += target_tokens

        train_state.packed_sequences += packed_sequences
        train_state.examples_seen += examples_in_batch
        train_state.input_tokens_seen += input_tokens
        train_state.target_tokens_seen += target_tokens

        if not should_sync:
            continue

        train_state.global_step += 1
        if train_state.global_step % config.train.log_every == 0:
            elapsed = max(1e-6, time.perf_counter() - window_start_time)
            train_metrics = {
                "train/loss": running_loss_numerator / max(1.0, float(window_target_tokens)),
                "train/aux_loss": running_aux_loss_total
                / float(config.train.log_every * config.train.grad_accum_steps),
                "train/lr": scheduler.get_last_lr()[0],
                "train/examples": window_examples,
                "train/input_toks_per_s": window_input_tokens / elapsed,
                "train/target_toks_per_s": window_target_tokens / elapsed,
                "train/examples_seen": train_state.examples_seen,
                "train/input_tokens_seen": train_state.input_tokens_seen,
                "train/target_tokens_seen": train_state.target_tokens_seen,
                "train/packed_sequences_seen": train_state.packed_sequences,
            }
            if config.profiling.enabled:
                timing_metrics = {
                    key: value / float(config.train.log_every)
                    for key, value in window_timing_metrics.items()
                }
                train_metrics.update(timing_metrics)
            rank0_print(
                distributed,
                "step="
                f"{train_state.global_step} "
                f"loss={train_metrics['train/loss']:.4f} "
                f"lr={train_metrics['train/lr']:.6g} "
                f"examples={window_examples} "
                f"input_toks/s={train_metrics['train/input_toks_per_s']:.1f} "
                f"target_toks/s={train_metrics['train/target_toks_per_s']:.1f}",
            )
            if config.profiling.enabled:
                rank0_print(
                    distributed,
                    "timing "
                    f"data={train_metrics.get('profile/data_wait_s', 0.0):.3f}s "
                    f"fwd={train_metrics.get('profile/forward_s', 0.0):.3f}s "
                    f"bwd={train_metrics.get('profile/backward_s', 0.0):.3f}s "
                    f"opt={train_metrics.get('profile/optimizer_s', 0.0):.3f}s "
                    f"embed={train_metrics.get('model/embed_input_ids_s', 0.0):.3f}s "
                    f"vision={train_metrics.get('model/vision_encode_s', 0.0):.3f}s "
                    f"bridge={train_metrics.get('model/bridge_project_s', 0.0):.3f}s "
                    f"inject={train_metrics.get('model/inject_embeddings_s', 0.0):.3f}s "
                    f"lm={train_metrics.get('model/language_model_forward_s', 0.0):.3f}s",
                )
            maybe_log_wandb(wandb_run, train_metrics, step=train_state.global_step)
            running_loss_numerator = 0.0
            running_aux_loss_total = 0.0
            window_examples = 0
            window_input_tokens = 0
            window_target_tokens = 0
            window_timing_metrics = {}
            window_start_time = time.perf_counter()

        if (
            validation_data is not None
            and validation_dataloader is not None
            and train_state.global_step % config.validation.every == 0
        ):
            validation_metrics = evaluate(
                model,
                validation_dataloader,
                validation_data,
                distributed,
                dtype=dtype,
                max_batches=config.validation.max_batches,
            )
            rank0_print(
                distributed,
                "validation "
                f"step={train_state.global_step} "
                f"loss={validation_metrics['validation/loss']:.4f} "
                f"batches={validation_metrics['validation/batches']:.0f} "
                f"examples={validation_metrics['validation/examples']:.0f}",
            )
            maybe_log_wandb(wandb_run, validation_metrics, step=train_state.global_step)

        if (
            config.checkpoint.save_every > 0
            and train_state.global_step % config.checkpoint.save_every == 0
        ):
            checkpoint_dir = save_checkpoint(
                model,
                optimizer,
                scheduler,
                train_state,
                config,
                distributed,
            )
            rank0_print(distributed, f"saved checkpoint to {checkpoint_dir}")

        if train_state.global_step >= config.train.max_steps:
            break

    if (
        config.checkpoint.save_every > 0
        and train_state.global_step % config.checkpoint.save_every != 0
    ):
        checkpoint_dir = save_checkpoint(
            model,
            optimizer,
            scheduler,
            train_state,
            config,
            distributed,
        )
        rank0_print(distributed, f"saved checkpoint to {checkpoint_dir}")

    rank0_print(distributed, f"finished training at step={train_state.global_step}")
    if wandb_run is not None:
        maybe_log_wandb(wandb_run, {"train/final_step": train_state.global_step}, step=train_state.global_step)
        wandb_run.finish()
    barrier_if_distributed(distributed)
    if distributed.world_size > 1:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if maybe_relaunch_with_torchrun(config_path, config):
        return
    train(config_path, config)


if __name__ == "__main__":
    main()
