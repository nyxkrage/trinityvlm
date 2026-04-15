from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import tomllib
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from .data import (
    MixedCaptionChatIterable,
    NemotronChatIterable,
    PackedCaptionCollator,
    PixMoCaptionIterable,
    PixMoQuestionAnswerIterable,
)
from .graft import TrinityMoondreamGraftConfig, TrinityMoondreamGraftModel
from .tokenizer_utils import load_trinity_tokenizer, save_trinity_chat_template


@dataclass
class ModelSection:
    trinity_model: str = "arcee-ai/Trinity-Nano-Preview"
    moondream_model: str = "moondream/moondream3-preview"
    projector_hidden_dim: int = 2048
    router_aux_loss_coef: float = 0.001
    use_cut_cross_entropy: bool = False
    cut_cross_entropy_impl: str = "cce"
    cut_cross_entropy_filter_eps: float | str | None = "auto"
    cut_cross_entropy_accum_e_fp32: bool = False
    cut_cross_entropy_accum_c_fp32: bool = False
    cut_cross_entropy_filter_e_grad: bool = True
    cut_cross_entropy_filter_c_grad: bool = True
    local_files_only: bool = False
    revision: str | None = None


@dataclass
class DataSection:
    dataset: str = "anthracite-org/pixmo-cap-images"
    dataset_config_name: str | None = None
    split: str = "train"
    cap_qa_dataset: str = "anthracite-org/pixmo-cap-qa-images"
    cap_qa_dataset_config_name: str | None = None
    cap_qa_split: str = "train"
    cap_qa_shard_start: int = 0
    cap_qa_shard_end: int | None = None
    point_explanations_dataset: str = "anthracite-org/pixmo-point-explanations-images"
    point_explanations_dataset_config_name: str | None = None
    point_explanations_split: str = "train"
    point_explanations_shard_start: int = 0
    point_explanations_shard_end: int | None = None
    prompt_text: str = "Describe the image in detail."
    user_message_template: str = "{image}\n{prompt}"
    max_prompt_tokens: int = 64
    max_caption_tokens: int = 384
    max_chat_input_tokens: int = 1536
    max_chat_output_tokens: int = 384
    max_seq_len: int = 2048
    packing_buffer_size: int = 32
    shuffle_buffer_size: int = 256
    limit: int | None = None
    streaming: bool = True
    cache_shards_locally: bool = False
    local_files_only: bool = False
    shard_start: int = 0
    shard_end: int | None = None
    caption_mix_weight: float = 1.0
    cap_qa_mix_weight: float = 0.0
    chat_mix_weight: float = 0.0
    chat_with_irrelevant_image_mix_weight: float = 0.0
    point_explanations_mix_weight: float = 0.0
    chat_dataset: str = "nvidia/Llama-Nemotron-Post-Training-Dataset"
    chat_dataset_config_name: str | None = "SFT"
    chat_split: str = "chat"
    chat_local_path: str | None = None


@dataclass
class ValidationSection:
    enabled: bool = False
    dataset: str | None = None
    dataset_config_name: str | None = None
    split: str | None = None
    cap_qa_dataset: str | None = None
    cap_qa_dataset_config_name: str | None = None
    cap_qa_split: str | None = None
    cap_qa_shard_start: int | None = None
    cap_qa_shard_end: int | None = None
    point_explanations_dataset: str | None = None
    point_explanations_dataset_config_name: str | None = None
    point_explanations_split: str | None = None
    point_explanations_shard_start: int | None = None
    point_explanations_shard_end: int | None = None
    prompt_text: str | None = None
    user_message_template: str | None = None
    every: int = 100
    max_batches: int = 8
    shuffle_buffer_size: int = 0
    limit: int | None = None
    streaming: bool | None = None
    packing_buffer_size: int | None = None
    cache_shards_locally: bool | None = None
    local_files_only: bool | None = None
    shard_start: int = 0
    shard_end: int | None = None
    caption_mix_weight: float | None = None
    cap_qa_mix_weight: float | None = None
    chat_mix_weight: float | None = None
    chat_with_irrelevant_image_mix_weight: float | None = None
    point_explanations_mix_weight: float | None = None
    chat_dataset: str | None = None
    chat_dataset_config_name: str | None = None
    chat_split: str | None = None
    chat_local_path: str | None = None


@dataclass
class WandbSection:
    enabled: bool = False
    project: str = "trinity-vlm"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str | None = None
    run_id: str | None = None
    resume: str | None = "allow"
    dir: str | None = None


@dataclass
class ProfilingSection:
    enabled: bool = False
    synchronize_cuda: bool = True


@dataclass
class TrainSection:
    output_dir: str = "checkpoints/fsdp2-trinity-vlm"
    max_steps: int = 1000
    grad_accum_steps: int = 1
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = "match_rms_adamw"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"
    device: str = "cuda"
    seed: int = 0
    log_every: int = 10
    activation_checkpointing: bool = True
    torch_compile: bool = False
    tf32: bool = True


@dataclass
class UnfreezeSection:
    train_vision: bool = False
    train_trinity: bool = False
    train_lm_head: bool = False
    unfreeze_last_n_layers: int = 0


@dataclass
class FSDPSection:
    enabled: bool = True
    reshard_after_forward: bool = True
    param_dtype: str | None = "bfloat16"
    reduce_dtype: str | None = "float32"
    output_dtype: str | None = None
    cast_forward_inputs: bool = True
    cpu_offload: bool = False


@dataclass
class DDPSection:
    broadcast_buffers: bool = False
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    bucket_cap_mb: int | None = None


@dataclass
class DistributedSection:
    nproc_per_node: int = 1
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    backend: str = "nccl"


@dataclass
class CheckpointSection:
    save_every: int = 200
    resume_from: str | None = None


@dataclass
class Config:
    model: ModelSection = field(default_factory=ModelSection)
    data: DataSection = field(default_factory=DataSection)
    validation: ValidationSection = field(default_factory=ValidationSection)
    wandb: WandbSection = field(default_factory=WandbSection)
    profiling: ProfilingSection = field(default_factory=ProfilingSection)
    train: TrainSection = field(default_factory=TrainSection)
    unfreeze: UnfreezeSection = field(default_factory=UnfreezeSection)
    fsdp: FSDPSection = field(default_factory=FSDPSection)
    ddp: DDPSection = field(default_factory=DDPSection)
    distributed: DistributedSection = field(default_factory=DistributedSection)
    checkpoint: CheckpointSection = field(default_factory=CheckpointSection)


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass
class ResolvedDataSection:
    dataset: str
    dataset_config_name: str | None
    split: str
    cap_qa_dataset: str
    cap_qa_dataset_config_name: str | None
    cap_qa_split: str
    cap_qa_shard_start: int
    cap_qa_shard_end: int | None
    point_explanations_dataset: str
    point_explanations_dataset_config_name: str | None
    point_explanations_split: str
    point_explanations_shard_start: int
    point_explanations_shard_end: int | None
    prompt_text: str
    user_message_template: str
    max_prompt_tokens: int
    max_caption_tokens: int
    max_chat_input_tokens: int
    max_chat_output_tokens: int
    max_seq_len: int
    packing_buffer_size: int
    shuffle_buffer_size: int
    limit: int | None
    streaming: bool
    cache_shards_locally: bool
    local_files_only: bool
    shard_start: int
    shard_end: int | None
    caption_mix_weight: float
    cap_qa_mix_weight: float
    chat_mix_weight: float
    chat_with_irrelevant_image_mix_weight: float
    point_explanations_mix_weight: float
    chat_dataset: str
    chat_dataset_config_name: str | None
    chat_split: str
    chat_local_path: str | None
    seed: int


@dataclass
class TrainState:
    global_step: int = 0
    micro_step: int = 0
    packed_sequences: int = 0
    examples_seen: int = 0
    input_tokens_seen: int = 0
    target_tokens_seen: int = 0


class HybridOptimizer:
    def __init__(self, optimizers: dict[str, Optimizer]) -> None:
        if not optimizers:
            raise ValueError("HybridOptimizer requires at least one sub-optimizer.")
        self.optimizers = optimizers
        self.param_groups = [group for optimizer in optimizers.values() for group in optimizer.param_groups]

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for name, optimizer in self.optimizers.items():
            if name in state_dict:
                optimizer.load_state_dict(state_dict[name])


class HybridScheduler:
    def __init__(self, schedulers: dict[str, Any]) -> None:
        if not schedulers:
            raise ValueError("HybridScheduler requires at least one sub-scheduler.")
        self.schedulers = schedulers

    def step(self) -> None:
        for scheduler in self.schedulers.values():
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        return {name: scheduler.state_dict() for name, scheduler in self.schedulers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for name, scheduler in self.schedulers.items():
            if name in state_dict:
                scheduler.load_state_dict(state_dict[name])

    def get_last_lr(self) -> list[float]:
        learning_rates: list[float] = []
        for scheduler in self.schedulers.values():
            learning_rates.extend(scheduler.get_last_lr())
        return learning_rates or [0.0]


def _read_toml_with_optional_shebang(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if text.startswith("#!"):
        newline_index = text.find("\n")
        text = "" if newline_index == -1 else text[newline_index + 1 :]
    return tomllib.loads(text)


def _parse_section(payload: dict[str, Any], key: str, cls):
    section_payload = payload.get(key, {})
    if not isinstance(section_payload, dict):
        raise TypeError(f"Config section {key!r} must be a table.")

    allowed = {field_info.name for field_info in fields(cls)}
    unknown = sorted(set(section_payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in [{key}]: {', '.join(unknown)}")
    return cls(**section_payload)


def load_config(path: Path) -> Config:
    payload = _read_toml_with_optional_shebang(path)
    allowed_top_level = {field_info.name for field_info in fields(Config)}
    unknown_top_level = sorted(set(payload) - allowed_top_level)
    if unknown_top_level:
        raise ValueError(f"Unknown top-level config sections: {', '.join(unknown_top_level)}")

    return Config(
        model=_parse_section(payload, "model", ModelSection),
        data=_parse_section(payload, "data", DataSection),
        validation=_parse_section(payload, "validation", ValidationSection),
        wandb=_parse_section(payload, "wandb", WandbSection),
        profiling=_parse_section(payload, "profiling", ProfilingSection),
        train=_parse_section(payload, "train", TrainSection),
        unfreeze=_parse_section(payload, "unfreeze", UnfreezeSection),
        fsdp=_parse_section(payload, "fsdp", FSDPSection),
        ddp=_parse_section(payload, "ddp", DDPSection),
        distributed=_parse_section(payload, "distributed", DistributedSection),
        checkpoint=_parse_section(payload, "checkpoint", CheckpointSection),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TrinityVLM with packed multimodal batches and FSDP2.")
    parser.add_argument("config", help="Path to a TOML config file. The file may start with a shebang.")
    return parser.parse_args(argv)


def maybe_relaunch_with_torchrun(config_path: Path, config: Config) -> bool:
    if os.environ.get("RANK") is not None:
        return False

    distributed_cfg = config.distributed
    if distributed_cfg.nproc_per_node <= 1 and distributed_cfg.nnodes <= 1:
        return False

    entrypoint = Path(sys.argv[0]).resolve()
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(distributed_cfg.nproc_per_node),
    ]
    if distributed_cfg.nnodes <= 1:
        command.append("--standalone")
    else:
        command.extend(
            [
                "--nnodes",
                str(distributed_cfg.nnodes),
                "--node_rank",
                str(distributed_cfg.node_rank),
                "--master_addr",
                distributed_cfg.master_addr,
                "--master_port",
                str(distributed_cfg.master_port),
            ]
        )
    command.extend([str(entrypoint), str(config_path)])
    subprocess.run(command, check=True)
    return True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "bfloat16" and device.type == "cpu":
        return torch.float32
    return getattr(torch, dtype_name)


def maybe_enable_tf32(config: Config, device: torch.device) -> None:
    if device.type != "cuda" or not config.train.tf32:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def maybe_synchronize_timing(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def accumulate_timing(metrics: dict[str, float], updates: dict[str, float]) -> None:
    for key, value in updates.items():
        metrics[key] = metrics.get(key, 0.0) + float(value)


def init_distributed(config: Config) -> DistributedContext:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if config.train.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        backend = config.distributed.backend
        if device.type == "cpu" and backend == "nccl":
            backend = "gloo"
        init_kwargs = {
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
        }
        if device.type == "cuda" and backend == "nccl":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def barrier_if_distributed(distributed: DistributedContext) -> None:
    if distributed.world_size > 1:
        if distributed.device.type == "cuda":
            dist.barrier(device_ids=[distributed.local_rank])
        else:
            dist.barrier()


def assert_consistent_image_presence_across_ranks(
    batch_images: list[list[Any]],
    distributed: DistributedContext,
    *,
    fsdp_enabled: bool,
) -> None:
    if not fsdp_enabled or distributed.world_size <= 1:
        return

    local_has_images = any(sample_images for sample_images in batch_images)
    local_flag = torch.tensor(
        1 if local_has_images else 0,
        device=distributed.device,
        dtype=torch.int32,
    )
    min_flag = local_flag.clone()
    max_flag = local_flag.clone()
    dist.all_reduce(min_flag, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_flag, op=dist.ReduceOp.MAX)
    if int(min_flag.item()) != int(max_flag.item()):
        raise RuntimeError(
            "Inconsistent image presence across ranks in an FSDP batch. "
            "Do not mix text-only samples with image-present samples across ranks."
        )


def rank0_print(distributed: DistributedContext, message: str) -> None:
    if distributed.is_main_process:
        print(message, flush=True)


def _coalesce(value, fallback):
    return fallback if value is None else value


def resolve_train_data_section(config: Config) -> ResolvedDataSection:
    return ResolvedDataSection(
        dataset=config.data.dataset,
        dataset_config_name=config.data.dataset_config_name,
        split=config.data.split,
        cap_qa_dataset=config.data.cap_qa_dataset,
        cap_qa_dataset_config_name=config.data.cap_qa_dataset_config_name,
        cap_qa_split=config.data.cap_qa_split,
        cap_qa_shard_start=config.data.cap_qa_shard_start,
        cap_qa_shard_end=config.data.cap_qa_shard_end,
        point_explanations_dataset=config.data.point_explanations_dataset,
        point_explanations_dataset_config_name=config.data.point_explanations_dataset_config_name,
        point_explanations_split=config.data.point_explanations_split,
        point_explanations_shard_start=config.data.point_explanations_shard_start,
        point_explanations_shard_end=config.data.point_explanations_shard_end,
        prompt_text=config.data.prompt_text,
        user_message_template=config.data.user_message_template,
        max_prompt_tokens=config.data.max_prompt_tokens,
        max_caption_tokens=config.data.max_caption_tokens,
        max_chat_input_tokens=config.data.max_chat_input_tokens,
        max_chat_output_tokens=config.data.max_chat_output_tokens,
        max_seq_len=config.data.max_seq_len,
        packing_buffer_size=config.data.packing_buffer_size,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
        limit=config.data.limit,
        streaming=config.data.streaming,
        cache_shards_locally=config.data.cache_shards_locally,
        local_files_only=config.data.local_files_only,
        shard_start=config.data.shard_start,
        shard_end=config.data.shard_end,
        caption_mix_weight=config.data.caption_mix_weight,
        cap_qa_mix_weight=config.data.cap_qa_mix_weight,
        chat_mix_weight=config.data.chat_mix_weight,
        chat_with_irrelevant_image_mix_weight=config.data.chat_with_irrelevant_image_mix_weight,
        point_explanations_mix_weight=config.data.point_explanations_mix_weight,
        chat_dataset=config.data.chat_dataset,
        chat_dataset_config_name=config.data.chat_dataset_config_name,
        chat_split=config.data.chat_split,
        chat_local_path=config.data.chat_local_path,
        seed=config.train.seed,
    )


def resolve_validation_data_section(config: Config) -> ResolvedDataSection | None:
    if not config.validation.enabled:
        return None

    return ResolvedDataSection(
        dataset=_coalesce(config.validation.dataset, config.data.dataset),
        dataset_config_name=_coalesce(
            config.validation.dataset_config_name,
            config.data.dataset_config_name,
        ),
        split=_coalesce(config.validation.split, config.data.split),
        cap_qa_dataset=_coalesce(config.validation.cap_qa_dataset, config.data.cap_qa_dataset),
        cap_qa_dataset_config_name=_coalesce(
            config.validation.cap_qa_dataset_config_name,
            config.data.cap_qa_dataset_config_name,
        ),
        cap_qa_split=_coalesce(config.validation.cap_qa_split, config.data.cap_qa_split),
        cap_qa_shard_start=_coalesce(config.validation.cap_qa_shard_start, config.data.cap_qa_shard_start),
        cap_qa_shard_end=_coalesce(config.validation.cap_qa_shard_end, config.data.cap_qa_shard_end),
        point_explanations_dataset=_coalesce(
            config.validation.point_explanations_dataset,
            config.data.point_explanations_dataset,
        ),
        point_explanations_dataset_config_name=_coalesce(
            config.validation.point_explanations_dataset_config_name,
            config.data.point_explanations_dataset_config_name,
        ),
        point_explanations_split=_coalesce(
            config.validation.point_explanations_split,
            config.data.point_explanations_split,
        ),
        point_explanations_shard_start=_coalesce(
            config.validation.point_explanations_shard_start,
            config.data.point_explanations_shard_start,
        ),
        point_explanations_shard_end=_coalesce(
            config.validation.point_explanations_shard_end,
            config.data.point_explanations_shard_end,
        ),
        prompt_text=_coalesce(config.validation.prompt_text, config.data.prompt_text),
        user_message_template=_coalesce(
            config.validation.user_message_template,
            config.data.user_message_template,
        ),
        max_prompt_tokens=config.data.max_prompt_tokens,
        max_caption_tokens=config.data.max_caption_tokens,
        max_chat_input_tokens=config.data.max_chat_input_tokens,
        max_chat_output_tokens=config.data.max_chat_output_tokens,
        max_seq_len=config.data.max_seq_len,
        packing_buffer_size=_coalesce(config.validation.packing_buffer_size, config.data.packing_buffer_size),
        shuffle_buffer_size=config.validation.shuffle_buffer_size,
        limit=config.validation.limit,
        streaming=_coalesce(config.validation.streaming, config.data.streaming),
        cache_shards_locally=_coalesce(
            config.validation.cache_shards_locally,
            config.data.cache_shards_locally,
        ),
        local_files_only=_coalesce(
            config.validation.local_files_only,
            config.data.local_files_only,
        ),
        shard_start=config.validation.shard_start,
        shard_end=config.validation.shard_end,
        caption_mix_weight=_coalesce(
            config.validation.caption_mix_weight,
            config.data.caption_mix_weight,
        ),
        cap_qa_mix_weight=_coalesce(
            config.validation.cap_qa_mix_weight,
            config.data.cap_qa_mix_weight,
        ),
        chat_mix_weight=_coalesce(
            config.validation.chat_mix_weight,
            config.data.chat_mix_weight,
        ),
        chat_with_irrelevant_image_mix_weight=_coalesce(
            config.validation.chat_with_irrelevant_image_mix_weight,
            config.data.chat_with_irrelevant_image_mix_weight,
        ),
        point_explanations_mix_weight=_coalesce(
            config.validation.point_explanations_mix_weight,
            config.data.point_explanations_mix_weight,
        ),
        chat_dataset=_coalesce(config.validation.chat_dataset, config.data.chat_dataset),
        chat_dataset_config_name=_coalesce(
            config.validation.chat_dataset_config_name,
            config.data.chat_dataset_config_name,
        ),
        chat_split=_coalesce(config.validation.chat_split, config.data.chat_split),
        chat_local_path=_coalesce(config.validation.chat_local_path, config.data.chat_local_path),
        seed=config.train.seed + 10_000,
    )


def maybe_init_wandb(config: Config, config_path: Path, distributed: DistributedContext):
    if not config.wandb.enabled or not distributed.is_main_process:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb.enabled=true but the wandb package is not installed.") from exc

    init_kwargs = {
        "project": config.wandb.project,
        "entity": config.wandb.entity,
        "name": config.wandb.name,
        "group": config.wandb.group,
        "tags": config.wandb.tags,
        "mode": config.wandb.mode,
        "id": config.wandb.run_id,
        "resume": config.wandb.resume,
        "dir": config.wandb.dir or config.train.output_dir,
        "config": asdict(config),
    }
    init_kwargs = {
        key: value
        for key, value in init_kwargs.items()
        if value is not None and value != []
    }
    run = wandb.init(**init_kwargs)
    if run is not None:
        run.config.update(
            {
                "config_path": str(config_path),
                "world_size": distributed.world_size,
            },
            allow_val_change=True,
        )
    return run


def maybe_log_wandb(run, metrics: dict[str, float | int], step: int) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def _build_lambda_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer: Optimizer | HybridOptimizer, warmup_steps: int, total_steps: int):
    if isinstance(optimizer, HybridOptimizer):
        return HybridScheduler(
            {
                name: _build_lambda_scheduler(sub_optimizer, warmup_steps, total_steps)
                for name, sub_optimizer in optimizer.optimizers.items()
            }
        )
    return _build_lambda_scheduler(optimizer, warmup_steps, total_steps)


def _resolve_dtype_or_none(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    return getattr(torch, name)

def apply_fsdp2(model: TrinityMoondreamGraftModel, config: Config, distributed: DistributedContext):
    if not config.fsdp.enabled or distributed.world_size <= 1:
        return model

    if distributed.device.type != "cuda":
        raise RuntimeError("FSDP2 training currently requires CUDA devices.")

    from torch.distributed.device_mesh import init_device_mesh

    try:
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
            register_fsdp_forward_method,
        )
    except ImportError:
        from torch.distributed._composable.fsdp import (  # type: ignore[attr-defined]
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            fully_shard,
            register_fsdp_forward_method,
        )

    mesh = init_device_mesh(distributed.device.type, (distributed.world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=_resolve_dtype_or_none(config.fsdp.param_dtype),
        reduce_dtype=_resolve_dtype_or_none(config.fsdp.reduce_dtype),
        output_dtype=_resolve_dtype_or_none(config.fsdp.output_dtype),
        cast_forward_inputs=config.fsdp.cast_forward_inputs,
    )
    offload_policy = CPUOffloadPolicy() if config.fsdp.cpu_offload else None
    fsdp_kwargs = {
        "mesh": mesh,
        "mp_policy": mp_policy,
        "offload_policy": offload_policy,
        "reshard_after_forward": config.fsdp.reshard_after_forward,
    }

    for block in model.vision_tower.blocks:
        fully_shard(block, **fsdp_kwargs)
    fully_shard(model.vision_tower, **fsdp_kwargs)
    register_fsdp_forward_method(model.vision_tower, "encode_images")

    for layer in model.language_model.model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model.language_model, **fsdp_kwargs)
    register_fsdp_forward_method(model.language_model, "embed_input_ids")
    fully_shard(model, **fsdp_kwargs)
    return model


def _checkpoint_options():
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=True,
    )


def _load_checkpoint_options():
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=True,
        strict=False,
        broadcast_from_rank0=True,
    )


def _tensor_to_cpu_contiguous(value: torch.Tensor) -> torch.Tensor:
    if hasattr(value, "full_tensor"):
        value = value.full_tensor()
    return value.detach().cpu().contiguous()


def _gather_trainable_delta_state_dict(model: TrinityMoondreamGraftModel) -> dict[str, torch.Tensor]:
    delta_state_dict: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta_state_dict[name.removeprefix("_orig_mod.")] = _tensor_to_cpu_contiguous(parameter)
    return delta_state_dict


def _optimizer_kind(optimizer: Optimizer | HybridOptimizer) -> str:
    if isinstance(optimizer, HybridOptimizer):
        return "hybrid_muon_adamw"
    return optimizer.__class__.__name__.lower()


def _should_use_muon(name: str, parameter: torch.Tensor) -> bool:
    lowered_name = name.lower()
    if parameter.ndim != 2:
        return False
    if "embed" in lowered_name:
        return False
    if "lm_head" in lowered_name:
        return False
    return True


def build_optimizer(
    model: TrinityMoondreamGraftModel,
    config: Config,
) -> tuple[Optimizer | HybridOptimizer, list[torch.nn.Parameter]]:
    named_trainable_parameters = [
        (name.removeprefix("_orig_mod."), parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_trainable_parameters:
        raise ValueError("No trainable parameters were selected.")

    optimizer_name = config.train.optimizer.lower()
    if optimizer_name == "adamw":
        return (
            AdamW(
                [parameter for _, parameter in named_trainable_parameters],
                lr=config.train.learning_rate,
                weight_decay=config.train.weight_decay,
            ),
            [parameter for _, parameter in named_trainable_parameters],
        )

    if optimizer_name != "muon":
        raise ValueError(f"Unsupported train.optimizer={config.train.optimizer!r}.")

    muon_parameters: list[tuple[str, torch.nn.Parameter]] = []
    adamw_parameters: list[torch.nn.Parameter] = []
    for name, parameter in named_trainable_parameters:
        if _should_use_muon(name, parameter):
            muon_parameters.append((name, parameter))
        else:
            adamw_parameters.append(parameter)

    optimizers: dict[str, Optimizer] = {}
    if muon_parameters:
        optimizers["muon"] = torch.optim.Muon(
            muon_parameters,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            momentum=config.train.muon_momentum,
            nesterov=config.train.muon_nesterov,
            ns_steps=config.train.muon_ns_steps,
            adjust_lr_fn=config.train.muon_adjust_lr_fn,
        )
    if adamw_parameters:
        optimizers["adamw"] = AdamW(
            adamw_parameters,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

    return HybridOptimizer(optimizers), [parameter for _, parameter in named_trainable_parameters]


def _get_optimizer_state_dict(
    model: TrinityMoondreamGraftModel,
    optimizer: Optimizer | HybridOptimizer,
    *,
    options,
) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    if isinstance(optimizer, HybridOptimizer):
        return {
            name: get_optimizer_state_dict(model, sub_optimizer, options=options)
            for name, sub_optimizer in optimizer.optimizers.items()
        }
    return get_optimizer_state_dict(model, optimizer, options=options)


def _set_optimizer_state_dict(
    model: TrinityMoondreamGraftModel,
    optimizer: Optimizer | HybridOptimizer,
    optimizer_state_dict: dict[str, Any],
    *,
    options,
) -> bool:
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    if isinstance(optimizer, HybridOptimizer):
        if not isinstance(optimizer_state_dict, dict):
            return False
        loaded_any = False
        for name, sub_optimizer in optimizer.optimizers.items():
            sub_state_dict = optimizer_state_dict.get(name)
            if sub_state_dict is None:
                continue
            set_optimizer_state_dict(model, sub_optimizer, sub_state_dict, options=options)
            loaded_any = True
        return loaded_any

    if not isinstance(optimizer_state_dict, dict):
        return False
    if "state" not in optimizer_state_dict or "param_groups" not in optimizer_state_dict:
        return False
    set_optimizer_state_dict(model, optimizer, optimizer_state_dict, options=options)
    return True


def _scheduler_state_is_compatible(scheduler, scheduler_state: Any) -> bool:
    if isinstance(scheduler, HybridScheduler):
        if not isinstance(scheduler_state, dict):
            return False
        return any(name in scheduler_state for name in scheduler.schedulers)
    return isinstance(scheduler_state, dict) and "last_epoch" in scheduler_state


def _fast_forward_scheduler(scheduler, global_step: int) -> None:
    if global_step <= 0:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(global_step):
            scheduler.step()


def save_checkpoint(
    model: TrinityMoondreamGraftModel,
    optimizer: Optimizer | HybridOptimizer,
    scheduler,
    train_state: TrainState,
    config: Config,
    distributed: DistributedContext,
) -> Path:
    checkpoint_dir = Path(config.train.output_dir) / "checkpoints" / f"step-{train_state.global_step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    options = _checkpoint_options()
    delta_state_dict = _gather_trainable_delta_state_dict(model)
    optimizer_state_dict = _get_optimizer_state_dict(model, optimizer, options=options)
    barrier_if_distributed(distributed)

    if distributed.is_main_process:
        save_file(
            {name: _tensor_to_cpu_contiguous(tensor) for name, tensor in delta_state_dict.items()},
            checkpoint_dir / "trainable.safetensors",
        )
        torch.save(optimizer_state_dict, checkpoint_dir / "optimizer.pt")
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
            "config": asdict(model.graft_config),
        }
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (Path(config.train.output_dir) / "latest_checkpoint.txt").write_text(str(checkpoint_dir))

    barrier_if_distributed(distributed)
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: Path,
    model: TrinityMoondreamGraftModel,
    optimizer: Optimizer | HybridOptimizer,
    scheduler,
    distributed: DistributedContext,
) -> TrainState:
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    options = _load_checkpoint_options()
    delta_state_dict = (
        load_file(checkpoint_dir / "trainable.safetensors")
        if distributed.is_main_process
        else {}
    )
    optimizer_state_dict = (
        torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu", weights_only=False)
        if distributed.is_main_process
        else {}
    )
    set_model_state_dict(model, delta_state_dict, options=options)
    optimizer_state_loaded = _set_optimizer_state_dict(
        model,
        optimizer,
        optimizer_state_dict,
        options=options,
    )
    barrier_if_distributed(distributed)

    trainer_state = torch.load(checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    scheduler_state = trainer_state.get("scheduler")
    if optimizer_state_loaded and _scheduler_state_is_compatible(scheduler, scheduler_state):
        scheduler.load_state_dict(scheduler_state)
    else:
        _fast_forward_scheduler(scheduler, int(trainer_state["global_step"]))
    return TrainState(
        global_step=int(trainer_state["global_step"]),
        micro_step=int(trainer_state["micro_step"]),
        packed_sequences=int(trainer_state["packed_sequences"]),
        examples_seen=int(trainer_state["examples_seen"]),
        input_tokens_seen=int(trainer_state["input_tokens_seen"]),
        target_tokens_seen=int(trainer_state["target_tokens_seen"]),
    )


def resolve_resume_checkpoint(config: Config) -> Path | None:
    resume_from = config.checkpoint.resume_from
    if not resume_from:
        return None
    if resume_from == "latest":
        latest_path = Path(config.train.output_dir) / "latest_checkpoint.txt"
        if not latest_path.exists():
            raise FileNotFoundError(f"Could not find {latest_path} to resume from latest checkpoint.")
        return Path(latest_path.read_text().strip())
    return Path(resume_from)


def maybe_set_gradient_sync(model, enabled: bool) -> None:
    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(enabled)


def maybe_enable_activation_checkpointing(model: TrinityMoondreamGraftModel, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model.language_model, "gradient_checkpointing_enable"):
        model.language_model.gradient_checkpointing_enable()
    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()


def maybe_compile_model(model, enabled: bool):
    if not enabled:
        return model
    return torch.compile(model)


def should_stagger_dataloader_startup(
    data_section: ResolvedDataSection,
    distributed: DistributedContext,
) -> bool:
    return (
        distributed.world_size > 1
        and data_section.streaming
        and not data_section.local_files_only
        and data_section.dataset == "anthracite-org/pixmo-cap-images"
        and (
            data_section.caption_mix_weight > 0
            or data_section.chat_with_irrelevant_image_mix_weight > 0
        )
    )


def create_dataloader_batches(
    dataloader: DataLoader,
    data_section: ResolvedDataSection,
    distributed: DistributedContext,
    *,
    label: str,
):
    def iter_nonempty_batches():
        for batch in dataloader:
            if batch is None:
                continue
            yield batch

    if not should_stagger_dataloader_startup(data_section, distributed):
        return iter_nonempty_batches()

    rank0_print(distributed, f"startup: warming {label} dataloader rank-by-rank")
    iterator = None
    first_batch = None
    for warmup_rank in range(distributed.world_size):
        if distributed.rank == warmup_rank:
            iterator = iter_nonempty_batches()
            first_batch = next(iterator)
        barrier_if_distributed(distributed)

    if iterator is None or first_batch is None:
        raise RuntimeError("Failed to initialize the staggered streaming dataloader.")

    return itertools.chain([first_batch], iterator)


def build_dataloader(
    data_section: ResolvedDataSection,
    tokenizer,
    token_info: dict[str, int],
    *,
    image_seq_len: int,
    sliding_window: int,
    distributed: DistributedContext,
) -> DataLoader:
    caption_dataset = None
    if (
        data_section.caption_mix_weight > 0
        or data_section.chat_with_irrelevant_image_mix_weight > 0
    ):
        caption_dataset = PixMoCaptionIterable(
            dataset_name=data_section.dataset,
            dataset_config_name=data_section.dataset_config_name,
            split=data_section.split,
            streaming=data_section.streaming,
            shuffle_buffer_size=data_section.shuffle_buffer_size,
            seed=data_section.seed,
            limit=data_section.limit,
            rank=distributed.rank,
            world_size=distributed.world_size,
            cache_shards_locally=data_section.cache_shards_locally,
            local_files_only=data_section.local_files_only,
            shard_start=data_section.shard_start,
            shard_end=data_section.shard_end,
        )

    cap_qa_dataset = None
    if data_section.cap_qa_mix_weight > 0:
        cap_qa_dataset = PixMoQuestionAnswerIterable(
            dataset_name=data_section.cap_qa_dataset,
            dataset_config_name=data_section.cap_qa_dataset_config_name,
            question_field="question",
            answer_field="answer",
            example_type="cap_qa",
            split=data_section.cap_qa_split,
            streaming=data_section.streaming,
            shuffle_buffer_size=data_section.shuffle_buffer_size,
            seed=data_section.seed,
            limit=data_section.limit,
            rank=distributed.rank,
            world_size=distributed.world_size,
            cache_shards_locally=data_section.cache_shards_locally,
            local_files_only=data_section.local_files_only,
            shard_start=data_section.cap_qa_shard_start,
            shard_end=data_section.cap_qa_shard_end,
        )

    point_explanations_dataset = None
    if data_section.point_explanations_mix_weight > 0:
        point_explanations_dataset = PixMoQuestionAnswerIterable(
            dataset_name=data_section.point_explanations_dataset,
            dataset_config_name=data_section.point_explanations_dataset_config_name,
            question_field="question",
            answer_field="response",
            example_type="point_explanation",
            split=data_section.point_explanations_split,
            streaming=data_section.streaming,
            shuffle_buffer_size=data_section.shuffle_buffer_size,
            seed=data_section.seed,
            limit=data_section.limit,
            rank=distributed.rank,
            world_size=distributed.world_size,
            cache_shards_locally=data_section.cache_shards_locally,
            local_files_only=data_section.local_files_only,
            shard_start=data_section.point_explanations_shard_start,
            shard_end=data_section.point_explanations_shard_end,
        )

    chat_dataset = None
    if (
        data_section.chat_mix_weight > 0
        or data_section.chat_with_irrelevant_image_mix_weight > 0
    ):
        chat_dataset = NemotronChatIterable(
            dataset_name=data_section.chat_dataset,
            dataset_config_name=data_section.chat_dataset_config_name,
            split=data_section.chat_split,
            local_path=data_section.chat_local_path,
            streaming=data_section.streaming,
            shuffle_buffer_size=data_section.shuffle_buffer_size,
            seed=data_section.seed,
            limit=data_section.limit,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )

    if (
        data_section.cap_qa_mix_weight > 0
        or data_section.point_explanations_mix_weight > 0
        or data_section.chat_mix_weight > 0
        or data_section.chat_with_irrelevant_image_mix_weight > 0
    ):
        dataset = MixedCaptionChatIterable(
            caption_dataset=caption_dataset,
            cap_qa_dataset=cap_qa_dataset,
            point_explanations_dataset=point_explanations_dataset,
            chat_dataset=chat_dataset,
            caption_weight=data_section.caption_mix_weight,
            cap_qa_weight=data_section.cap_qa_mix_weight,
            chat_weight=data_section.chat_mix_weight,
            chat_with_irrelevant_image_weight=data_section.chat_with_irrelevant_image_mix_weight,
            point_explanations_weight=data_section.point_explanations_mix_weight,
            seed=data_section.seed + distributed.rank,
        )
    elif caption_dataset is not None:
        dataset = caption_dataset
    elif cap_qa_dataset is not None:
        dataset = cap_qa_dataset
    elif point_explanations_dataset is not None:
        dataset = point_explanations_dataset
    else:
        raise ValueError("No training data sources were enabled by the dataset mix weights.")

    collator = PackedCaptionCollator(
        tokenizer=tokenizer,
        prompt_text=data_section.prompt_text,
        user_message_template=data_section.user_message_template,
        max_prompt_tokens=data_section.max_prompt_tokens,
        max_caption_tokens=data_section.max_caption_tokens,
        max_chat_input_tokens=data_section.max_chat_input_tokens,
        max_chat_output_tokens=data_section.max_chat_output_tokens,
        max_seq_len=data_section.max_seq_len,
        sliding_window=sliding_window,
        vision_start_token_id=int(token_info["vision_start_token_id"]),
        image_token_id=int(token_info["image_token_id"]),
        vision_end_token_id=int(token_info["vision_end_token_id"]),
        image_seq_len=image_seq_len,
        target_examples=data_section.packing_buffer_size,
    )
    return DataLoader(
        dataset,
        batch_size=data_section.packing_buffer_size * 2,
        collate_fn=collator,
        num_workers=0,
    )


def evaluate(
    model: TrinityMoondreamGraftModel,
    dataloader: DataLoader,
    data_section: ResolvedDataSection,
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

            assert_consistent_image_presence_across_ranks(
                batch["images"],
                distributed,
                fsdp_enabled=distributed.world_size > 1,
            )

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


def build_model_and_tokenizer(config: Config, dtype: torch.dtype):
    print("startup: loading Trinity tokenizer", flush=True)
    tokenizer = load_trinity_tokenizer(
        config.model.trinity_model,
        revision=config.model.revision,
        local_files_only=config.model.local_files_only,
    )
    print("startup: Trinity tokenizer loaded", flush=True)

    print("startup: constructing TrinityMoondreamGraftModel", flush=True)
    model = TrinityMoondreamGraftModel(
        TrinityMoondreamGraftConfig(
            trinity_model_name=config.model.trinity_model,
            moondream_model_name=config.model.moondream_model,
            projector_hidden_dim=config.model.projector_hidden_dim,
            router_aux_loss_coef=config.model.router_aux_loss_coef,
            use_cut_cross_entropy=config.model.use_cut_cross_entropy,
            cut_cross_entropy_impl=config.model.cut_cross_entropy_impl,
            cut_cross_entropy_filter_eps=config.model.cut_cross_entropy_filter_eps,
            cut_cross_entropy_accum_e_fp32=config.model.cut_cross_entropy_accum_e_fp32,
            cut_cross_entropy_accum_c_fp32=config.model.cut_cross_entropy_accum_c_fp32,
            cut_cross_entropy_filter_e_grad=config.model.cut_cross_entropy_filter_e_grad,
            cut_cross_entropy_filter_c_grad=config.model.cut_cross_entropy_filter_c_grad,
            freeze_trinity=not config.unfreeze.train_trinity,
            freeze_vision=not config.unfreeze.train_vision,
            train_lm_head=config.unfreeze.train_lm_head,
            unfreeze_last_n_layers=config.unfreeze.unfreeze_last_n_layers,
        ),
        torch_dtype=dtype,
        local_files_only=config.model.local_files_only,
        revision=config.model.revision,
    )
    print("startup: TrinityMoondreamGraftModel constructed", flush=True)
    model.configure_profiling(
        enabled=config.profiling.enabled,
        synchronize_cuda=config.profiling.synchronize_cuda,
    )
    token_info = model.ensure_image_special_tokens(tokenizer)
    return model, tokenizer, token_info


def train(config_path: Path, config: Config) -> None:
    distributed = init_distributed(config)
    if distributed.world_size > 1 and not config.fsdp.enabled:
        raise RuntimeError("Distributed training requires fsdp.enabled = true.")
    if config.validation.enabled and config.validation.every <= 0:
        raise ValueError("validation.every must be positive when validation is enabled.")

    set_seed(config.train.seed + distributed.rank)
    maybe_enable_tf32(config, distributed.device)

    train_data = resolve_train_data_section(config)
    validation_data = resolve_validation_data_section(config)
    dtype = resolve_dtype(config.train.dtype, distributed.device)
    rank0_print(distributed, "startup: loading tokenizer and model")
    model, tokenizer, token_info = build_model_and_tokenizer(config, dtype)
    rank0_print(distributed, "startup: model loaded")
    maybe_enable_activation_checkpointing(model, config.train.activation_checkpointing)
    model.to(distributed.device)
    rank0_print(distributed, f"startup: model moved to {distributed.device}")
    sliding_window = getattr(model.language_model.config, "sliding_window", train_data.max_seq_len)
    model = apply_fsdp2(model, config, distributed)
    rank0_print(distributed, "startup: fsdp setup complete")
    model = maybe_compile_model(model, config.train.torch_compile)
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
        image_seq_len=model.config.image_seq_len,
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
            image_seq_len=model.config.image_seq_len,
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
    optimizer, trainable_parameters = build_optimizer(model, config)
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
        maybe_set_gradient_sync(model, should_sync)

        packed_sequences = len(batch["images"])
        examples_in_batch = int(batch["example_counts"].sum().item())
        input_tokens = int(batch["sequence_lengths"].sum().item())
        target_tokens = int((batch["labels"] != -100).sum().item())

        assert_consistent_image_presence_across_ranks(
            batch["images"],
            distributed,
            fsdp_enabled=config.fsdp.enabled,
        )

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
            maybe_log_wandb(
                wandb_run,
                validation_metrics,
                step=train_state.global_step,
            )

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
