#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from trinity_vlm.train_fsdp2 import load_config, resolve_train_data_section, resolve_validation_data_section


_PIXMO_PARQUET_SHARD_COUNT = 75
_PIXMO_CAP_DATASET = "anthracite-org/pixmo-cap-images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch TrinityVLM training datasets into the HF/datasets cache.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/bridge_mix_pixmo_nemotron_local_20260414.toml",
        help="Path to the training TOML config.",
    )
    return parser.parse_args()


def _prefetch_pixmo_caption_shards(dataset_name: str, split: str, shard_start: int, shard_end: int) -> None:
    if dataset_name != _PIXMO_CAP_DATASET:
        return
    if shard_end > _PIXMO_PARQUET_SHARD_COUNT:
        raise ValueError(f"shard_end={shard_end} exceeds PixMo shard count {_PIXMO_PARQUET_SHARD_COUNT}.")
    for shard_index in range(shard_start, shard_end):
        filename = f"data/{split}-{shard_index:05d}-of-{_PIXMO_PARQUET_SHARD_COUNT:05d}.parquet"
        hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=filename,
        )
        print(f"prefetched {dataset_name}:{filename}", flush=True)


def _prefetch_dataset_repo(
    dataset_name: str,
    *,
    shard_start: int | None = None,
    shard_end: int | None = None,
) -> None:
    allowed_suffixes = (".json", ".jsonl", ".md", ".txt")
    filenames = []
    for filename in list_repo_files(dataset_name, repo_type="dataset"):
        if filename == ".gitattributes" or filename.endswith(allowed_suffixes):
            filenames.append(filename)
            continue
        if not filename.startswith("data/"):
            continue
        if shard_start is None or shard_end is None:
            filenames.append(filename)
            continue
        stem = Path(filename).name
        parts = stem.split("-")
        if len(parts) < 4:
            continue
        shard_index = int(parts[1])
        if shard_start <= shard_index < shard_end:
            filenames.append(filename)
    for filename in filenames:
        hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            filename=filename,
        )
        print(f"prefetched {dataset_name}:{filename}", flush=True)
    print(f"prefetched dataset repo {dataset_name}", flush=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    train_data = resolve_train_data_section(config)
    validation_data = resolve_validation_data_section(config)

    if train_data.shard_end is not None:
        _prefetch_pixmo_caption_shards(
            train_data.dataset,
            train_data.split,
            train_data.shard_start,
            train_data.shard_end,
        )
    if validation_data is not None and validation_data.shard_end is not None:
        _prefetch_pixmo_caption_shards(
            validation_data.dataset,
            validation_data.split,
            validation_data.shard_start,
            validation_data.shard_end,
        )

    if train_data.cap_qa_mix_weight > 0:
        _prefetch_dataset_repo(
            train_data.cap_qa_dataset,
            shard_start=train_data.cap_qa_shard_start,
            shard_end=train_data.cap_qa_shard_end,
        )

    if train_data.point_explanations_mix_weight > 0:
        _prefetch_dataset_repo(
            train_data.point_explanations_dataset,
            shard_start=train_data.point_explanations_shard_start,
            shard_end=train_data.point_explanations_shard_end,
        )


if __name__ == "__main__":
    main()
