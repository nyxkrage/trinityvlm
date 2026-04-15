from __future__ import annotations

import copy
import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import DownloadConfig, load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
import pyarrow.parquet as pq
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from .chat_content import build_user_message_content, truncate_text_to_token_budget

_PIXMO_PARQUET_SHARD_COUNT = 75
_PIXMO_DATASET_NAME = "anthracite-org/pixmo-cap-images"
_PIXMO_CAP_QA_DATASET_NAME = "anthracite-org/pixmo-cap-qa-images"
_PIXMO_POINT_EXPLANATIONS_DATASET_NAME = "anthracite-org/pixmo-point-explanations-images"
_PIXMO_CAP_QA_PARQUET_SHARD_COUNT = 25
_PIXMO_POINT_EXPLANATIONS_PARQUET_SHARD_COUNT = 4
_NEMOTRON_DATASET_NAME = "nvidia/Llama-Nemotron-Post-Training-Dataset"
_NEMOTRON_DATASET_CONFIG = "SFT"


class PixMoCaptionIterable(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: str | None = None,
        split: str = "train",
        streaming: bool = True,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
        limit: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        cache_shards_locally: bool = False,
        local_files_only: bool = False,
        shard_start: int = 0,
        shard_end: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.limit = limit
        self.rank = rank
        self.world_size = world_size
        self.cache_shards_locally = cache_shards_locally
        self.local_files_only = local_files_only
        self.shard_start = shard_start
        self.shard_end = shard_end

    def _resolve_direct_parquet_files(self) -> list[str] | None:
        if self.dataset_name != _PIXMO_DATASET_NAME:
            return None

        if self.shard_start < 0:
            raise ValueError("shard_start must be non-negative.")
        shard_end = self.shard_end or _PIXMO_PARQUET_SHARD_COUNT
        if shard_end > _PIXMO_PARQUET_SHARD_COUNT:
            raise ValueError(
                f"shard_end={shard_end} exceeds PixMo shard count {_PIXMO_PARQUET_SHARD_COUNT}."
            )
        if self.shard_start >= shard_end:
            raise ValueError(
                f"shard_start={self.shard_start} must be smaller than shard_end={shard_end}."
            )

        all_shard_filenames = [
            f"data/{self.split}-{shard_index:05d}-of-{_PIXMO_PARQUET_SHARD_COUNT:05d}.parquet"
            for shard_index in range(_PIXMO_PARQUET_SHARD_COUNT)
        ]
        shard_filenames = all_shard_filenames[self.shard_start : shard_end]
        shard_filenames = shard_filenames[self.rank :: self.world_size]
        if self.limit is not None:
            shard_filenames = shard_filenames[:1]

        files = (
            shard_filenames
            if self.cache_shards_locally
            else [f"hf://datasets/{self.dataset_name}/{filename}" for filename in shard_filenames]
        )

        if not files:
            raise ValueError(
                f"No PixMo parquet shards were assigned to rank {self.rank} "
                f"out of world size {self.world_size}."
            )

        return files

    def _iter_cached_pixmo_parquet_rows(self, parquet_files: list[str]) -> Iterable[dict[str, Any]]:
        columns = ["image", "image_url", "caption", "transcripts"]
        for filename in parquet_files:
            parquet_path = hf_hub_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                filename=filename,
                local_files_only=self.local_files_only,
            )
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(
                batch_size=64,
                columns=columns,
                use_threads=False,
            ):
                for row in batch.to_pylist():
                    yield {
                        "image": row["image"],
                        "caption": row["caption"],
                        "image_url": row.get("image_url"),
                        "transcripts": row.get("transcripts"),
                    }

    def _shuffle_examples(self, examples: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        if self.shuffle_buffer_size <= 0:
            yield from examples
            return

        rng = random.Random(self.seed + self.rank)
        buffer: list[dict[str, Any]] = []
        for example in examples:
            buffer.append(example)
            if len(buffer) < self.shuffle_buffer_size:
                continue
            yield buffer.pop(rng.randrange(len(buffer)))

        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))

    def __iter__(self):
        direct_parquet_files = self._resolve_direct_parquet_files()
        should_modulo_shard = direct_parquet_files is None
        download_config = DownloadConfig(local_files_only=self.local_files_only)

        if direct_parquet_files is not None and self.cache_shards_locally:
            examples = self._iter_cached_pixmo_parquet_rows(direct_parquet_files)
            dataset = self._shuffle_examples(examples)
        elif direct_parquet_files is None:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                split=self.split,
                streaming=self.streaming,
                download_config=download_config,
            )
            if self.shuffle_buffer_size > 0:
                dataset = dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.seed,
                )
        else:
            dataset = load_dataset(
                "parquet",
                data_files={self.split: direct_parquet_files},
                split=self.split,
                streaming=self.streaming,
                download_config=download_config,
            )
            if self.shuffle_buffer_size > 0:
                dataset = dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.seed,
                )

        yielded = 0
        for index, example in enumerate(dataset):
            if should_modulo_shard and index % self.world_size != self.rank:
                continue
            if self.limit is not None and yielded >= self.limit:
                break
            yielded += 1
            yield {
                "example_type": "caption",
                "image": example["image"],
                "caption": example["caption"],
                "image_url": example.get("image_url"),
                "transcripts": example.get("transcripts"),
            }


class PixMoQuestionAnswerIterable(IterableDataset):
    def __init__(
        self,
        *,
        dataset_name: str,
        question_field: str,
        answer_field: str,
        example_type: str,
        dataset_config_name: str | None = None,
        split: str = "train",
        streaming: bool = True,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
        limit: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        cache_shards_locally: bool = False,
        local_files_only: bool = False,
        shard_start: int = 0,
        shard_end: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.question_field = question_field
        self.answer_field = answer_field
        self.example_type = example_type
        self.split = split
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.limit = limit
        self.rank = rank
        self.world_size = world_size
        self.cache_shards_locally = cache_shards_locally
        self.local_files_only = local_files_only
        self.shard_start = shard_start
        self.shard_end = shard_end

    def _parquet_shard_count(self) -> int | None:
        if self.dataset_name == _PIXMO_CAP_QA_DATASET_NAME:
            return _PIXMO_CAP_QA_PARQUET_SHARD_COUNT
        if self.dataset_name == _PIXMO_POINT_EXPLANATIONS_DATASET_NAME:
            return _PIXMO_POINT_EXPLANATIONS_PARQUET_SHARD_COUNT
        return None

    def _resolve_direct_parquet_files(self) -> list[str] | None:
        shard_count = self._parquet_shard_count()
        if shard_count is None:
            return None

        if self.shard_start < 0:
            raise ValueError("shard_start must be non-negative.")
        shard_end = self.shard_end or shard_count
        if shard_end > shard_count:
            raise ValueError(f"shard_end={shard_end} exceeds shard count {shard_count}.")
        if self.shard_start >= shard_end:
            raise ValueError(
                f"shard_start={self.shard_start} must be smaller than shard_end={shard_end}."
            )

        shard_filenames = [
            f"data/{self.split}-{shard_index:05d}-of-{shard_count:05d}.parquet"
            for shard_index in range(self.shard_start, shard_end)
        ]
        shard_filenames = shard_filenames[self.rank :: self.world_size]
        if self.limit is not None:
            shard_filenames = shard_filenames[:1]

        files = (
            shard_filenames
            if self.cache_shards_locally
            else [f"hf://datasets/{self.dataset_name}/{filename}" for filename in shard_filenames]
        )

        if not files:
            raise ValueError(
                f"No parquet shards were assigned to rank {self.rank} "
                f"out of world size {self.world_size}."
            )
        return files

    def _iter_cached_parquet_rows(self, parquet_files: list[str]) -> Iterable[dict[str, Any]]:
        columns = ["image", "image_url", self.question_field, self.answer_field]
        for filename in parquet_files:
            parquet_path = hf_hub_download(
                repo_id=self.dataset_name,
                repo_type="dataset",
                filename=filename,
                local_files_only=self.local_files_only,
            )
            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(
                batch_size=64,
                columns=columns,
                use_threads=False,
            ):
                for row in batch.to_pylist():
                    yield {
                        "image": row["image"],
                        "image_url": row.get("image_url"),
                        self.question_field: row[self.question_field],
                        self.answer_field: row[self.answer_field],
                    }

    def __iter__(self):
        direct_parquet_files = self._resolve_direct_parquet_files()
        should_modulo_shard = direct_parquet_files is None
        download_config = DownloadConfig(local_files_only=self.local_files_only)

        if direct_parquet_files is not None and self.cache_shards_locally:
            examples = self._iter_cached_parquet_rows(direct_parquet_files)
            if self.shuffle_buffer_size > 0:
                buffer_rng = random.Random(self.seed + self.rank)
                buffer: list[dict[str, Any]] = []
                def shuffled_examples():
                    for example in examples:
                        buffer.append(example)
                        if len(buffer) < self.shuffle_buffer_size:
                            continue
                        yield buffer.pop(buffer_rng.randrange(len(buffer)))
                    while buffer:
                        yield buffer.pop(buffer_rng.randrange(len(buffer)))
                dataset = shuffled_examples()
            else:
                dataset = examples
        elif direct_parquet_files is None:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                split=self.split,
                streaming=self.streaming,
                download_config=download_config,
            )
            if self.shuffle_buffer_size > 0:
                dataset = dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.seed,
                )
        else:
            dataset = load_dataset(
                "parquet",
                data_files={self.split: direct_parquet_files},
                split=self.split,
                streaming=self.streaming,
                download_config=download_config,
            )
            if self.shuffle_buffer_size > 0:
                dataset = dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.seed,
                )

        yielded = 0
        for index, example in enumerate(dataset):
            if should_modulo_shard and index % self.world_size != self.rank:
                continue
            if self.limit is not None and yielded >= self.limit:
                break
            yielded += 1
            yield {
                "example_type": self.example_type,
                "image": example["image"],
                "prompt_text": example[self.question_field],
                "assistant_output": example[self.answer_field],
                "image_url": example.get("image_url"),
            }


class NemotronChatIterable(IterableDataset):
    def __init__(
        self,
        dataset_name: str = _NEMOTRON_DATASET_NAME,
        dataset_config_name: str | None = _NEMOTRON_DATASET_CONFIG,
        split: str = "chat",
        local_path: str | None = None,
        streaming: bool = True,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
        limit: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.split = split
        self.local_path = local_path
        self.streaming = streaming
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.limit = limit
        self.rank = rank
        self.world_size = world_size

    def _iter_local_examples(self) -> Iterable[dict[str, Any]]:
        path = Path(self.local_path).expanduser()
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                yield {
                    "input": example["input_messages"],
                    "output": example["assistant_output"],
                    "system_prompt": example.get("system_prompt"),
                }

    def __iter__(self):
        if self.local_path:
            dataset = self._iter_local_examples()
        else:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config_name,
                split=self.split,
                streaming=self.streaming,
            )
            if self.shuffle_buffer_size > 0:
                dataset = dataset.shuffle(
                    buffer_size=self.shuffle_buffer_size,
                    seed=self.seed,
                )

        yielded = 0
        for index, example in enumerate(dataset):
            if index % self.world_size != self.rank:
                continue
            if self.limit is not None and yielded >= self.limit:
                break
            yielded += 1
            yield {
                "example_type": "chat",
                "input_messages": example["input"],
                "assistant_output": example["output"],
                "system_prompt": example.get("system_prompt"),
            }


class MixedCaptionChatIterable(IterableDataset):
    def __init__(
        self,
        *,
        caption_dataset: PixMoCaptionIterable | None,
        cap_qa_dataset: PixMoQuestionAnswerIterable | None,
        point_explanations_dataset: PixMoQuestionAnswerIterable | None,
        chat_dataset: NemotronChatIterable | None,
        caption_weight: float,
        cap_qa_weight: float,
        chat_weight: float,
        chat_with_irrelevant_image_weight: float,
        point_explanations_weight: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.caption_dataset = caption_dataset
        self.cap_qa_dataset = cap_qa_dataset
        self.point_explanations_dataset = point_explanations_dataset
        self.chat_dataset = chat_dataset
        self.caption_weight = caption_weight
        self.cap_qa_weight = cap_qa_weight
        self.chat_weight = chat_weight
        self.chat_with_irrelevant_image_weight = chat_with_irrelevant_image_weight
        self.point_explanations_weight = point_explanations_weight
        self.seed = seed

    def _next_from_factory(self, factory, iterator):
        if iterator is None:
            iterator = iter(factory())
        while True:
            try:
                return next(iterator), iterator
            except StopIteration:
                iterator = iter(factory())

    def __iter__(self):
        rng = random.Random(self.seed)

        sources: list[tuple[str, float]] = []
        if self.caption_dataset is not None and self.caption_weight > 0:
            sources.append(("caption", self.caption_weight))
        if self.cap_qa_dataset is not None and self.cap_qa_weight > 0:
            sources.append(("cap_qa", self.cap_qa_weight))
        if self.chat_dataset is not None and self.chat_weight > 0:
            sources.append(("chat", self.chat_weight))
        if (
            self.caption_dataset is not None
            and self.chat_dataset is not None
            and self.chat_with_irrelevant_image_weight > 0
        ):
            sources.append(("chat_with_irrelevant_image", self.chat_with_irrelevant_image_weight))
        if self.point_explanations_dataset is not None and self.point_explanations_weight > 0:
            sources.append(("point_explanation", self.point_explanations_weight))
        if not sources:
            raise ValueError("At least one positive dataset mix weight is required.")

        source_names = [name for name, _ in sources]
        source_weights = [weight for _, weight in sources]

        caption_factory = (lambda: self.caption_dataset) if self.caption_dataset is not None else None
        cap_qa_factory = (lambda: self.cap_qa_dataset) if self.cap_qa_dataset is not None else None
        point_explanations_factory = (
            (lambda: self.point_explanations_dataset)
            if self.point_explanations_dataset is not None
            else None
        )
        chat_factory = (lambda: self.chat_dataset) if self.chat_dataset is not None else None

        caption_iterator = None
        cap_qa_iterator = None
        chat_iterator = None
        irrelevant_chat_iterator = None
        irrelevant_image_iterator = None
        point_explanations_iterator = None

        while True:
            source_name = rng.choices(source_names, weights=source_weights, k=1)[0]
            if source_name == "caption":
                example, caption_iterator = self._next_from_factory(caption_factory, caption_iterator)
                yield example
                continue
            if source_name == "cap_qa":
                example, cap_qa_iterator = self._next_from_factory(cap_qa_factory, cap_qa_iterator)
                yield example
                continue
            if source_name == "chat":
                example, chat_iterator = self._next_from_factory(chat_factory, chat_iterator)
                yield example
                continue
            if source_name == "point_explanation":
                example, point_explanations_iterator = self._next_from_factory(
                    point_explanations_factory,
                    point_explanations_iterator,
                )
                yield example
                continue

            chat_example, irrelevant_chat_iterator = self._next_from_factory(
                chat_factory,
                irrelevant_chat_iterator,
            )
            image_example, irrelevant_image_iterator = self._next_from_factory(
                caption_factory,
                irrelevant_image_iterator,
            )
            mixed_example = dict(chat_example)
            mixed_example["example_type"] = "chat_with_irrelevant_image"
            mixed_example["irrelevant_image"] = image_example["image"]
            yield mixed_example


def _to_pil_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")

    raise TypeError(f"Unsupported image payload type: {type(value)!r}")


def _try_to_pil_image(value: Any) -> Image.Image | None:
    try:
        return _to_pil_image(value)
    except (Image.DecompressionBombError, OSError, ValueError, TypeError):
        return None


def _build_multimodal_user_content(
    tokenizer,
    *,
    prompt_text: str,
    user_message_template: str,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    return build_user_message_content(
        tokenizer,
        prompt_text=prompt_text,
        user_message_template=user_message_template,
        max_prompt_tokens=max_prompt_tokens,
        include_image=True,
        truncate_prompt=False,
    )


def _build_single_turn_multimodal_messages(
    tokenizer,
    *,
    prompt_text: str,
    user_message_template: str,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": _build_multimodal_user_content(
                tokenizer,
                prompt_text=prompt_text,
                user_message_template=user_message_template,
                max_prompt_tokens=max_prompt_tokens,
            ),
        }
    ]


def _normalize_text_content(content: str) -> str:
    return content.strip()


def _normalize_structured_content(content: Any) -> Any:
    if isinstance(content, str):
        return _normalize_text_content(content)
    if isinstance(content, list):
        normalized_parts: list[Any] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    normalized_parts.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError(f"Unsupported structured message content item: {type(item)!r}")
            normalized_item = dict(item)
            if normalized_item.get("type") == "text" and "text" in normalized_item:
                normalized_item["text"] = _normalize_text_content(normalized_item["text"])
            normalized_parts.append(normalized_item)
        return normalized_parts
    return content


def _prepend_image_to_content(content: Any) -> list[dict[str, Any]]:
    normalized_content = _normalize_structured_content(content)
    image_part = {"type": "image"}
    newline_part = {"type": "text", "text": "\n"}

    if isinstance(normalized_content, str):
        normalized_content = normalized_content.strip()
        if normalized_content:
            return [image_part, {"type": "text", "text": f"\n{normalized_content}"}]
        return [image_part]

    if isinstance(normalized_content, list):
        if not normalized_content:
            return [image_part]
        return [image_part, newline_part, *normalized_content]

    raise TypeError(f"Unsupported message content type for image attachment: {type(content)!r}")


def _build_chat_messages(
    *,
    input_messages: list[dict[str, Any]],
    system_prompt: str | None,
    attach_irrelevant_image: bool,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": _normalize_text_content(system_prompt)})

    normalized_input_messages: list[dict[str, Any]] = []
    for raw_message in input_messages:
        normalized_input_messages.append(
            {
                "role": raw_message["role"],
                "content": _normalize_structured_content(raw_message["content"]),
            }
        )

    if attach_irrelevant_image:
        for message in reversed(normalized_input_messages):
            if message["role"] == "user":
                message["content"] = _prepend_image_to_content(message["content"])
                break
        else:
            raise ValueError("Cannot attach an irrelevant image because the example has no user message.")

    messages.extend(normalized_input_messages)
    return messages


def _render_chat_example(
    tokenizer,
    *,
    messages: list[dict[str, Any]],
    assistant_content: str,
    image_seq_len: int,
) -> dict[str, list[int]]:
    return tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": assistant_content}],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
        image_seq_len=image_seq_len,
    )


def _tokenize_text_without_truncation(
    tokenizer,
    text: str,
) -> list[int]:
    return tokenizer(
        text.strip(),
        add_special_tokens=False,
    )["input_ids"]


def _iter_content_text_fragments(content: Any) -> Iterable[str]:
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            yield stripped
        return
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    yield stripped
                continue
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                stripped = item["text"].strip()
                if stripped:
                    yield stripped


def _count_prompt_text_tokens(
    tokenizer,
    messages: list[dict[str, Any]],
) -> int:
    total = 0
    for message in messages:
        for fragment in _iter_content_text_fragments(message["content"]):
            total += len(_tokenize_text_without_truncation(tokenizer, fragment))
    return total


def _drop_oldest_history_turn(messages: list[dict[str, Any]]) -> bool:
    history_limit = len(messages) - 1
    for index, message in enumerate(messages[:history_limit]):
        if message["role"] == "system":
            continue

        delete_end = index + 1
        if (
            message["role"] == "user"
            and delete_end < history_limit
            and messages[delete_end]["role"] == "assistant"
        ):
            delete_end += 1

        del messages[index:delete_end]
        return True

    return False


def _fit_chat_messages_to_prompt_budget(
    tokenizer,
    *,
    messages: list[dict[str, Any]],
    max_prompt_tokens: int,
) -> list[dict[str, Any]] | None:
    if max_prompt_tokens <= 0:
        return copy.deepcopy(messages)

    fitted_messages = copy.deepcopy(messages)
    while True:
        if _count_prompt_text_tokens(tokenizer, fitted_messages) <= max_prompt_tokens:
            return fitted_messages
        if not _drop_oldest_history_turn(fitted_messages):
            return None


def _fit_chat_example_to_seq_len(
    tokenizer,
    *,
    messages: list[dict[str, Any]],
    assistant_content: str,
    image_seq_len: int,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]] | None:
    fitted_messages = copy.deepcopy(messages)
    while True:
        rendered = _render_chat_example(
            tokenizer,
            messages=fitted_messages,
            assistant_content=assistant_content,
            image_seq_len=image_seq_len,
        )
        if len(rendered["input_ids"]) <= max_seq_len:
            return fitted_messages, rendered
        if not _drop_oldest_history_turn(fitted_messages):
            return None


@dataclass
class CaptionCollator:
    tokenizer: Any
    prompt_text: str
    user_message_template: str
    max_prompt_tokens: int
    max_caption_tokens: int
    vision_start_token_id: int
    image_token_id: int
    vision_end_token_id: int
    image_seq_len: int

    def __post_init__(self) -> None:
        self.user_messages = [
            {
                "role": "user",
                "content": _build_multimodal_user_content(
                    self.tokenizer,
                    prompt_text=self.prompt_text,
                    user_message_template=self.user_message_template,
                    max_prompt_tokens=self.max_prompt_tokens,
                ),
            }
        ]

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [_to_pil_image(example["image"]) for example in batch]

        input_tensors: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        label_tensors: list[torch.Tensor] = []

        for example in batch:
            caption_text = truncate_text_to_token_budget(
                self.tokenizer,
                example["caption"],
                max_tokens=self.max_caption_tokens,
            )
            rendered = _render_chat_example(
                self.tokenizer,
                messages=self.user_messages,
                assistant_content=caption_text,
                image_seq_len=self.image_seq_len,
            )

            full_input_ids = torch.tensor(rendered["input_ids"], dtype=torch.long)
            full_attention_mask = torch.tensor(rendered["attention_mask"], dtype=torch.long)
            assistant_mask = torch.tensor(rendered["assistant_masks"], dtype=torch.bool)
            full_labels = torch.where(
                assistant_mask,
                full_input_ids,
                torch.full_like(full_input_ids, -100),
            )

            input_tensors.append(full_input_ids)
            attention_masks.append(full_attention_mask)
            label_tensors.append(full_labels)

        input_ids = pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class TokenizedCaptionSample:
    images: list[Image.Image]
    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.numel())


@dataclass
class PackedCaptionSample:
    images: list[Image.Image]
    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor
    doc_lengths: list[int]

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.numel())

    @property
    def doc_count(self) -> int:
        return len(self.doc_lengths)


def build_attention_mask_mapping(
    doc_lengths: list[int],
    padded_seq_len: int,
    sliding_window: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    if padded_seq_len <= 0:
        raise ValueError("padded_seq_len must be positive.")

    doc_ids = torch.full((padded_seq_len,), -1, dtype=torch.long)
    cursor = 0
    for doc_index, doc_len in enumerate(doc_lengths):
        doc_ids[cursor : cursor + doc_len] = doc_index
        cursor += doc_len

    real_token_mask = doc_ids >= 0
    query_positions = torch.arange(padded_seq_len, dtype=torch.long).unsqueeze(1)
    key_positions = torch.arange(padded_seq_len, dtype=torch.long).unsqueeze(0)

    same_doc = (
        (doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))
        & real_token_mask.unsqueeze(1)
        & real_token_mask.unsqueeze(0)
    )
    causal = key_positions <= query_positions
    full_allowed = same_doc & causal
    sliding_allowed = full_allowed & ((query_positions - key_positions) < sliding_window)

    full_allowed.fill_diagonal_(True)
    sliding_allowed.fill_diagonal_(True)

    mask_value = torch.finfo(dtype).min
    full_mask = torch.full((padded_seq_len, padded_seq_len), mask_value, dtype=dtype)
    sliding_mask = torch.full((padded_seq_len, padded_seq_len), mask_value, dtype=dtype)
    full_mask[full_allowed] = 0
    sliding_mask[sliding_allowed] = 0

    return {
        "full_attention": full_mask.unsqueeze(0),
        "sliding_attention": sliding_mask.unsqueeze(0),
    }


@dataclass
class PackedCaptionCollator:
    tokenizer: Any
    prompt_text: str
    user_message_template: str
    max_prompt_tokens: int
    max_caption_tokens: int
    max_chat_input_tokens: int
    max_chat_output_tokens: int
    max_seq_len: int
    sliding_window: int
    vision_start_token_id: int
    image_token_id: int
    vision_end_token_id: int
    image_seq_len: int
    target_examples: int
    sort_by_length: bool = True
    _carryover_examples: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.target_examples <= 0:
            raise ValueError("target_examples must be positive.")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.caption_messages = [
            {
                "role": "user",
                "content": _build_multimodal_user_content(
                    self.tokenizer,
                    prompt_text=self.prompt_text,
                    user_message_template=self.user_message_template,
                    max_prompt_tokens=self.max_prompt_tokens,
                ),
            }
        ]

    def _build_sample(self, example: dict[str, Any]) -> TokenizedCaptionSample | None:
        example_type = example["example_type"]
        enforce_prompt_budget = False
        allow_history_dropping = False
        if example_type == "caption":
            messages = self.caption_messages
            image = _try_to_pil_image(example["image"])
            if image is None:
                return None
            images = [image]
            assistant_text_budget = self.max_caption_tokens
            assistant_source_text = example["caption"]
        elif example_type in {"cap_qa", "point_explanation"}:
            messages = _build_single_turn_multimodal_messages(
                self.tokenizer,
                prompt_text=example["prompt_text"],
                user_message_template=self.user_message_template,
                max_prompt_tokens=max(1, self.max_chat_input_tokens),
            )
            image = _try_to_pil_image(example["image"])
            if image is None:
                return None
            images = [image]
            assistant_text_budget = self.max_chat_output_tokens
            assistant_source_text = example["assistant_output"]
            enforce_prompt_budget = True
        elif example_type in {"chat", "chat_with_irrelevant_image"}:
            attach_irrelevant_image = example_type == "chat_with_irrelevant_image"
            messages = _build_chat_messages(
                input_messages=copy.deepcopy(example["input_messages"]),
                system_prompt=example.get("system_prompt"),
                attach_irrelevant_image=attach_irrelevant_image,
            )
            images = []
            if attach_irrelevant_image:
                irrelevant_image = _try_to_pil_image(example["irrelevant_image"])
                if irrelevant_image is None:
                    return None
                images.append(irrelevant_image)

            assistant_text_budget = self.max_chat_output_tokens
            assistant_source_text = example["assistant_output"]
            enforce_prompt_budget = True
            allow_history_dropping = True
        else:
            raise ValueError(f"Unsupported example_type: {example_type}")

        if enforce_prompt_budget and self.max_chat_input_tokens > 0:
            if allow_history_dropping:
                messages = _fit_chat_messages_to_prompt_budget(
                    self.tokenizer,
                    messages=messages,
                    max_prompt_tokens=max(1, self.max_chat_input_tokens),
                )
                if messages is None:
                    return None
            else:
                if _count_prompt_text_tokens(self.tokenizer, messages) > self.max_chat_input_tokens:
                    return None

        assistant_ids = _tokenize_text_without_truncation(
            self.tokenizer,
            assistant_source_text,
        )
        if assistant_text_budget > 0 and len(assistant_ids) > assistant_text_budget:
            return None

        assistant_text = assistant_source_text.strip()
        if allow_history_dropping:
            fitted = _fit_chat_example_to_seq_len(
                self.tokenizer,
                messages=messages,
                assistant_content=assistant_text,
                image_seq_len=self.image_seq_len,
                max_seq_len=self.max_seq_len,
            )
            if fitted is None:
                return None
            _, rendered = fitted
        else:
            rendered = _render_chat_example(
                self.tokenizer,
                messages=messages,
                assistant_content=assistant_text,
                image_seq_len=self.image_seq_len,
            )
            if len(rendered["input_ids"]) > self.max_seq_len:
                return None

        input_ids = torch.tensor(rendered["input_ids"], dtype=torch.long)
        assistant_mask = torch.tensor(rendered["assistant_masks"], dtype=torch.bool)
        labels = torch.where(
            assistant_mask,
            input_ids,
            torch.full_like(input_ids, -100),
        )
        position_ids = torch.arange(input_ids.numel(), dtype=torch.long)
        return TokenizedCaptionSample(
            images=images,
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
        )

    def _materialize_pack(self, samples: list[TokenizedCaptionSample]) -> PackedCaptionSample:
        return PackedCaptionSample(
            images=[image for sample in samples for image in sample.images],
            input_ids=torch.cat([sample.input_ids for sample in samples], dim=0),
            labels=torch.cat([sample.labels for sample in samples], dim=0),
            position_ids=torch.cat([sample.position_ids for sample in samples], dim=0),
            doc_lengths=[sample.seq_len for sample in samples],
        )

    def _pack_samples(self, samples: list[TokenizedCaptionSample]) -> list[PackedCaptionSample]:
        ordered_samples = list(samples)
        if self.sort_by_length:
            ordered_samples.sort(key=lambda sample: sample.seq_len, reverse=True)

        bins: list[list[TokenizedCaptionSample]] = []
        bin_lengths: list[int] = []
        for sample in ordered_samples:
            placed = False
            for bin_index, current_length in enumerate(bin_lengths):
                if current_length + sample.seq_len <= self.max_seq_len:
                    bins[bin_index].append(sample)
                    bin_lengths[bin_index] += sample.seq_len
                    placed = True
                    break
            if not placed:
                bins.append([sample])
                bin_lengths.append(sample.seq_len)

        return [self._materialize_pack(samples_in_bin) for samples_in_bin in bins]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        candidates = [*self._carryover_examples, *batch]
        self._carryover_examples = []

        tokenized_samples: list[TokenizedCaptionSample] = []
        candidate_index = 0
        for candidate_index, example in enumerate(candidates):
            sample = self._build_sample(example)
            if sample is None:
                continue
            tokenized_samples.append(sample)
            if len(tokenized_samples) >= self.target_examples:
                self._carryover_examples = candidates[candidate_index + 1 :]
                break

        if not tokenized_samples:
            return None
        packed_samples = self._pack_samples(tokenized_samples)
        max_seq_len = max(sample.seq_len for sample in packed_samples)

        input_ids = torch.full(
            (len(packed_samples), max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (len(packed_samples), max_seq_len),
            -100,
            dtype=torch.long,
        )
        position_ids = torch.zeros(
            (len(packed_samples), max_seq_len),
            dtype=torch.long,
        )
        full_masks: list[torch.Tensor] = []
        sliding_masks: list[torch.Tensor] = []
        sequence_lengths: list[int] = []
        example_counts: list[int] = []
        images: list[list[Image.Image]] = []

        for batch_index, packed_sample in enumerate(packed_samples):
            seq_len = packed_sample.seq_len
            input_ids[batch_index, :seq_len] = packed_sample.input_ids
            labels[batch_index, :seq_len] = packed_sample.labels
            position_ids[batch_index, :seq_len] = packed_sample.position_ids

            attention_masks = build_attention_mask_mapping(
                packed_sample.doc_lengths,
                max_seq_len,
                self.sliding_window,
            )
            full_masks.append(attention_masks["full_attention"])
            sliding_masks.append(attention_masks["sliding_attention"])
            sequence_lengths.append(seq_len)
            example_counts.append(packed_sample.doc_count)
            images.append(packed_sample.images)

        return {
            "images": images,
            "input_ids": input_ids,
            "attention_mask": {
                "full_attention": torch.stack(full_masks, dim=0),
                "sliding_attention": torch.stack(sliding_masks, dim=0),
            },
            "position_ids": position_ids,
            "labels": labels,
            "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
            "example_counts": torch.tensor(example_counts, dtype=torch.long),
        }
