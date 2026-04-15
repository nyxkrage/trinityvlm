from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast


TRINITY_TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

TRINITY_CHAT_TEMPLATE_PATH = Path(__file__).with_name("chat_template.jinja")
TRINITY_MISTRAL_TEXT_SPLIT_INDEX = 5
MISTRAL_FIXED_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def snapshot_repo(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    local_files_only: bool = False,
) -> Path:
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_files_only=local_files_only,
        )
    )


def read_trinity_chat_template() -> str:
    return TRINITY_CHAT_TEMPLATE_PATH.read_text()


def install_trinity_chat_template(tokenizer):
    tokenizer.chat_template = read_trinity_chat_template()
    return tokenizer


def patch_trinity_mistral_regex(tokenizer):
    import tokenizers

    current_pretokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    split_pretokenizer = tokenizers.pre_tokenizers.Split(
        pattern=tokenizers.Regex(MISTRAL_FIXED_REGEX),
        behavior="isolated",
    )
    if not isinstance(current_pretokenizer, tokenizers.pre_tokenizers.Sequence):
        raise TypeError(
            "Expected Trinity tokenizer pre_tokenizer to be a Sequence; "
            f"got {type(current_pretokenizer)!r}."
        )
    try:
        current_split = current_pretokenizer[TRINITY_MISTRAL_TEXT_SPLIT_INDEX]
    except IndexError as exc:
        raise RuntimeError(
            "Trinity tokenizer pretokenizer layout is shorter than expected."
        ) from exc
    current_split_repr = repr(current_split)
    if MISTRAL_FIXED_REGEX in current_split_repr:
        tokenizer.fix_mistral_regex = True
        return tokenizer
    if "[A-Za-z]+" not in current_split_repr or r"\p{L}\p{P}\p{S}" not in current_split_repr:
        raise RuntimeError(
            "Trinity tokenizer pretokenizer layout changed; expected the incorrect "
            "Mistral text split at index 5."
        )
    current_pretokenizer[TRINITY_MISTRAL_TEXT_SPLIT_INDEX] = split_pretokenizer
    tokenizer.fix_mistral_regex = True
    return tokenizer


def save_trinity_chat_template(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chat_template_path = output_path / "chat_template.jinja"
    chat_template_path.write_text(read_trinity_chat_template())
    return chat_template_path


def load_trinity_tokenizer(
    repo_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
):
    candidate_path = Path(repo_id).expanduser()
    if candidate_path.exists():
        snapshot_dir = candidate_path.resolve()
    else:
        snapshot_dir = snapshot_repo(
            repo_id,
            revision=revision,
            allow_patterns=TRINITY_TOKENIZER_PATTERNS,
            local_files_only=local_files_only,
        )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        str(snapshot_dir),
        local_files_only=True,
        fix_mistral_regex=False,
    )
    tokenizer = patch_trinity_mistral_regex(tokenizer)
    install_trinity_chat_template(tokenizer)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
