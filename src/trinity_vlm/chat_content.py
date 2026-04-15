from __future__ import annotations

from typing import Any


_IMAGE_SENTINEL = "\u0000TRINITY_VLM_IMAGE\u0000"
_PROMPT_SENTINEL = "\u0000TRINITY_VLM_PROMPT\u0000"


def truncate_text_to_token_budget(
    tokenizer,
    text: str,
    *,
    max_tokens: int,
) -> str:
    token_ids = tokenizer(
        text.strip(),
        add_special_tokens=False,
        truncation=True,
        max_length=max(1, max_tokens),
    )["input_ids"]
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()


def _append_text_part(parts: list[dict[str, str]], text: str) -> None:
    if not text:
        return
    if parts and parts[-1].get("type") == "text":
        parts[-1]["text"] += text
        return
    parts.append({"type": "text", "text": text})


def build_user_message_content(
    tokenizer,
    *,
    prompt_text: str,
    user_message_template: str,
    max_prompt_tokens: int,
    include_image: bool = True,
    truncate_prompt: bool = False,
) -> list[dict[str, Any]]:
    if "{prompt}" not in user_message_template or "{image}" not in user_message_template:
        raise ValueError(
            "user_message_template must contain both {prompt} and {image} placeholders."
        )

    if truncate_prompt:
        prompt_text = truncate_text_to_token_budget(
            tokenizer,
            prompt_text,
            max_tokens=max_prompt_tokens,
        )
    else:
        prompt_text = prompt_text.strip()
    rendered = user_message_template.format(
        prompt=_PROMPT_SENTINEL,
        image=_IMAGE_SENTINEL if include_image else "",
    )
    rendered_lines = [line.rstrip() for line in rendered.splitlines()]
    rendered = "\n".join(rendered_lines).strip()

    parts: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(rendered):
        next_image = rendered.find(_IMAGE_SENTINEL, cursor)
        next_prompt = rendered.find(_PROMPT_SENTINEL, cursor)
        matches = [
            (next_image, _IMAGE_SENTINEL),
            (next_prompt, _PROMPT_SENTINEL),
        ]
        matches = [match for match in matches if match[0] >= 0]
        if not matches:
            _append_text_part(parts, rendered[cursor:])
            break

        next_index, sentinel = min(matches, key=lambda match: match[0])
        _append_text_part(parts, rendered[cursor:next_index])
        if sentinel == _IMAGE_SENTINEL:
            parts.append({"type": "image"})
        elif prompt_text:
            _append_text_part(parts, prompt_text)
        cursor = next_index + len(sentinel)

    return parts
