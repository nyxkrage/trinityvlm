from __future__ import annotations


def build_image_token_span(
    *,
    vision_start_token_id: int | None,
    image_token_id: int | None,
    vision_end_token_id: int | None,
    image_seq_len: int,
    bos_token_id: int | None = None,
) -> list[int]:
    if vision_start_token_id is None:
        raise ValueError("vision_start_token_id is not configured.")
    if image_token_id is None:
        raise ValueError("image_token_id is not configured.")
    if vision_end_token_id is None:
        raise ValueError("vision_end_token_id is not configured.")

    token_ids: list[int] = []
    if bos_token_id is not None:
        token_ids.append(bos_token_id)
    token_ids.append(vision_start_token_id)
    token_ids.extend([image_token_id] * image_seq_len)
    token_ids.append(vision_end_token_id)
    return token_ids
