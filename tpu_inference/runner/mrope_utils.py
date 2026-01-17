from __future__ import annotations

from typing import Iterable, Sequence

import jax.numpy as jnp
import numpy as np


def _normalize_grid_thw(grid_thw: object) -> list[tuple[int, int, int]]:
    if grid_thw is None:
        return []

    if isinstance(grid_thw, (list, tuple)):
        if len(grid_thw) == 0:
            return []
        if len(grid_thw) == 3 and all(
                isinstance(v, (int, np.integer)) for v in grid_thw):
            return [tuple(int(v) for v in grid_thw)]
        if all(isinstance(row, (list, tuple, np.ndarray)) for row in grid_thw):
            if grid_thw and isinstance(grid_thw[0], (list, tuple, np.ndarray)):
                if grid_thw[0] and isinstance(grid_thw[0][0],
                                              (list, tuple, np.ndarray)):
                    flat_rows = [row for batch in grid_thw for row in batch]
                    return [tuple(int(v) for v in row) for row in flat_rows]
            return [tuple(int(v) for v in row) for row in grid_thw]

    arr = np.asarray(grid_thw)
    if arr.size == 0:
        return []
    if arr.ndim == 1 and arr.shape[0] == 3:
        return [tuple(int(v) for v in arr.tolist())]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return [tuple(int(v) for v in row) for row in arr.tolist()]
    if arr.ndim == 3 and arr.shape[2] == 3:
        flat = arr.reshape(-1, 3)
        return [tuple(int(v) for v in row) for row in flat.tolist()]

    raise ValueError(
        "Incorrect type/shape of grid_thw. Expected (3,), (N, 3), or (B, N, 3)."
    )


def get_qwen3vl_mrope_input_positions_raw(
    input_tokens: Sequence[int],
    image_grid_thw: list[tuple[int, int, int]] | None,
    video_grid_thw: list[tuple[int, int, int]] | None,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
):
    """Compute M-RoPE 3D position IDs for a single sequence (Qwen3-VL)."""
    del vision_start_token_id

    image_grid_thw = image_grid_thw or []
    video_grid_thw = video_grid_thw or []

    image_nums = len(image_grid_thw)
    video_nums = len(video_grid_thw)
    llm_pos_ids_list = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums
    image_index, video_index = 0, 0

    input_tokens = list(input_tokens)

    for _ in range(image_nums + video_nums):
        if remain_images > 0:
            try:
                ed_image = input_tokens.index(image_token_id, st)
            except ValueError:
                ed_image = len(input_tokens) + 1
        else:
            ed_image = len(input_tokens) + 1

        if remain_videos > 0:
            try:
                ed_video = input_tokens.index(video_token_id, st)
            except ValueError:
                ed_video = len(input_tokens) + 1
        else:
            ed_video = len(input_tokens) + 1

        if ed_image < ed_video:
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            remain_images -= 1
            ed = ed_image
        else:
            t, h, w = video_grid_thw[video_index]
            video_index += 1
            remain_videos -= 1
            ed = ed_video

        llm_grid_t = t
        llm_grid_h = h // spatial_merge_size
        llm_grid_w = w // spatial_merge_size
        text_len = ed - st

        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            ) + st_idx)

        t_index = jnp.broadcast_to(
            jnp.arange(llm_grid_t, dtype=jnp.int32).reshape(-1, 1),
            (llm_grid_t, llm_grid_h * llm_grid_w),
        ).flatten()

        h_index = jnp.broadcast_to(
            jnp.arange(llm_grid_h, dtype=jnp.int32).reshape(1, -1, 1),
            (llm_grid_t, llm_grid_h, llm_grid_w),
        ).flatten()

        w_index = jnp.broadcast_to(
            jnp.arange(llm_grid_w, dtype=jnp.int32).reshape(1, 1, -1),
            (llm_grid_t, llm_grid_h, llm_grid_w),
        ).flatten()

        llm_pos_ids_list.append(
            jnp.stack([t_index, h_index, w_index]) + text_len + st_idx)

        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    if st < len(input_tokens):
        st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
        text_len = len(input_tokens) - st
        llm_pos_ids_list.append(
            jnp.broadcast_to(
                jnp.arange(text_len, dtype=jnp.int32).reshape(1, -1),
                (3, text_len),
            ) + st_idx)

    if not llm_pos_ids_list:
        llm_positions = jnp.zeros((3, 0), dtype=jnp.int32)
        return llm_positions, 0

    llm_positions = jnp.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
    mrope_position_delta = int(llm_positions.max()) + 1 - len(input_tokens)

    return llm_positions, mrope_position_delta


def _expand_video_grid_thw_for_qwen3vl_moe(
    video_grid_thw: list[tuple[int, int, int]] | None,
) -> list[tuple[int, int, int]] | None:
    if not video_grid_thw:
        return video_grid_thw

    expanded_video = []
    for t, h, w in video_grid_thw:
        t_val = int(t)
        expanded_video.extend([(1, int(h), int(w))] * t_val)
    return expanded_video


def get_qwen3vl_mrope_input_positions(
    input_tokens: Sequence[int],
    hf_config=None,
    image_grid_thw=None,
    video_grid_thw=None,
    second_per_grid_ts=None,
    context_len: int = 0,
    seq_len: int | None = None,
    audio_feature_lengths=None,
    use_audio_in_video: bool = False,
):
    """Wrapper for Qwen3-VL M-RoPE, matching runner callback signature."""
    del second_per_grid_ts, audio_feature_lengths, use_audio_in_video

    if hf_config is None:
        raise ValueError("hf_config is required for Qwen3-VL M-RoPE.")

    image_grid_thw = _normalize_grid_thw(image_grid_thw)
    video_grid_thw = _normalize_grid_thw(video_grid_thw)

    llm_positions, mrope_position_delta = get_qwen3vl_mrope_input_positions_raw(
        input_tokens=input_tokens,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
    )

    llm_positions = llm_positions[:, context_len:seq_len]
    return llm_positions, mrope_position_delta


def get_qwen3vl_moe_mrope_input_positions(
    input_tokens: Sequence[int],
    hf_config=None,
    image_grid_thw=None,
    video_grid_thw=None,
    second_per_grid_ts=None,
    context_len: int = 0,
    seq_len: int | None = None,
    audio_feature_lengths=None,
    use_audio_in_video: bool = False,
):
    """Wrapper for Qwen3-VL-MoE M-RoPE, matching runner callback signature."""
    del second_per_grid_ts, audio_feature_lengths, use_audio_in_video

    if hf_config is None:
        raise ValueError("hf_config is required for Qwen3-VL-MoE M-RoPE.")

    image_grid_thw = _normalize_grid_thw(image_grid_thw)
    video_grid_thw = _normalize_grid_thw(video_grid_thw)
    video_grid_thw = _expand_video_grid_thw_for_qwen3vl_moe(video_grid_thw)

    llm_positions, mrope_position_delta = get_qwen3vl_mrope_input_positions_raw(
        input_tokens=input_tokens,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
    )

    llm_positions = llm_positions[:, context_len:seq_len]
    return llm_positions, mrope_position_delta


def is_qwen3vl_moe_config(hf_config) -> bool:
    if hf_config is None:
        return False

    model_type = getattr(hf_config, "model_type", "")
    if isinstance(model_type, str):
        model_type_lower = model_type.lower()
        if "qwen3_vl_moe" in model_type_lower:
            return True

    architectures = getattr(hf_config, "architectures", None) or []
    for arch in architectures:
        if "Qwen3VLMoe" in arch or "qwen3_vl_moe" in arch.lower():
            return True

    name_or_path = getattr(hf_config, "_name_or_path", "")
    if isinstance(name_or_path, str):
        name_lower = name_or_path.lower()
        if "qwen3-vl" in name_lower and "moe" in name_lower:
            return True

    return False


def is_qwen3vl_config(hf_config) -> bool:
    if hf_config is None:
        return False

    if is_qwen3vl_moe_config(hf_config):
        return False

    model_type = getattr(hf_config, "model_type", "")
    if isinstance(model_type, str):
        model_type_lower = model_type.lower()
        if "qwen3_vl" in model_type_lower:
            return True

    architectures = getattr(hf_config, "architectures", None) or []
    for arch in architectures:
        if "Qwen3VL" in arch or "qwen3_vl" in arch.lower():
            return True

    name_or_path = getattr(hf_config, "_name_or_path", "")
    if isinstance(name_or_path, str) and "qwen3-vl" in name_or_path.lower():
        return True

    return False
