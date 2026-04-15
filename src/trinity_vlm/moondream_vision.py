from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open
from torch import nn


@dataclass(frozen=True)
class MoondreamVisionConfig:
    enc_dim: int = 1152
    enc_patch_size: int = 14
    enc_n_layers: int = 27
    enc_ff_dim: int = 4304
    enc_n_heads: int = 16
    proj_out_dim: int = 2048
    crop_size: int = 378
    in_channels: int = 3
    max_crops: int = 12
    overlap_margin: int = 4
    proj_inner_dim: int = 8192

    @property
    def image_seq_len(self) -> int:
        return (self.crop_size // self.enc_patch_size) ** 2


def select_tiling(height: int, width: int, crop_size: int, max_crops: int) -> tuple[int, int]:
    if height <= crop_size or width <= crop_size:
        return (1, 1)

    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)

    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))

    h_tiles = math.floor(math.sqrt(max_crops * height / width))
    w_tiles = math.floor(math.sqrt(max_crops * width / height))

    h_tiles = max(h_tiles, min_h)
    w_tiles = max(w_tiles, min_w)

    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)

    return (max(1, h_tiles), max(1, w_tiles))


def overlap_crop_image(
    image: np.ndarray,
    overlap_margin: int,
    max_crops: int,
    base_size: tuple[int, int] = (378, 378),
    patch_size: int = 14,
) -> tuple[np.ndarray, tuple[int, int]]:
    original_h, original_w = image.shape[:2]
    margin_pixels = patch_size * overlap_margin
    total_margin_pixels = margin_pixels * 2
    crop_patches = base_size[0] // patch_size
    crop_window_patches = crop_patches - (2 * overlap_margin)
    crop_window_size = crop_window_patches * patch_size

    tiling = select_tiling(
        original_h - total_margin_pixels,
        original_w - total_margin_pixels,
        crop_window_size,
        max_crops,
    )

    n_crops = tiling[0] * tiling[1] + 1
    crops = np.zeros((n_crops, base_size[0], base_size[1], image.shape[2]), dtype=np.uint8)

    target_size = (
        tiling[0] * crop_window_size + total_margin_pixels,
        tiling[1] * crop_window_size + total_margin_pixels,
    )
    pil_img = Image.fromarray(image)
    resized = pil_img.resize(
        (int(target_size[1]), int(target_size[0])),
        resample=Image.Resampling.LANCZOS,
    )
    image = np.asarray(resized)

    global_pil = pil_img.resize(
        (int(base_size[1]), int(base_size[0])),
        resample=Image.Resampling.LANCZOS,
    )
    crops[0] = np.asarray(global_pil)

    for i in range(tiling[0]):
        for j in range(tiling[1]):
            y0 = i * crop_window_size
            x0 = j * crop_window_size
            y_end = min(y0 + base_size[0], image.shape[0])
            x_end = min(x0 + base_size[1], image.shape[1])
            crop_region = image[y0:y_end, x0:x_end]
            crops[1 + i * tiling[1] + j, : crop_region.shape[0], : crop_region.shape[1]] = crop_region

    return crops, tiling


@torch.compiler.disable
def reconstruct_from_crops(
    crops: torch.Tensor,
    tiling: tuple[int, int],
    overlap_margin: int,
    patch_size: int = 14,
) -> torch.Tensor:
    tiling_h, tiling_w = tiling
    crop_height, crop_width = crops[0].shape[:2]
    margin_pixels = overlap_margin * patch_size
    output_h = (crop_height - 2 * margin_pixels) * tiling_h + 2 * margin_pixels
    output_w = (crop_width - 2 * margin_pixels) * tiling_w + 2 * margin_pixels

    reconstructed = torch.zeros(
        (output_h, output_w, crops[0].shape[2]),
        device=crops[0].device,
        dtype=crops[0].dtype,
    )

    for i, crop in enumerate(crops):
        tile_y = i // tiling_w
        tile_x = i % tiling_w
        x_start = 0 if tile_x == 0 else margin_pixels
        x_end = crop_width if tile_x == tiling_w - 1 else crop_width - margin_pixels
        y_start = 0 if tile_y == 0 else margin_pixels
        y_end = crop_height if tile_y == tiling_h - 1 else crop_height - margin_pixels
        out_x = tile_x * (crop_width - 2 * margin_pixels)
        out_y = tile_y * (crop_height - 2 * margin_pixels)
        reconstructed[
            out_y + y_start : out_y + y_end,
            out_x + x_start : out_x + x_end,
        ] = crop[y_start:y_end, x_start:x_end]

    return reconstructed


@torch.compiler.disable
def prepare_crops(
    image: Image.Image,
    config: MoondreamVisionConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, int]]:
    np_image = np.array(image.convert("RGB"))
    crops, tiling = overlap_crop_image(
        np_image,
        max_crops=config.max_crops,
        overlap_margin=config.overlap_margin,
        base_size=(config.crop_size, config.crop_size),
        patch_size=config.enc_patch_size,
    )
    crops = np.transpose(crops, (0, 3, 1, 2))
    crops_tensor = torch.from_numpy(crops).to(device=device, dtype=dtype)
    crops_tensor = crops_tensor.div_(255.0).sub_(0.5).div_(0.5)
    return crops_tensor, tiling


def create_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    x = x.reshape(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(batch, (height // patch_size) * (width // patch_size), channels * patch_size * patch_size)
    return x


class MoondreamAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim, dtype=dtype)
        self.proj = nn.Linear(dim, dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        head_dim = dim // self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        return self.proj(out)


class MoondreamMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x), approximate="tanh")
        return self.fc2(x)


class MoondreamVisionBlock(nn.Module):
    def __init__(self, config: MoondreamVisionConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.enc_dim, dtype=dtype)
        self.attn = MoondreamAttention(config.enc_dim, config.enc_n_heads, dtype)
        self.ln2 = nn.LayerNorm(config.enc_dim, dtype=dtype)
        self.mlp = MoondreamMLP(config.enc_dim, config.enc_ff_dim, config.enc_dim, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MoondreamVisionTower(nn.Module):
    def __init__(
        self,
        config: MoondreamVisionConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config or MoondreamVisionConfig()
        self.patch_emb = nn.Linear(
            self.config.enc_patch_size * self.config.enc_patch_size * self.config.in_channels,
            self.config.enc_dim,
            dtype=dtype,
        )
        self.blocks = nn.ModuleList(
            [MoondreamVisionBlock(self.config, dtype=dtype) for _ in range(self.config.enc_n_layers)]
        )
        self.post_ln = nn.LayerNorm(self.config.enc_dim, dtype=dtype)
        self.proj_mlp = MoondreamMLP(
            self.config.enc_dim * 2,
            self.config.proj_inner_dim,
            self.config.proj_out_dim,
            dtype,
        )
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.config.image_seq_len, self.config.enc_dim, dtype=dtype)
        )

    @property
    def image_seq_len(self) -> int:
        return self.config.image_seq_len

    def encode_crops(self, inputs_bchw: torch.Tensor) -> torch.Tensor:
        x = create_patches(inputs_bchw, self.config.enc_patch_size)
        x = self.patch_emb(x)
        x = x + self.pos_emb
        for block in self.blocks:
            x = block(x)
        return self.post_ln(x)

    def project_features(self, global_features: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        reconstructed = reconstructed.permute(2, 0, 1)
        reconstructed = F.adaptive_avg_pool2d(
            reconstructed,
            output_size=(self.config.enc_n_layers, self.config.enc_n_layers),
        )
        reconstructed = reconstructed.permute(1, 2, 0).reshape(self.image_seq_len, self.config.enc_dim)
        return self.proj_mlp(torch.cat([global_features, reconstructed], dim=-1))

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image, got {type(image)!r}")

        device = self.pos_emb.device
        dtype = self.pos_emb.dtype
        crops, tiling = prepare_crops(image, self.config, device=device, dtype=dtype)
        outputs = self.encode_crops(crops)
        global_features = outputs[0]
        local_features = outputs[1:].view(
            -1,
            self.config.enc_n_layers,
            self.config.enc_n_layers,
            self.config.enc_dim,
        )
        reconstructed = reconstruct_from_crops(
            local_features,
            tiling,
            patch_size=1,
            overlap_margin=self.config.overlap_margin,
        )
        return self.project_features(global_features, reconstructed)

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        encoded = [self.encode_image(image) for image in images]
        return torch.stack(encoded, dim=0)

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "MoondreamVisionTower":
        snapshot_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=[
                    "model.safetensors.index.json",
                    "modelv2-00001-of-00004.safetensors",
                ],
                local_files_only=local_files_only,
            )
        )
        model = cls(dtype=dtype)
        state_dict = {}
        with safe_open(
            snapshot_dir / "modelv2-00001-of-00004.safetensors",
            framework="pt",
            device="cpu",
        ) as handle:
            for key in handle.keys():
                if key.startswith("model.vision."):
                    state_dict[key.removeprefix("model.vision.")] = handle.get_tensor(key)

        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise ValueError(f"Vision load mismatch. Missing={missing}, unexpected={unexpected}")
        return model
