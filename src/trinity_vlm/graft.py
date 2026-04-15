from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from safetensors.torch import load_file, save_file

import torch

from .configuration_trinity_vlm import TrinityVLMConfig
from .modeling_trinity_vlm import TrinityVLMForConditionalGeneration


def _normalize_delta_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        normalized_name = name.removeprefix("_orig_mod.")
        if normalized_name.startswith("bridge."):
            normalized_name = "multi_modal_projector." + normalized_name[len("bridge.") :]
        normalized_state_dict[normalized_name] = tensor
    return normalized_state_dict


@dataclass
class TrinityMoondreamGraftConfig:
    trinity_model_name: str = "arcee-ai/Trinity-Nano-Preview"
    moondream_model_name: str = "moondream/moondream3-preview"
    projector_hidden_dim: int = 2048
    router_aux_loss_coef: float = 0.001
    use_cut_cross_entropy: bool = False
    cut_cross_entropy_impl: str = "cce"
    cut_cross_entropy_filter_eps: float | str | None = "auto"
    cut_cross_entropy_accum_e_fp32: bool = False
    cut_cross_entropy_accum_c_fp32: bool = False
    cut_cross_entropy_filter_e_grad: bool = True
    cut_cross_entropy_filter_c_grad: bool = True
    freeze_trinity: bool = True
    freeze_vision: bool = True
    unfreeze_last_n_layers: int = 0
    train_lm_head: bool = False
    trust_remote_code: bool = True


class TrinityMoondreamGraftModel(TrinityVLMForConditionalGeneration):
    def __init__(
        self,
        config: TrinityMoondreamGraftConfig | None = None,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> None:
        self.graft_config = config or TrinityMoondreamGraftConfig()
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self.revision = revision

        model_config = TrinityVLMConfig(
            trinity_model_name_or_path=self.graft_config.trinity_model_name,
            moondream_model_name_or_path=self.graft_config.moondream_model_name,
            projector_hidden_dim=self.graft_config.projector_hidden_dim,
            router_aux_loss_coef=self.graft_config.router_aux_loss_coef,
            use_cut_cross_entropy=self.graft_config.use_cut_cross_entropy,
            cut_cross_entropy_impl=self.graft_config.cut_cross_entropy_impl,
            cut_cross_entropy_filter_eps=self.graft_config.cut_cross_entropy_filter_eps,
            cut_cross_entropy_accum_e_fp32=self.graft_config.cut_cross_entropy_accum_e_fp32,
            cut_cross_entropy_accum_c_fp32=self.graft_config.cut_cross_entropy_accum_c_fp32,
            cut_cross_entropy_filter_e_grad=self.graft_config.cut_cross_entropy_filter_e_grad,
            cut_cross_entropy_filter_c_grad=self.graft_config.cut_cross_entropy_filter_c_grad,
            trust_remote_code=self.graft_config.trust_remote_code,
            trinity_revision=revision,
            moondream_revision=revision,
        )
        super().__init__(
            model_config,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

        self.configure_trainable_parameters()

    @property
    def device(self) -> torch.device:
        return self.language_model.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.language_model.get_input_embeddings().weight.dtype

    def configure_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

        for parameter in self.bridge.parameters():
            parameter.requires_grad = True
        for parameter in self.special_token_embeddings.parameters():
            parameter.requires_grad = True

        if not self.graft_config.freeze_vision:
            for parameter in self.vision_tower.parameters():
                parameter.requires_grad = True

        if not self.graft_config.freeze_trinity:
            for parameter in self.language_model.parameters():
                parameter.requires_grad = True
            return

        if self.graft_config.unfreeze_last_n_layers > 0:
            decoder_layers = self.language_model.model.layers[-self.graft_config.unfreeze_last_n_layers :]
            for layer in decoder_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.language_model.model.norm.parameters():
                parameter.requires_grad = True

        if self.graft_config.train_lm_head:
            for parameter in self.language_model.lm_head.parameters():
                parameter.requires_grad = True

    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return trainable, total

    def delta_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    def save_delta(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_file(self.delta_state_dict(), output_path / "trainable.safetensors")
        metadata = {
            "config": asdict(self.graft_config),
            "trainable_parameter_names": [
                name for name, parameter in self.named_parameters() if parameter.requires_grad
            ],
        }
        (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def load_delta(self, output_dir: str | Path) -> None:
        state_dict = _normalize_delta_state_dict_keys(
            load_file(Path(output_dir) / "trainable.safetensors")
        )
        _, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys in delta checkpoint: {unexpected}")
