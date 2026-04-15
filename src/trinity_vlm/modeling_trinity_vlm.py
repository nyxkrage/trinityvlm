from __future__ import annotations

import importlib
import shutil
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import modeling_utils as hf_modeling_utils
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

try:
    from cut_cross_entropy import linear_cross_entropy
except ImportError:
    linear_cross_entropy = None

from .configuration_trinity_vlm import TrinityVLMConfig
from .image_tokens import build_image_token_span
from .moondream_vision import MoondreamVisionTower
from .tokenizer_utils import load_trinity_tokenizer


def _compute_default_rope_parameters(
    config=None,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    del seq_len, layer_type
    if config is None:
        raise ValueError("config is required to compute default RoPE parameters.")

    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


if "default" not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _compat_compute_default_rope_parameters(
    self,
    config=None,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    return _compute_default_rope_parameters(
        config=config or self.config,
        device=device,
        seq_len=seq_len,
        layer_type=layer_type,
    )


_ORIGINAL_PRETRAINEDMODEL_INIT_WEIGHTS = hf_modeling_utils.PreTrainedModel._init_weights


def _patched_trinity_compat_init_weights(self, module):
    if (
        "RotaryEmbedding" in module.__class__.__name__
        and hasattr(module, "original_inv_freq")
        and getattr(module, "rope_type", None) == "default"
        and not hasattr(module, "compute_default_rope_parameters")
    ):
        buffer_value, _ = _compute_default_rope_parameters(module.config)
        with torch.no_grad():
            module.inv_freq.copy_(buffer_value)
            module.original_inv_freq.copy_(buffer_value)
        return
    return _ORIGINAL_PRETRAINEDMODEL_INIT_WEIGHTS(self, module)


if hf_modeling_utils.PreTrainedModel._init_weights is not _patched_trinity_compat_init_weights:
    hf_modeling_utils.PreTrainedModel._init_weights = _patched_trinity_compat_init_weights


def _trinity_router_forward(
    router: nn.Module,
    hidden_states: torch.Tensor,
    expert_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    router_logits = router.gate(hidden_states_flat)

    if router.score_func == "sigmoid":
        router_probs = torch.sigmoid(router_logits.to(torch.float32))
    else:
        router_probs = F.softmax(router_logits.to(torch.float32), dim=-1)

    if expert_bias is not None:
        _, selected_experts = torch.topk(
            router_probs + expert_bias,
            k=router.top_k,
            dim=1,
        )
        top_scores = router_probs.gather(dim=1, index=selected_experts)
    else:
        top_scores, selected_experts = torch.topk(
            router_probs,
            k=router.top_k,
            dim=1,
        )

    if router.score_func == "sigmoid" and router.route_norm:
        denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
        top_scores = top_scores / denominator

    top_scores = top_scores * router.route_scale
    return hidden_states_flat, router_logits, router_probs, top_scores, selected_experts


def _trinity_router_aux_loss(
    router_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    *,
    num_experts: int,
) -> torch.Tensor:
    selected_flat = selected_experts.reshape(-1)
    top_k = max(1, selected_experts.shape[-1])
    token_count = max(1, selected_experts.shape[0])
    tokens_per_expert = torch.bincount(
        selected_flat,
        minlength=num_experts,
    ).to(torch.float32) / float(token_count * top_k)
    router_prob_per_expert = router_probs.mean(dim=0)
    return num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)


def _trinity_can_use_grouped_mm(hidden_states: torch.Tensor) -> bool:
    return (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and hasattr(F, "grouped_mm")
    )


def _trinity_has_packed_experts(moe_layer: nn.Module) -> bool:
    return all(
        isinstance(getattr(moe_layer, parameter_name, None), nn.Parameter)
        for parameter_name in ("packed_gate_proj", "packed_up_proj", "packed_down_proj")
    )


def _trinity_pack_expert_projection(
    moe_layer: nn.Module,
    *,
    projection_name: str,
) -> tuple[torch.Tensor, bool]:
    experts = getattr(moe_layer, "experts", None)
    if experts is None:
        raise ValueError("Cannot pack experts because the layer no longer owns per-expert modules.")

    projection_weights = []
    requires_grad = False
    for expert in experts:
        weight = getattr(expert, projection_name).weight
        requires_grad = requires_grad or weight.requires_grad
        projection_weights.append(weight.detach().transpose(0, 1).contiguous())
    return torch.stack(projection_weights, dim=0).contiguous(), requires_grad


def _trinity_pack_moe_experts_(moe_layer: nn.Module) -> bool:
    if _trinity_has_packed_experts(moe_layer):
        return False

    if getattr(moe_layer, "experts", None) is None:
        return False

    gate_weight, gate_requires_grad = _trinity_pack_expert_projection(
        moe_layer,
        projection_name="gate_proj",
    )
    up_weight, up_requires_grad = _trinity_pack_expert_projection(
        moe_layer,
        projection_name="up_proj",
    )
    down_weight, down_requires_grad = _trinity_pack_expert_projection(
        moe_layer,
        projection_name="down_proj",
    )

    moe_layer.packed_gate_proj = nn.Parameter(
        gate_weight,
        requires_grad=gate_requires_grad,
    )
    moe_layer.packed_up_proj = nn.Parameter(
        up_weight,
        requires_grad=up_requires_grad,
    )
    moe_layer.packed_down_proj = nn.Parameter(
        down_weight,
        requires_grad=down_requires_grad,
    )
    delattr(moe_layer, "experts")
    return True


def _trinity_pack_moe_layers(language_model: nn.Module) -> int:
    packed_layers = 0
    decoder_layers = getattr(getattr(language_model, "model", None), "layers", ())
    for decoder_layer in decoder_layers:
        if not getattr(decoder_layer, "moe_enabled", False):
            continue
        if _trinity_pack_moe_experts_(decoder_layer.mlp):
            packed_layers += 1
    return packed_layers


def _normalize_tied_weight_keys_for_save_(module: nn.Module) -> None:
    for submodule in module.modules():
        tied_weight_keys = getattr(submodule, "_tied_weights_keys", None)
        if isinstance(tied_weight_keys, list):
            submodule._tied_weights_keys = {key: None for key in tied_weight_keys}


def _trinity_get_grouped_projection_weights(
    moe_layer: nn.Module,
    expert_ids: torch.Tensor,
    *,
    projection_name: str,
) -> torch.Tensor:
    if expert_ids.numel() == 0:
        raise ValueError("Cannot select grouped weights for an empty expert set.")

    packed_weights = getattr(moe_layer, f"packed_{projection_name}", None)
    if isinstance(packed_weights, nn.Parameter):
        if expert_ids.numel() == packed_weights.shape[0]:
            full_expert_ids = torch.arange(
                packed_weights.shape[0],
                device=expert_ids.device,
                dtype=expert_ids.dtype,
            )
            if torch.equal(expert_ids, full_expert_ids):
                return packed_weights
        return packed_weights.index_select(0, expert_ids).contiguous()

    experts = getattr(moe_layer, "experts", None)
    if experts is None:
        raise ValueError(f"Layer has neither packed experts nor per-expert modules for {projection_name}.")

    projection_weights = []
    requires_grad = False
    for expert_id in expert_ids.tolist():
        weight = getattr(experts[expert_id], projection_name).weight
        requires_grad = requires_grad or weight.requires_grad
        projection_weights.append(weight.transpose(0, 1))

    if not requires_grad:
        projection_weights = [weight.detach() for weight in projection_weights]

    return torch.stack(projection_weights, dim=0).contiguous()


def _trinity_dense_packed_moe_forward(
    moe_layer: nn.Module,
    routed_input: torch.Tensor,
    token_to_expert: torch.Tensor,
    *,
    modeling_module,
) -> torch.Tensor:
    routed_output = torch.zeros(
        routed_input.shape[0],
        moe_layer.config.hidden_size,
        device=routed_input.device,
        dtype=routed_input.dtype,
    )

    packed_gate_proj = getattr(moe_layer, "packed_gate_proj", None)
    packed_up_proj = getattr(moe_layer, "packed_up_proj", None)
    packed_down_proj = getattr(moe_layer, "packed_down_proj", None)
    if not all(
        isinstance(weight, nn.Parameter)
        for weight in (packed_gate_proj, packed_up_proj, packed_down_proj)
    ):
        for expert_id in range(moe_layer.config.num_experts):
            mask = token_to_expert == expert_id
            if not mask.any():
                continue
            expert_input = routed_input[mask]
            expert_out = moe_layer.experts[expert_id](expert_input)
            routed_output[mask] = expert_out
        return routed_output

    act_fn = modeling_module.ACT2FN[moe_layer.config.hidden_act]
    for expert_id in range(moe_layer.config.num_experts):
        mask = token_to_expert == expert_id
        if not mask.any():
            continue
        expert_input = routed_input[mask]
        gate_proj = F.linear(expert_input, packed_gate_proj[expert_id].transpose(0, 1))
        up_proj = F.linear(expert_input, packed_up_proj[expert_id].transpose(0, 1))
        activated = act_fn(gate_proj) * up_proj
        expert_out = F.linear(activated, packed_down_proj[expert_id].transpose(0, 1))
        routed_output[mask] = expert_out
    return routed_output


def _trinity_accumulate_routed_output(
    shared_output: torch.Tensor,
    routed_output: torch.Tensor,
    top_scores_sorted: torch.Tensor,
    token_indices_sorted: torch.Tensor,
) -> torch.Tensor:
    output = shared_output.to(torch.float32)
    if routed_output.numel() == 0:
        return output

    hidden_dim = routed_output.shape[-1]
    bytes_per_row = max(1, hidden_dim * 4)
    target_chunk_bytes = 16 * 1024 * 1024
    rows_per_chunk = max(1, target_chunk_bytes // bytes_per_row)

    for start in range(0, routed_output.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, routed_output.shape[0])
        weighted_chunk = routed_output[start:end].to(torch.float32)
        weighted_chunk.mul_(top_scores_sorted[start:end].unsqueeze(-1))
        output.index_add_(0, token_indices_sorted[start:end], weighted_chunk)

    return output

def _patch_trinity_remote_modeling_module(modeling_module) -> None:
    if getattr(modeling_module, "_trinity_vlm_grouped_moe_patched", False):
        return

    def patched_afmoe_moe_forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat, _router_logits, router_probs, top_scores, selected_experts = _trinity_router_forward(
            self.router,
            hidden_states,
            self.expert_bias,
        )

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat)
        else:
            shared_output = torch.zeros_like(hidden_states_flat)

        token_indices_sorted = torch.argsort(selected_experts.view(-1), stable=True)
        top_scores_sorted = top_scores.view(-1)[token_indices_sorted]
        token_to_expert = selected_experts.view(-1)[token_indices_sorted]
        token_indices_sorted = token_indices_sorted // self.config.num_experts_per_tok

        token_indices_expanded = token_indices_sorted.unsqueeze(-1).expand(-1, hidden_dim)
        routed_input = torch.gather(
            hidden_states_flat,
            dim=0,
            index=token_indices_expanded,
        ).contiguous()

        routed_output: torch.Tensor | None = None
        use_grouped_mm = bool(getattr(self.config, "enable_grouped_moe", True)) and _trinity_can_use_grouped_mm(
            routed_input
        )
        if use_grouped_mm:
            expert_counts = torch.bincount(
                token_to_expert,
                minlength=self.config.num_experts,
            )
            grouped_offsets = torch.cumsum(
                expert_counts,
                dim=0,
                dtype=torch.int32,
            )
            packed_gate_proj = getattr(self, "packed_gate_proj", None)
            packed_up_proj = getattr(self, "packed_up_proj", None)
            packed_down_proj = getattr(self, "packed_down_proj", None)

            if all(
                isinstance(weight, nn.Parameter)
                for weight in (packed_gate_proj, packed_up_proj, packed_down_proj)
            ):
                gate_weights = packed_gate_proj
                up_weights = packed_up_proj
                down_weights = packed_down_proj
            else:
                active_expert_ids = torch.nonzero(expert_counts > 0, as_tuple=False).flatten()
                if active_expert_ids.numel() == 0:
                    routed_output = torch.zeros_like(routed_input)
                    gate_weights = up_weights = down_weights = None
                else:
                    grouped_offsets = torch.cumsum(
                        expert_counts.index_select(0, active_expert_ids),
                        dim=0,
                        dtype=torch.int32,
                    )
                    gate_weights = _trinity_get_grouped_projection_weights(
                        self,
                        active_expert_ids,
                        projection_name="gate_proj",
                    )
                    up_weights = _trinity_get_grouped_projection_weights(
                        self,
                        active_expert_ids,
                        projection_name="up_proj",
                    )
                    down_weights = _trinity_get_grouped_projection_weights(
                        self,
                        active_expert_ids,
                        projection_name="down_proj",
                    )

            if routed_output is None:
                gate_proj = F.grouped_mm(routed_input, gate_weights, offs=grouped_offsets)
                up_proj = F.grouped_mm(routed_input, up_weights, offs=grouped_offsets)
                activated = modeling_module.ACT2FN[self.config.hidden_act](gate_proj) * up_proj
                routed_output = F.grouped_mm(activated, down_weights, offs=grouped_offsets)
        else:
            routed_output = _trinity_dense_packed_moe_forward(
                self,
                routed_input,
                token_to_expert,
                modeling_module=modeling_module,
            )

        if routed_output is None:
            raise RuntimeError("MoE forward did not produce routed output.")

        if use_grouped_mm:
            del expert_counts, grouped_offsets
            if "active_expert_ids" in locals():
                del active_expert_ids
            if "gate_weights" in locals():
                del gate_weights, up_weights, down_weights
            if "gate_proj" in locals():
                del gate_proj, up_proj, activated

        output = _trinity_accumulate_routed_output(
            shared_output=shared_output,
            routed_output=routed_output,
            top_scores_sorted=top_scores_sorted,
            token_indices_sorted=token_indices_sorted,
        )

        aux_loss_coef = float(getattr(self.config, "router_aux_loss_coef", 0.0) or 0.0)
        self._last_router_aux_loss = None
        if aux_loss_coef > 0.0:
            self._last_router_aux_loss = _trinity_router_aux_loss(
                router_probs,
                selected_experts,
                num_experts=self.config.num_experts,
            )

        self._last_router_logits = None
        if getattr(self.config, "output_router_logits", False):
            self._last_router_logits = router_probs.view(batch_size, seq_len, self.config.num_experts)

        return output.to(hidden_states.dtype).view(batch_size, seq_len, hidden_dim)

    def patched_afmoe_model_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = modeling_module.DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": modeling_module.create_causal_mask(**mask_kwargs),
                "sliding_attention": modeling_module.create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        if self.config.mup_enabled:
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        router_aux_losses = []
        collect_router_logits = bool(getattr(self.config, "output_router_logits", False))
        router_logits = [] if collect_router_logits else None

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if not getattr(decoder_layer, "moe_enabled", False):
                continue

            layer_aux_loss = getattr(decoder_layer.mlp, "_last_router_aux_loss", None)
            if layer_aux_loss is not None:
                router_aux_losses.append(layer_aux_loss)
            if router_logits is not None:
                layer_router_logits = getattr(decoder_layer.mlp, "_last_router_logits", None)
                if layer_router_logits is not None:
                    router_logits.append(layer_router_logits)
            decoder_layer.mlp._last_router_aux_loss = None
            decoder_layer.mlp._last_router_logits = None

        hidden_states = self.norm(hidden_states)
        self._last_router_aux_loss = (
            torch.stack(router_aux_losses).mean() if router_aux_losses else None
        )

        return modeling_module.MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=tuple(router_logits) if router_logits else None,
        )

    def patched_afmoe_for_causal_lm_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        del token_type_ids
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        aux_loss = getattr(self.model, "_last_router_aux_loss", None)
        aux_loss_coef = float(getattr(self.config, "router_aux_loss_coef", 0.0) or 0.0)
        use_cce = (
            labels is not None
            and bool(getattr(self.config, "use_cut_cross_entropy", False))
            and linear_cross_entropy is not None
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 0
        )

        logits = None
        loss = None
        if use_cce:
            loss = linear_cross_entropy(
                hidden_states,
                self.lm_head.weight,
                labels.to(hidden_states.device),
                bias=getattr(self.lm_head, "bias", None),
                ignore_index=-100,
                shift=True,
                impl=getattr(self.config, "cut_cross_entropy_impl", "cce"),
                filter_eps=getattr(self.config, "cut_cross_entropy_filter_eps", "auto"),
                accum_e_fp32=bool(
                    getattr(self.config, "cut_cross_entropy_accum_e_fp32", False)
                ),
                accum_c_fp32=bool(
                    getattr(self.config, "cut_cross_entropy_accum_c_fp32", False)
                ),
                filter_e_grad=bool(
                    getattr(self.config, "cut_cross_entropy_filter_e_grad", True)
                ),
                filter_c_grad=bool(
                    getattr(self.config, "cut_cross_entropy_filter_c_grad", True)
                ),
                reduction="mean",
            )
        else:
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None and loss is None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        if loss is not None and aux_loss is not None and aux_loss_coef > 0.0:
            loss = loss + (aux_loss * aux_loss_coef)

        return modeling_module.MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    modeling_module.AfmoeMoE.forward = patched_afmoe_moe_forward
    modeling_module.AfmoeModel.forward = patched_afmoe_model_forward
    modeling_module.AfmoeForCausalLM.forward = patched_afmoe_for_causal_lm_forward
    modeling_module._trinity_vlm_patched_afmoe_moe_forward = patched_afmoe_moe_forward
    modeling_module._trinity_vlm_patched_afmoe_model_forward = patched_afmoe_model_forward
    modeling_module._trinity_vlm_patched_afmoe_for_causal_lm_forward = patched_afmoe_for_causal_lm_forward
    modeling_module._trinity_vlm_grouped_moe_patched = True


def _patch_trinity_language_model_instance(language_model: nn.Module, modeling_module) -> None:
    patched_model_forward = getattr(modeling_module, "_trinity_vlm_patched_afmoe_model_forward", None)
    if patched_model_forward is not None:
        language_model.model.forward = types.MethodType(
            patched_model_forward,
            language_model.model,
        )

    patched_lm_forward = getattr(modeling_module, "_trinity_vlm_patched_afmoe_for_causal_lm_forward", None)
    if patched_lm_forward is not None:
        language_model.forward = types.MethodType(
            patched_lm_forward,
            language_model,
        )

    patched_moe_forward = getattr(modeling_module, "_trinity_vlm_patched_afmoe_moe_forward", None)
    if patched_moe_forward is None:
        return

    for decoder_layer in getattr(language_model.model, "layers", ()):
        if not getattr(decoder_layer, "moe_enabled", False):
            continue
        decoder_layer.mlp.forward = types.MethodType(
            patched_moe_forward,
            decoder_layer.mlp,
        )


class VisionBridge(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, dtype=dtype)
        self.fc1 = nn.Linear(in_dim, hidden_dim, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.nn.functional.gelu(self.fc1(x), approximate="tanh")
        return self.fc2(x)


class TrinityVLMForConditionalGeneration(PreTrainedModel):
    config_class = TrinityVLMConfig
    base_model_prefix = "trinity_vlm"
    main_input_name = "input_ids"

    def __init__(
        self,
        config: TrinityVLMConfig,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(config)
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self._profiling_enabled = False
        self._profiling_synchronize_cuda = False
        self._profile_stats: dict[str, float] = {}
        if config.use_cut_cross_entropy and linear_cross_entropy is None:
            raise ImportError(
                "use_cut_cross_entropy=True requires the cut-cross-entropy package to be installed."
            )

        trinity_text_config = self._load_trinity_text_config(
            config,
            local_files_only=local_files_only,
        )
        self._trinity_modeling_module = importlib.import_module(
            trinity_text_config.__class__.__module__.replace("configuration_", "modeling_")
        )

        print("startup: loading Trinity language model", flush=True)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.trinity_model_name_or_path,
            config=trinity_text_config,
            revision=config.trinity_revision,
            trust_remote_code=config.trust_remote_code,
            dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        print("startup: Trinity language model loaded", flush=True)
        self.language_model.config.use_cache = False
        _patch_trinity_language_model_instance(self.language_model, self._trinity_modeling_module)
        packed_moe_layers = _trinity_pack_moe_layers(self.language_model)
        print(f"startup: packed Trinity MoE experts in {packed_moe_layers} layers", flush=True)
        self._install_language_model_helpers()

        print("startup: loading Moondream vision tower", flush=True)
        self.vision_tower = MoondreamVisionTower.from_hf(
            config.moondream_model_name_or_path,
            dtype=torch_dtype,
            revision=config.moondream_revision,
            local_files_only=local_files_only,
        )
        print("startup: Moondream vision tower loaded", flush=True)
        self.multi_modal_projector = VisionBridge(
            in_dim=config.vision_feature_dim,
            hidden_dim=config.projector_hidden_dim,
            out_dim=self.language_model.config.hidden_size,
            dtype=torch_dtype,
        )
        self.special_token_embeddings = nn.Embedding(
            3,
            self.language_model.config.hidden_size,
            dtype=torch_dtype,
        )
        with torch.no_grad():
            mean_embedding = self.get_input_embeddings().weight.detach().mean(dim=0, keepdim=True)
            self.special_token_embeddings.weight.copy_(mean_embedding.repeat(3, 1))

        self.config.hidden_size = self.language_model.config.hidden_size
        self.config.vocab_size = self.language_model.config.vocab_size
        self.config.bos_token_id = self.language_model.config.bos_token_id
        self.config.eos_token_id = self.language_model.config.eos_token_id
        self.config.pad_token_id = self.language_model.config.pad_token_id

    @staticmethod
    def _load_trinity_text_config(
        config: TrinityVLMConfig,
        *,
        local_files_only: bool,
    ):
        print("startup: loading Trinity config", flush=True)
        text_config = AutoConfig.from_pretrained(
            config.trinity_model_name_or_path,
            revision=config.trinity_revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=local_files_only,
        )
        print("startup: Trinity config loaded", flush=True)
        missing_token_ids = [
            token_name
            for token_name in ("pad_token_id", "bos_token_id", "eos_token_id")
            if getattr(text_config, token_name, None) is None
        ]
        if missing_token_ids:
            print("startup: loading Trinity tokenizer for missing token ids", flush=True)
            tokenizer = load_trinity_tokenizer(
                config.trinity_model_name_or_path,
                revision=config.trinity_revision,
                local_files_only=local_files_only,
            )
            for token_name in missing_token_ids:
                setattr(text_config, token_name, getattr(tokenizer, token_name))
            print("startup: Trinity token ids patched into config", flush=True)

        print("startup: importing Trinity remote modeling module", flush=True)
        modeling_module_name = text_config.__class__.__module__.replace("configuration_", "modeling_")
        modeling_module = importlib.import_module(modeling_module_name)
        rotary_cls = getattr(modeling_module, "AfmoeRotaryEmbedding", None)
        if rotary_cls is not None and not hasattr(rotary_cls, "compute_default_rope_parameters"):
            rotary_cls.compute_default_rope_parameters = _compat_compute_default_rope_parameters
        text_config.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.0)
        text_config.enable_grouped_moe = getattr(config, "enable_grouped_moe", True)
        text_config.output_router_logits = getattr(config, "output_router_logits", False)
        _patch_trinity_remote_modeling_module(modeling_module)
        print("startup: Trinity remote modeling module ready", flush=True)
        return text_config

    @property
    def image_special_tokens(self) -> list[str]:
        return [
            self.config.vision_start_token,
            self.config.vision_end_token,
            self.config.image_token,
        ]

    def _install_language_model_helpers(self) -> None:
        if hasattr(self.language_model, "embed_input_ids"):
            return

        def embed_input_ids(module, input_ids: torch.Tensor) -> torch.Tensor:
            return module.get_input_embeddings()(input_ids)

        self.language_model.embed_input_ids = types.MethodType(embed_input_ids, self.language_model)

    def configure_profiling(
        self,
        *,
        enabled: bool,
        synchronize_cuda: bool = False,
    ) -> None:
        self._profiling_enabled = enabled
        self._profiling_synchronize_cuda = synchronize_cuda
        self._profile_stats = {}

    def reset_profile_stats(self) -> None:
        self._profile_stats = {}

    def pop_profile_stats(self) -> dict[str, float]:
        stats = dict(self._profile_stats)
        self._profile_stats = {}
        return stats

    def _synchronize_for_profile(self) -> None:
        if (
            self._profiling_enabled
            and self._profiling_synchronize_cuda
            and self.device.type == "cuda"
        ):
            torch.cuda.synchronize(self.device)

    def _profile_call(self, key: str, fn):
        if not self._profiling_enabled:
            return fn()
        self._synchronize_for_profile()
        start_time = time.perf_counter()
        result = fn()
        self._synchronize_for_profile()
        self._profile_stats[key] = self._profile_stats.get(key, 0.0) + (
            time.perf_counter() - start_time
        )
        return result

    @property
    def device(self) -> torch.device:
        return self.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.get_input_embeddings().weight.dtype

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self):
        if hasattr(self.language_model, "get_decoder"):
            return self.language_model.get_decoder()
        return self.language_model

    def set_decoder(self, decoder):
        if hasattr(self.language_model, "set_decoder"):
            self.language_model.set_decoder(decoder)
        else:
            self.language_model = decoder

    @property
    def bridge(self):
        return self.multi_modal_projector

    @bridge.setter
    def bridge(self, value):
        self.multi_modal_projector = value

    def ensure_image_special_tokens(self, tokenizer) -> dict[str, int | None]:
        vocab = tokenizer.get_vocab()
        tokens_to_add = [token for token in self.image_special_tokens if token not in vocab]
        added = 0
        if tokens_to_add:
            added = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        max_token_id = max(tokenizer.get_vocab().values())
        required_vocab_size = max_token_id + 1
        current_vocab_size = int(self.get_input_embeddings().weight.shape[0])
        if required_vocab_size > current_vocab_size:
            self.language_model.resize_token_embeddings(required_vocab_size)
            current_vocab_size = int(self.get_input_embeddings().weight.shape[0])

        self.config.vision_start_token_id = tokenizer.convert_tokens_to_ids(
            self.config.vision_start_token
        )
        self.config.vision_end_token_id = tokenizer.convert_tokens_to_ids(
            self.config.vision_end_token
        )
        self.config.image_token_id = tokenizer.convert_tokens_to_ids(self.config.image_token)
        self.config.bos_token_id = tokenizer.bos_token_id
        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id
        self.config.vocab_size = current_vocab_size

        return {
            "added_tokens": added,
            "required_vocab_size": required_vocab_size,
            "language_model_vocab_size": current_vocab_size,
            "vision_start_token_id": self.config.vision_start_token_id,
            "vision_end_token_id": self.config.vision_end_token_id,
            "image_token_id": self.config.image_token_id,
            "bos_token_id": self.config.bos_token_id,
            "eos_token_id": self.config.eos_token_id,
            "pad_token_id": self.config.pad_token_id,
        }

    def build_image_token_span(self, *, include_bos: bool = True) -> list[int]:
        return build_image_token_span(
            vision_start_token_id=self.config.vision_start_token_id,
            image_token_id=self.config.image_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            image_seq_len=self.config.image_seq_len,
            bos_token_id=self.config.bos_token_id if include_bos else None,
        )

    def _project_image_feature_tensor(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.shape[-1] == self.language_model.config.hidden_size:
            return image_features.to(device=self.device, dtype=self.dtype)
        if image_features.shape[-1] == self.config.vision_feature_dim:
            return self.bridge(image_features.to(device=self.device, dtype=self.dtype))
        raise ValueError(
            "Tensor image features must already be in Trinity hidden size or Moondream feature size."
        )

    @torch.compiler.disable
    def encode_images(
        self,
        images: list[Any] | list[list[Any]] | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if images is None:
            return None

        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                projected = self._project_image_feature_tensor(images)
                image_counts = torch.ones(
                    projected.size(0),
                    device=projected.device,
                    dtype=torch.long,
                )
                return projected.unsqueeze(1), image_counts
            if images.ndim == 4:
                projected = self._project_image_feature_tensor(images)
                image_counts = torch.full(
                    (projected.size(0),),
                    projected.size(1),
                    device=projected.device,
                    dtype=torch.long,
                )
                return projected, image_counts
            raise ValueError("Tensor images must have shape [batch, seq, dim] or [batch, images, seq, dim].")

        if not isinstance(images, (list, tuple)):
            raise TypeError(f"Unsupported image batch type: {type(images)!r}")

        if not images:
            empty_embeds = torch.empty(
                0,
                0,
                self.config.image_seq_len,
                self.language_model.config.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
            empty_counts = torch.empty(0, device=self.device, dtype=torch.long)
            return empty_embeds, empty_counts

        first_item = images[0]
        if isinstance(first_item, Image.Image):
            image_batches = [[image] for image in images]
        elif isinstance(first_item, (list, tuple)):
            image_batches = [list(sample_images) for sample_images in images]
        else:
            raise TypeError(f"Unsupported image batch type: {type(first_item)!r}")

        batch_size = len(image_batches)
        image_count_list = [len(sample_images) for sample_images in image_batches]
        image_counts = torch.tensor(
            image_count_list,
            device=self.device,
            dtype=torch.long,
        )
        max_images = max(image_count_list) if image_count_list else 0
        encoded_batches = torch.zeros(
            batch_size,
            max_images,
            self.config.image_seq_len,
            self.language_model.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )

        flat_images: list[Image.Image] = []
        sample_offsets: list[tuple[int, int]] = []
        running_offset = 0
        for sample_images in image_batches:
            for image in sample_images:
                if not isinstance(image, Image.Image):
                    raise TypeError(f"Expected PIL images, got {type(image)!r}")
            flat_images.extend(sample_images)
            next_offset = running_offset + len(sample_images)
            sample_offsets.append((running_offset, next_offset))
            running_offset = next_offset

        if not flat_images:
            return encoded_batches, image_counts

        if self._profiling_enabled:
            image_features = self._profile_call(
                "model/vision_encode_s",
                lambda: self.vision_tower.encode_images(flat_images).to(
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            projected_image_features = self._profile_call(
                "model/bridge_project_s",
                lambda: self.bridge(image_features),
            )
        else:
            image_features = self.vision_tower.encode_images(flat_images).to(
                device=self.device,
                dtype=self.dtype,
            )
            projected_image_features = self.bridge(image_features)

        for batch_index, (start_offset, end_offset) in enumerate(sample_offsets):
            if start_offset == end_offset:
                continue
            encoded_batches[batch_index, : end_offset - start_offset] = projected_image_features[
                start_offset:end_offset
            ]
        return encoded_batches, image_counts

    def _inject_special_token_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        token_id_to_index = {
            self.config.vision_start_token_id: 0,
            self.config.vision_end_token_id: 1,
            self.config.image_token_id: 2,
        }
        if all(token_id is None for token_id in token_id_to_index):
            return inputs_embeds

        token_ids = torch.tensor(
            [
                token_id_to_index_key if token_id_to_index_key is not None else -1
                for token_id_to_index_key in token_id_to_index
            ],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        special_matches = input_ids.unsqueeze(-1) == token_ids.view(1, 1, -1)
        special_mask = special_matches.any(dim=-1)

        replacement_indices = special_matches.to(torch.long).argmax(dim=-1)
        special_embeds = self.special_token_embeddings(replacement_indices.to(self.device)).to(dtype=self.dtype)
        return torch.where(
            special_mask.unsqueeze(-1),
            special_embeds,
            inputs_embeds,
        )

    def _inject_image_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        image_embeds_by_sample: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        image_token_id = self.config.image_token_id
        if image_token_id is None:
            raise ValueError("image_token_id is not configured. Call ensure_image_special_tokens(tokenizer).")

        image_token_mask = input_ids == image_token_id
        image_token_counts = image_token_mask.sum(dim=1)

        if image_embeds_by_sample is None:
            torch._assert(
                torch.logical_not(torch.any(image_token_counts > 0)),
                "Input contains image placeholder tokens but no images were provided.",
            )
            return inputs_embeds

        image_embeds, image_counts = image_embeds_by_sample
        torch._assert(
            image_embeds.ndim == 4,
            "Image embeddings must have shape [batch, max_images, seq, hidden].",
        )
        torch._assert(
            image_counts.ndim == 1,
            "Image counts must have shape [batch].",
        )
        torch._assert(
            image_embeds.size(0) == input_ids.size(0),
            "Image batch size must match text batch size.",
        )
        torch._assert(
            image_counts.size(0) == input_ids.size(0),
            "Image count batch size must match text batch size.",
        )

        start_id = self.config.vision_start_token_id
        end_id = self.config.vision_end_token_id
        num_images = image_counts.to(image_token_counts.dtype)
        per_image_seq_len = image_embeds.size(2)
        expected_image_token_counts = num_images * per_image_seq_len

        torch._assert(
            torch.all(image_token_counts == expected_image_token_counts),
            "Image placeholder count mismatch.",
        )

        if start_id is not None:
            start_counts = (input_ids == start_id).sum(dim=1)
            torch._assert(
                torch.all(start_counts == num_images),
                "vision_start token count mismatch.",
            )
        if end_id is not None:
            end_counts = (input_ids == end_id).sum(dim=1)
            torch._assert(
                torch.all(end_counts == num_images),
                "vision_end token count mismatch.",
            )

        if image_embeds.size(1) == 0:
            return inputs_embeds

        flat_image_embeds = image_embeds.flatten(1, 2)
        hidden_size = inputs_embeds.size(-1)
        image_token_offsets = image_token_mask.to(torch.long).cumsum(dim=1) - 1
        clamped_offsets = image_token_offsets.clamp_min(0)
        gathered_image_embeds = flat_image_embeds.gather(
            dim=1,
            index=clamped_offsets.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        valid_image_mask = image_token_mask & (
            clamped_offsets < expected_image_token_counts.unsqueeze(1)
        )
        return torch.where(
            valid_image_mask.unsqueeze(-1),
            gathered_image_embeds,
            inputs_embeds,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if attention_mask is None:
            return torch.ones(
                (batch_size, seq_len),
                dtype=torch.long,
                device=self.device,
            )

        if isinstance(attention_mask, dict):
            return {
                key: value.to(self.device)
                for key, value in attention_mask.items()
            }

        return attention_mask.to(self.device)

    def prepare_multimodal_inputs(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        images: list[Any] | list[list[Any]] | torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided.")
            input_ids = input_ids.to(self.device)
            if self._profiling_enabled:
                inputs_embeds = self._profile_call(
                    "model/embed_input_ids_s",
                    lambda: self.language_model.embed_input_ids(input_ids),
                )
            else:
                inputs_embeds = self.language_model.embed_input_ids(input_ids)
        else:
            inputs_embeds = inputs_embeds.to(self.device, dtype=self.dtype)

        batch_size, seq_len = inputs_embeds.shape[:2]
        attention_mask = self._prepare_attention_mask(attention_mask, batch_size, seq_len)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long).unsqueeze(0).expand(
                batch_size,
                seq_len,
            )
        else:
            position_ids = position_ids.to(self.device)

        if self._profiling_enabled:
            image_embeds_by_sample = self._profile_call(
                "model/prepare_images_s",
                lambda: self.encode_images(images),
            )
        else:
            image_embeds_by_sample = self.encode_images(images)

        if input_ids is None:
            raise ValueError("input_ids are required for multimodal placeholder replacement.")

        if self._profiling_enabled:
            inputs_embeds = self._profile_call(
                "model/inject_embeddings_s",
                lambda: self._inject_image_embeddings(
                    inputs_embeds=self._inject_special_token_embeddings(
                        inputs_embeds=inputs_embeds,
                        input_ids=input_ids,
                    ),
                    input_ids=input_ids,
                    image_embeds_by_sample=image_embeds_by_sample,
                ),
            )
        else:
            inputs_embeds = self._inject_image_embeddings(
                inputs_embeds=self._inject_special_token_embeddings(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                ),
                input_ids=input_ids,
                image_embeds_by_sample=image_embeds_by_sample,
            )

        if labels is not None:
            labels = labels.to(self.device)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        images: list[Any] | list[list[Any]] | torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        prepared = self.prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            images=images,
            inputs_embeds=inputs_embeds,
        )
        if self._profiling_enabled:
            return self._profile_call(
                "model/language_model_forward_s",
                lambda: self.language_model(
                    input_ids=None,
                    inputs_embeds=prepared["inputs_embeds"],
                    attention_mask=prepared["attention_mask"],
                    position_ids=prepared["position_ids"],
                    labels=prepared["labels"],
                    use_cache=False,
                    **kwargs,
                ),
            )

        return self.language_model(
            input_ids=None,
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            position_ids=prepared["position_ids"],
            labels=prepared["labels"],
            use_cache=False,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        images: list[Any] | list[list[Any]] | torch.Tensor | None = None,
        **kwargs,
    ):
        if images is None:
            return self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )

        if input_ids is None:
            if isinstance(images, torch.Tensor):
                batch_size = images.size(0)
            else:
                if images and isinstance(images[0], (list, tuple)):
                    raise ValueError(
                        "Automatic multimodal generation only supports one image per sample. "
                        "Provide explicit input_ids for multi-image prompts."
                    )
                batch_size = len(images)
            prefix = self.build_image_token_span(include_bos=True)
            input_ids = torch.tensor(
                [prefix] * batch_size,
                device=self.device,
                dtype=torch.long,
            )

        prepared = self.prepare_multimodal_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,
            images=images,
        )
        return self.language_model.generate(
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"],
            position_ids=prepared["position_ids"],
            **kwargs,
        )

    def export_pretrained(
        self,
        save_directory: str | Path,
        *,
        safe_serialization: bool = True,
        max_shard_size: str = "10GB",
    ) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.config.architectures = [self.__class__.__name__]
        self.config.auto_map = {
            "AutoConfig": "configuration_trinity_vlm.TrinityVLMConfig",
            "AutoModelForCausalLM": "modeling_trinity_vlm.TrinityVLMForConditionalGeneration",
        }
        _normalize_tied_weight_keys_for_save_(self)
        filtered_state_dict = {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("trinity_model.")
        }

        self.save_pretrained(
            save_path,
            state_dict=filtered_state_dict,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )

        package_dir = Path(__file__).resolve().parent
        for filename in [
            "__init__.py",
            "configuration_trinity_vlm.py",
            "image_tokens.py",
            "modeling_trinity_vlm.py",
            "moondream_vision.py",
        ]:
            shutil.copy2(package_dir / filename, save_path / filename)
