from __future__ import annotations

from transformers import PretrainedConfig


class TrinityVLMConfig(PretrainedConfig):
    model_type = "trinity_vlm"
    is_composition = True

    def __init__(
        self,
        trinity_model_name_or_path: str = "arcee-ai/Trinity-Nano-Preview",
        moondream_model_name_or_path: str = "moondream/moondream3-preview",
        projector_hidden_dim: int = 2048,
        vision_feature_dim: int = 2048,
        image_seq_len: int = 729,
        router_aux_loss_coef: float = 0.001,
        use_cut_cross_entropy: bool = False,
        cut_cross_entropy_impl: str = "cce",
        cut_cross_entropy_filter_eps: float | str | None = "auto",
        cut_cross_entropy_accum_e_fp32: bool = False,
        cut_cross_entropy_accum_c_fp32: bool = False,
        cut_cross_entropy_filter_e_grad: bool = True,
        cut_cross_entropy_filter_c_grad: bool = True,
        enable_grouped_moe: bool = True,
        output_router_logits: bool = False,
        vision_start_token: str = "<|vision_start|>",
        vision_end_token: str = "<|vision_end|>",
        image_token: str = "<|image_pad|>",
        vision_start_token_id: int | None = None,
        vision_end_token_id: int | None = None,
        image_token_id: int | None = None,
        trust_remote_code: bool = True,
        trinity_revision: str | None = None,
        moondream_revision: str | None = None,
        hidden_size: int | None = None,
        vocab_size: int | None = None,
        **kwargs,
    ) -> None:
        self.trinity_model_name_or_path = trinity_model_name_or_path
        self.moondream_model_name_or_path = moondream_model_name_or_path
        self.projector_hidden_dim = projector_hidden_dim
        self.vision_feature_dim = vision_feature_dim
        self.image_seq_len = image_seq_len
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_cut_cross_entropy = use_cut_cross_entropy
        self.cut_cross_entropy_impl = cut_cross_entropy_impl
        self.cut_cross_entropy_filter_eps = cut_cross_entropy_filter_eps
        self.cut_cross_entropy_accum_e_fp32 = cut_cross_entropy_accum_e_fp32
        self.cut_cross_entropy_accum_c_fp32 = cut_cross_entropy_accum_c_fp32
        self.cut_cross_entropy_filter_e_grad = cut_cross_entropy_filter_e_grad
        self.cut_cross_entropy_filter_c_grad = cut_cross_entropy_filter_c_grad
        self.enable_grouped_moe = enable_grouped_moe
        self.output_router_logits = output_router_logits
        self.vision_start_token = vision_start_token
        self.vision_end_token = vision_end_token
        self.image_token = image_token
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.image_token_id = image_token_id
        self.trust_remote_code = trust_remote_code
        self.trinity_revision = trinity_revision
        self.moondream_revision = moondream_revision
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        super().__init__(**kwargs)
