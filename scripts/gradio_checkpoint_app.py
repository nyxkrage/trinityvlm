#!/home/carsten/trinity-vlm/.venv/bin/python
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import fields
from pathlib import Path
from threading import RLock
from typing import Any

import gradio as gr
import torch
from PIL import Image
from transformers import PreTrainedTokenizerFast

from trinity_vlm.chat_content import build_user_message_content
from trinity_vlm.graft import TrinityMoondreamGraftConfig, TrinityMoondreamGraftModel
from trinity_vlm.tokenizer_utils import install_trinity_chat_template, load_trinity_tokenizer


DEFAULT_RUN_DIR = Path("checkpoints/bridge-mix-pixmo-nemotron-local-20260415-ddp-cce")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a small Gradio app for testing a TrinityVLM delta checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
        help="Training output directory that contains checkpoints/, tokenizer/, and latest_checkpoint.txt.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Specific checkpoint directory. If omitted, latest_checkpoint.txt from --run-dir is used.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu, auto, cuda, or cuda:N. Default is cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load base models and tokenizer from the local HF cache only.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if normalized == "cuda":
            return torch.device("cuda:0")
        return torch.device(normalized)
    if normalized == "auto" and torch.cuda.is_available():
        best_index = 0
        best_free_bytes = -1
        for device_index in range(torch.cuda.device_count()):
            with torch.cuda.device(device_index):
                free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes > best_free_bytes:
                best_free_bytes = free_bytes
                best_index = device_index
        return torch.device(f"cuda:{best_index}")
    return torch.device("cpu")


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = getattr(torch, dtype_name)
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def normalize_image(image: Image.Image | Any | None) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def load_saved_tokenizer(tokenizer_dir: Path):
    tokenizer_kwargs = {
        "local_files_only": True,
        "fix_mistral_regex": True,
    }
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(tokenizer_dir),
            **tokenizer_kwargs,
        )
    except AttributeError as exc:
        if "backend_tokenizer" not in str(exc):
            raise
        print(
            "startup: transformers failed to apply fix_mistral_regex; "
            "falling back to the raw saved tokenizer",
            flush=True,
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
        )
    install_trinity_chat_template(tokenizer)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_run_dir(run_dir: str | Path) -> Path:
    resolved = Path(run_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Run directory does not exist: {resolved}")
    return resolved


def list_run_checkpoints(run_dir: Path) -> list[str]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    return sorted(
        (path.name for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.startswith("step-")),
        key=lambda name: int(name.split("-")[-1]),
    )


def resolve_checkpoint_dir(
    run_dir: Path,
    checkpoint_dir: str | Path | None,
) -> Path:
    if checkpoint_dir is not None:
        candidate = Path(checkpoint_dir).expanduser()
        if not candidate.is_absolute():
            candidate_options = [
                (run_dir / candidate).resolve(),
                (run_dir / "checkpoints" / candidate).resolve(),
            ]
        else:
            candidate_options = [candidate.resolve()]
        for resolved_candidate in candidate_options:
            if (resolved_candidate / "trainable.safetensors").exists():
                return resolved_candidate
        raise FileNotFoundError(f"Checkpoint directory does not contain trainable.safetensors: {candidate}")

    latest_path = run_dir / "latest_checkpoint.txt"
    if latest_path.exists():
        latest_checkpoint = (run_dir / latest_path.read_text().strip()).resolve()
        if (latest_checkpoint / "trainable.safetensors").exists():
            return latest_checkpoint

    checkpoint_names = list_run_checkpoints(run_dir)
    if not checkpoint_names:
        raise FileNotFoundError(f"No checkpoint directories found under {run_dir / 'checkpoints'}")
    return (run_dir / "checkpoints" / checkpoint_names[-1]).resolve()


def load_graft_config(checkpoint_dir: Path) -> TrinityMoondreamGraftConfig:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        return TrinityMoondreamGraftConfig()

    metadata = json.loads(metadata_path.read_text())
    config_data = metadata.get("config", {})
    valid_fields = {field.name for field in fields(TrinityMoondreamGraftConfig)}
    filtered_config = {
        key: value
        for key, value in config_data.items()
        if key in valid_fields
    }
    return TrinityMoondreamGraftConfig(**filtered_config)


def format_load_status(
    checkpoint_dir: Path,
    inference_device: torch.device,
    dtype: torch.dtype,
    token_info: dict[str, int | None],
) -> str:
    trainer_state_path = checkpoint_dir / "trainer_state.pt"
    global_step = None
    if trainer_state_path.exists():
        trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        global_step = int(trainer_state["global_step"])

    status = {
        "checkpoint_dir": str(checkpoint_dir),
        "global_step": global_step,
        "checkpoint_resident_device": str(inference_device),
        "inference_device": str(inference_device),
        "vision_device": str(inference_device),
        "dtype": str(dtype),
        "image_token_id": token_info.get("image_token_id"),
        "vision_start_token_id": token_info.get("vision_start_token_id"),
        "vision_end_token_id": token_info.get("vision_end_token_id"),
    }
    return json.dumps(status, indent=2)


class CheckpointApp:
    def __init__(
        self,
        *,
        run_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
        local_files_only: bool,
    ) -> None:
        self.run_dir = run_dir
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self._lock = RLock()
        self.model: TrinityMoondreamGraftModel | None = None
        self.tokenizer = None
        self.loaded_checkpoint: Path | None = None
        self.vision_device = self.device

    def _release_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_tokenizer(self, graft_config: TrinityMoondreamGraftConfig):
        tokenizer_dir = self.run_dir / "tokenizer"
        if tokenizer_dir.exists():
            return load_saved_tokenizer(tokenizer_dir)
        return load_trinity_tokenizer(
            graft_config.trinity_model_name,
            local_files_only=self.local_files_only,
        )

    def load_checkpoint(self, checkpoint_name: str | None = None) -> tuple[str, str]:
        with self._lock:
            checkpoint_dir = resolve_checkpoint_dir(self.run_dir, checkpoint_name)
            graft_config = load_graft_config(checkpoint_dir)

            self._release_model()

            model = TrinityMoondreamGraftModel(
                graft_config,
                torch_dtype=self.dtype,
                local_files_only=self.local_files_only,
            )
            tokenizer = self._load_tokenizer(graft_config)
            token_info = model.ensure_image_special_tokens(tokenizer)
            model.load_delta(checkpoint_dir)
            model.to(self.device)
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            self.loaded_checkpoint = checkpoint_dir

            status = format_load_status(
                checkpoint_dir=checkpoint_dir,
                inference_device=self.device,
                dtype=self.dtype,
                token_info=token_info,
            )
            return str(checkpoint_dir.name), status

    def refresh_checkpoint_choices(self, checkpoint_name: str | None = None):
        with self._lock:
            checkpoint_names = list_run_checkpoints(self.run_dir)
            loaded_name = self.loaded_checkpoint.name if self.loaded_checkpoint is not None else None

            selected_name = checkpoint_name
            if selected_name not in checkpoint_names:
                selected_name = loaded_name
            if selected_name not in checkpoint_names and checkpoint_names:
                selected_name = checkpoint_names[-1]

            return gr.update(
                choices=checkpoint_names,
                value=selected_name,
            )

    def generate(
        self,
        checkpoint_name: str,
        image: Image.Image | Any | None,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[str, str]:
        with self._lock:
            if self.loaded_checkpoint is None or self.loaded_checkpoint.name != checkpoint_name:
                self.load_checkpoint(checkpoint_name)

            if self.model is None or self.tokenizer is None:
                raise gr.Error("Model is not loaded.")

            pil_image = normalize_image(image)
            prompt = prompt.strip()

            if pil_image is None and not prompt:
                raise gr.Error("Provide an image, a prompt, or both.")

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "repetition_penalty": float(repetition_penalty),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                    }
                )
            else:
                generation_kwargs["do_sample"] = False

            input_ids = None
            attention_mask = None
            images = None

            user_content = build_user_message_content(
                self.tokenizer,
                prompt_text=prompt,
                user_message_template="{image}\n{prompt}",
                include_image=pil_image is not None,
                max_prompt_tokens=256,
            )
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                image_seq_len=self.model.config.image_seq_len,
            )
            input_ids = torch.tensor(
                [encoded["input_ids"]],
                device=self.device,
                dtype=torch.long,
            )
            attention_mask = torch.tensor(
                [encoded["attention_mask"]],
                device=self.device,
                dtype=torch.long,
            )
            if pil_image is not None:
                images = [pil_image]

            autocast_enabled = self.device.type == "cuda" and self.dtype != torch.float32
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=autocast_enabled,
                ):
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                        **generation_kwargs,
                    )

            prompt_len = int(input_ids.shape[1]) if input_ids is not None else 0
            sequence_ids = output_ids[0]
            generated_ids = sequence_ids[prompt_len:] if sequence_ids.numel() > prompt_len else sequence_ids
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not text:
                text = self.tokenizer.decode(sequence_ids, skip_special_tokens=True).strip()

            metadata = {
                "checkpoint": str(self.loaded_checkpoint),
                "prompt_tokens": prompt_len,
                "generated_tokens": int(generated_ids.numel()),
            }
            return text, json.dumps(metadata, indent=2)


def build_demo(
    app: CheckpointApp,
    checkpoint_names: list[str],
    initial_checkpoint_name: str,
    initial_status: str,
) -> gr.Blocks:
    with gr.Blocks(title="TrinityVLM Checkpoint Tester") as demo:
        gr.Markdown(
            "# TrinityVLM Checkpoint Tester\n"
            f"Run dir: `{app.run_dir}`"
        )

        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                label="Checkpoint",
                choices=checkpoint_names,
                value=initial_checkpoint_name,
                allow_custom_value=False,
            )
            load_button = gr.Button("Load Checkpoint", variant="secondary")
            refresh_button = gr.Button("Refresh Checkpoints")

        load_status = gr.Code(
            label="Loaded Checkpoint",
            language="json",
            value=initial_status,
        )

        with gr.Row():
            image_input = gr.Image(label="Image", type="pil")
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="Describe this image.",
                    lines=4,
                )
                max_new_tokens = gr.Slider(
                    label="Max New Tokens",
                    minimum=16,
                    maximum=512,
                    step=16,
                    value=160,
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    step=0.05,
                    value=0.0,
                )
                top_p = gr.Slider(
                    label="Top P",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.95,
                )
                repetition_penalty = gr.Slider(
                    label="Repetition Penalty",
                    minimum=1.0,
                    maximum=1.5,
                    step=0.01,
                    value=1.05,
                )
                generate_button = gr.Button("Generate", variant="primary")

        output_text = gr.Textbox(label="Output", lines=12)
        output_metadata = gr.Code(label="Generation Metadata", language="json")

        load_button.click(
            fn=app.load_checkpoint,
            inputs=[checkpoint_dropdown],
            outputs=[checkpoint_dropdown, load_status],
        )
        refresh_button.click(
            fn=app.refresh_checkpoint_choices,
            inputs=[checkpoint_dropdown],
            outputs=[checkpoint_dropdown],
        )
        generate_button.click(
            fn=app.generate,
            inputs=[
                checkpoint_dropdown,
                image_input,
                prompt_input,
                max_new_tokens,
                temperature,
                top_p,
                repetition_penalty,
            ],
            outputs=[output_text, output_metadata],
        )

    return demo


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    checkpoint_dir = resolve_checkpoint_dir(run_dir, args.checkpoint_dir)
    checkpoint_names = list_run_checkpoints(run_dir)
    initial_checkpoint_name = checkpoint_dir.name
    if initial_checkpoint_name not in checkpoint_names:
        checkpoint_names.append(initial_checkpoint_name)
        checkpoint_names = sorted(checkpoint_names, key=lambda name: int(name.split("-")[-1]))

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    app = CheckpointApp(
        run_dir=run_dir,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    initial_checkpoint_name, initial_status = app.load_checkpoint(initial_checkpoint_name)
    demo = build_demo(
        app,
        checkpoint_names,
        initial_checkpoint_name,
        initial_status,
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
