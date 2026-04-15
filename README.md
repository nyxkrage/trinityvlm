# Trinity VLM Experiments

This repo is a scaffold for grafting the Moondream 3 preview vision encoder onto `arcee-ai/Trinity-Nano-Preview` and training the bridge on `anthracite-org/pixmo-cap-images`.

The current setup is intentionally simple:

- Trinity is loaded by HF repo id through the shared Hugging Face cache.
- Moondream vision weights are loaded from the cached `moondream/moondream3-preview` repo.
- PixMo is streamed from the dataset repo by name.
- Local output paths are used only for experiment checkpoints and deltas.

## Custom Model Code

The repo now includes a real multimodal model type instead of only a training wrapper:

- [src/trinity_vlm/configuration_trinity_vlm.py](/home/carsten/trinity-vlm/src/trinity_vlm/configuration_trinity_vlm.py)
- [src/trinity_vlm/modeling_trinity_vlm.py](/home/carsten/trinity-vlm/src/trinity_vlm/modeling_trinity_vlm.py)

`TrinityVLMForConditionalGeneration` is a custom `PreTrainedModel` that:

- loads Trinity as the language model
- loads Moondream's vision tower
- uses explicit image placeholder tokens in the text stream
- replaces `<|image_pad|>` slots with projected image embeddings
- supports multiple image spans inside a packed sequence
- exposes a standard `forward(images, input_ids, attention_mask, position_ids, labels)` style API

The training-only class in [src/trinity_vlm/graft.py](/home/carsten/trinity-vlm/src/trinity_vlm/graft.py) now subclasses that custom model and only adds freeze policy plus delta-checkpoint helpers.

## What The Model Does

The multimodal path is:

1. Insert Qwen-style vision sentinels into the text stream:
   `<|vision_start|>`, repeated `<|image_pad|>`, then `<|vision_end|>`.
2. Encode each image with Moondream's multi-crop vision tower.
3. Project Moondream's `2048`-dim image tokens into Trinity's `1024`-dim hidden space.
4. Replace the `<|image_pad|>` embedding slots with the image embeddings.
5. Train with loss only on the caption tokens.

The new packed training path also:

- concatenates multiple caption examples into one training sequence
- resets `position_ids` at document boundaries
- builds block-diagonal causal masks so packed examples do not attend across boundaries
- produces both full-attention and sliding-window masks for Trinity's mixed attention pattern

By default the script trains only the bridge. You can also:

- unfreeze the Moondream vision tower
- unfreeze the Trinity LM head
- unfreeze the last `N` Trinity decoder layers
- unfreeze all Trinity weights

## Warm The Cache

Use the HF CLI without `--local-dir` so the models stay in the shared cache:

```bash
hf download arcee-ai/Trinity-Nano-Preview --type model
hf download moondream/moondream3-preview --type model \
  --include '*.py' \
  --include '*.json' \
  --include '*.md' \
  --include 'LICENSE*' \
  --include 'tokenizer*' \
  --include 'model.safetensors.index.json' \
  --include 'modelv2-00001-of-00004.safetensors'
```

The dataset is streamed at training time, so you do not need to cache all PixMo images up front.

## Install

```bash
uv sync
```

## Train

The default training path is now [scripts/train_fsdp2.py](/home/carsten/trinity-vlm/scripts/train_fsdp2.py), driven by a TOML config.

An example config lives at [configs/bridge_fsdp2.toml](/home/carsten/trinity-vlm/configs/bridge_fsdp2.toml). It starts with:

```toml
#!/home/carsten/trinity-vlm/scripts/train_fsdp2.py
```

That means you can make the config executable and run the config itself:

```bash
chmod +x scripts/train_fsdp2.py configs/bridge_fsdp2.toml
./configs/bridge_fsdp2.toml
```

For multi-GPU runs, set `distributed.nproc_per_node` in the TOML. Running the config directly will relaunch through `torch.distributed.run` automatically. You can also launch it explicitly:

```bash
torchrun --standalone --nproc_per_node 4 ./configs/bridge_fsdp2.toml
```

The FSDP2 trainer:

- uses first-fit-decreasing sample packing within a configurable packing buffer
- supports proper multimodal packed batches with multiple image spans per packed row
- wraps Trinity and Moondream with bottom-up `fully_shard(...)`
- registers `encode_images` as an FSDP2 forward method on the vision tower
- registers the Trinity token embedding path as an FSDP2 forward method on the language model
- supports periodic validation loss on a separate streamed dataloader
- supports optional Weights & Biases logging through a `[wandb]` TOML section
- supports shard-range holdouts for PixMo via `data.shard_start` / `data.shard_end`
- saves resumable distributed checkpoints with `torch.distributed.checkpoint`

For bridge-only training where Trinity stays frozen, `activation_checkpointing = false` is the better default if it fits in memory. On this setup it materially improved throughput over checkpointing the frozen LM.

For a real 2-GPU example with validation and offline W&B logging, see [configs/bridge_full_example.toml](/home/carsten/trinity-vlm/configs/bridge_full_example.toml).

The older [main.py](/home/carsten/trinity-vlm/main.py) entrypoint is still present as a simple single-process training loop, but the TOML-driven FSDP2 script is the intended path for real runs.

## Export

To build a local self-contained `TrinityVLM` model directory with the correct custom code files:

```bash
uv run python scripts/export_trinityvlm.py \
  --local-files-only \
  --output-dir checkpoints/trinityvlm-export
```

To export a model with a trained delta applied first:

```bash
uv run python scripts/export_trinityvlm.py \
  --local-files-only \
  --delta-dir checkpoints/bridge-only/step-001000 \
  --output-dir checkpoints/trinityvlm-export-step-001000
```

The exported directory can then be loaded directly with:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/trinityvlm-export",
    trust_remote_code=True,
)
```

## Gradio Test App

To inspect a saved delta checkpoint directly, launch:

```bash
uv run python scripts/gradio_checkpoint_app.py \
  --run-dir checkpoints/bridge-full-train-20260413-seq2048-ckpt-compile \
  --device cpu
```

By default the app resolves `latest_checkpoint.txt`, which currently points at:

```text
checkpoints/bridge-full-train-20260413-seq2048-ckpt-compile/checkpoints/step-002750
```

You can also force a specific checkpoint:

```bash
uv run python scripts/gradio_checkpoint_app.py \
  --run-dir checkpoints/bridge-full-train-20260413-seq2048-ckpt-compile \
  --checkpoint-dir checkpoints/step-002500 \
  --device cpu
```

The app defaults to CPU inference, loads the Trinity base model and Moondream vision tower from the HF cache, applies the local delta checkpoint, and exposes a small image-plus-prompt generation UI.

## Notes

- The FSDP2 trainer checkpoints the distributed model and optimizer state under `output_dir/checkpoints/step-*`.
- The simple legacy trainer still exists, but it does not provide packing or FSDP2.
- Image encoding is still done per image because Moondream's crop count is dynamic.
- If you want a full local multimodal model directory, call `export_pretrained(...)` on `TrinityVLMForConditionalGeneration` or `TrinityMoondreamGraftModel`.
