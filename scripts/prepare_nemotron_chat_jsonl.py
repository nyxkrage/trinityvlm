#!/home/carsten/trinity-vlm/.venv/bin/python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize Nemotron chat examples into a local JSONL file.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--dataset", default="nvidia/Llama-Nemotron-Post-Training-Dataset")
    parser.add_argument("--dataset-config-name", default="SFT")
    parser.add_argument("--split", default="chat")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset,
        args.dataset_config_name,
        split=args.split,
        streaming=True,
    )

    with output_path.open("w") as handle:
        for index, example in enumerate(dataset):
            if index >= args.limit:
                break
            record = {
                "input_messages": example["input"],
                "assistant_output": example["output"],
                "system_prompt": example.get("system_prompt"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
