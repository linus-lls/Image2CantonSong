#!/usr/bin/env python3
"""CLI for image → prompt generation using the demo's multimodal LLM.

This script processes all supported images in an input directory and writes:
- one JSON payload per image in the output directory
- one combined paired JSON file with all generated fields plus image_path

Defaults:
- input: PROJECT_ROOT/Images
- output: DEMO/outputs/MM_outputs
- paired output: IPP/image-prompt-pairs.json

Example usage:
  python cli_image_to_prompt.py
  python cli_image_to_prompt.py --input /path/to/images --output /path/to/output_dir
  python cli_image_to_prompt.py --max-side-length 128 --downscaled-output ./outputs/downscaled_images
  python cli_image_to_prompt.py --paired-output ./Image-Prompt-Pairs/paired.json
"""
from __future__ import annotations
import argparse
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image
from modules.mm_direct_gen import generate_from_image

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from paths import PROJECT_ROOT, DEMO, IMAGES, IPP


def parse_args() -> argparse.Namespace:
    default_input = IMAGES
    default_output = DEMO / "outputs" / "MM_outputs"

    parser = argparse.ArgumentParser(
        description="Generate a lyrics/prompt JSON bundle from all images in a directory using the demo's multimodal LLM."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Directory containing input images. Defaults to $REPO_ROOT/Images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Directory to write output JSON files. Defaults to the demo folder's outputs/MM_outputs.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face multimodal model ID to use.",
    )
    parser.add_argument(
        "--style",
        default="cantopop-ballad",
        help="Optional style preset used only for prompt text generation.",
    )
    parser.add_argument(
        "--line-count",
        type=int,
        choices=[4, 8],
        default=8,
        help="Number of lyric lines to request from the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the multimodal LLM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate from the multimodal model.",
    )
    parser.add_argument(
        "--user-style-hints",
        default="male or female cantopop vocal, emotionally expressive",
        help="Additional style hints to append to the prompt.",
    )
    parser.add_argument(
        "--max-side-length",
        type=int,
        default=0,
        help="If >0, downscale the input image so its longest side is at most this many pixels before inference.",
    )
    parser.add_argument(
        "--downscaled-output",
        type=Path,
        default=None,
        help="Optional directory to save downscaled images before inference.",
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=None,
        help="Optional path to save paired JSON entries for CLIP evaluation. Defaults to IPP/image-prompt-pairs.json.",
    )
    parser.add_argument(
        "--run-on-cpu",
        action="store_true",
        help="Force the multimodal model to run on CPU instead of CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")
    if not args.input.is_dir():
        raise NotADirectoryError(f"--input must be a directory: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.downscaled_output is not None:
        args.downscaled_output.mkdir(parents=True, exist_ok=True)

    paired_entries = []
    paired_output = args.paired_output or (IPP / "image-prompt-pairs.json")
    paired_output.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            path
            for path in args.input.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        ]
    )
    if not image_paths:
        raise ValueError(f"No supported image files found in {args.input}")

    for img_path in image_paths:
        if args.max_side_length > 0:
            with Image.open(img_path) as image:
                image = image.convert("RGB")
                image.thumbnail((args.max_side_length, args.max_side_length), Image.LANCZOS)
                downscaled_dir = args.downscaled_output or args.output
                downscaled_path = downscaled_dir / f"{img_path.stem}_downscaled.jpg"
                image.save(downscaled_path, format="JPEG", quality=95)
                print(f"Saved downscaled image to: {downscaled_path}")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                image_bytes = buffer.getvalue()
        else:
            image_bytes = img_path.read_bytes()

        bundle = generate_from_image(
            image_bytes=image_bytes,
            model_id=args.model_id,
            style=args.style,
            line_count=args.line_count,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            user_style_hints=args.user_style_hints,
            run_on_cpu=args.run_on_cpu,
        )

        output_json = args.output / f"{img_path.stem}.json"
        payload = asdict(bundle)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        paired_entry = payload.copy()
        paired_entry["image_path"] = str(img_path.relative_to(IMAGES))
        paired_entries.append(paired_entry)

        print(f"Wrote image→prompt bundle to: {output_json}")

    with paired_output.open("w", encoding="utf-8") as f:
        json.dump(paired_entries, f, ensure_ascii=False, indent=2)
    print(f"Wrote paired JSON to: {paired_output}")


if __name__ == "__main__":
    main()
