#!/usr/bin/env python3
"""Evaluate image-text alignment using CLIP.

This script computes cosine similarity between image and text embeddings from a CLIP model.
It supports paired evaluation from a CSV or JSONL file, where each row/object contains
an image path and text caption.
"""

import argparse
import csv
import io
import sys
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from paths import PROJECT_ROOT, EVAL, IMAGES, IPP

DEFAULT_MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-text alignment using Chinese CLIP embeddings.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(IMAGES),
        help="Directory containing image files referenced in the paired file. Defaults to IMAGES.")
    parser.add_argument(
        "--paired-file",
        type=str,
        default=str(IPP / "image-prompt-pairs.json"),
        help="CSV or JSONL file containing paired image_path and text entries.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face Chinese CLIP model name or path.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on (cpu or cuda).")
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(EVAL / "image_lyrics_alignment" / "outputs" / "clip_similarity.json"),
        help="Optional JSON/JSONL file to save input rows with appended clip_similarity. Defaults to EVAL/image_lyrics_alignment/outputs/clip_similarity.jsonl.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings before computing cosine similarity.")
    return parser.parse_args()


def load_pairs(paired_file: Path) -> List[Dict[str, str]]:
    ext = paired_file.suffix.lower()
    if ext == ".csv":
        return load_pairs_from_csv(paired_file)
    if ext in {".jsonl", ".json"}:
        return load_pairs_from_jsonl(paired_file)
    raise ValueError("Unsupported paired-file format: use CSV or JSONL.")


def load_pairs_from_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        pairs = []
        for row in reader:
            pairs.append({k: v for k, v in row.items() if v is not None})
    return pairs


def load_pairs_from_jsonl(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        pairs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
        return pairs

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("JSON array must contain only objects")
        return payload

    raise ValueError(
        "Unsupported JSON format for paired file: expected object, array, or JSONL lines."
    )


def standardize_pair(pair: Dict[str, str], index: int) -> Dict[str, str]:
    image_key = next(
        (k for k in pair if k.lower() in {"image_path", "image", "filename", "file"}), None)
    text_key = next(
        (k for k in pair if k.lower() in {"text", "caption", "sentence", "description", "lyrics_text", "lyrics"}), None)
    if image_key is None or text_key is None:
        raise ValueError(
            f"Paired file row {index} is missing an image_path or text field: got {list(pair.keys())}")
    return {"image_path": pair[image_key], "text": pair[text_key]}


def load_text_source(text_source: str, base_dir: Path) -> str:
    return text_source


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def resolve_text_from_payload(payload: Union[str, Dict[str, str]]) -> str:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("json_input must be a JSON string or dict")
    text_key = next(
        (k for k in payload if k.lower() in {"text", "caption", "sentence", "description", "lyrics_text", "lyrics"}),
        None,
    )
    if text_key is None:
        raise ValueError(f"JSON payload is missing a text field; got {list(payload.keys())}")
    return payload[text_key]


def load_chinese_clip(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[ChineseCLIPProcessor, ChineseCLIPModel, torch.device]:
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    model = ChineseCLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


def score_image_text_similarity(
    image_bytes: bytes,
    json_input: Union[str, Dict[str, str]],
    processor: Optional[ChineseCLIPProcessor] = None,
    model: Optional[ChineseCLIPModel] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None,
    normalize: bool = True,
) -> float:
    image = load_image_from_bytes(image_bytes)
    text = resolve_text_from_payload(json_input)

    if processor is None and model is None:
        processor, model, device = load_chinese_clip(model_name=model_name, device=device)
    elif processor is None:
        processor = ChineseCLIPProcessor.from_pretrained(model_name)
        device = torch.device(device if device else next(model.parameters()).device)
    elif model is None:
        _, model, device = load_chinese_clip(model_name=model_name, device=device)
    else:
        device = torch.device(device if device else next(model.parameters()).device)

    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = processor(text=[text], padding=True, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    similarity = compute_similarity(
        images=[image],
        texts=[text],
        processor=processor,
        model=model,
        device=device,
        normalize=normalize,
    )
    return float(similarity.squeeze().cpu().item())


def compute_similarity(
    images: List[Image.Image],
    texts: List[str],
    processor: ChineseCLIPProcessor,
    model: ChineseCLIPModel,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    image_inputs = processor(images=images, return_tensors="pt")
    text_inputs = processor(text=texts, padding=True, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)
        text_embeds = model.get_text_features(**text_inputs)

    if normalize:
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    return torch.nn.functional.cosine_similarity(image_embeds, text_embeds, dim=-1)


def compute_pair_similarity_scores(
    pairs: List[Dict[str, str]],
    processor: ChineseCLIPProcessor,
    model: ChineseCLIPModel,
    device: torch.device,
    batch_size: int,
    normalize: bool,
) -> List[Dict[str, object]]:
    results = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        images = [load_image(pair["image_path"]) for pair in batch]
        texts = [pair["text"] for pair in batch]

        sims = compute_similarity(
            images=images,
            texts=texts,
            processor=processor,
            model=model,
            device=device,
            normalize=normalize,
        )

        for pair, sim in zip(batch, sims.cpu().tolist()):
            results.append(
                {
                    "image_path": pair["image_path"],
                    "text": pair["text"],
                    "similarity": float(sim),
                }
            )
    return results


def summarize(results: List[Dict[str, object]]) -> Dict[str, float]:
    sims = [item["similarity"] for item in results]
    if not sims:
        return {}
    sims_tensor = torch.tensor(sims)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    counts = {f">= {th}": int((sims_tensor >= th).sum().item()) for th in thresholds}
    return {
        "num_pairs": len(sims),
        "mean_similarity": float(sims_tensor.mean().item()),
        "median_similarity": float(sims_tensor.median().item()),
        **counts,
    }


def write_output_json(
    input_pairs: List[Dict[str, str]],
    results: List[Dict[str, object]],
    output_path: Path,
) -> None:
    output_data = []
    for raw, result in zip(input_pairs, results):
        item = raw.copy()
        item["clip_similarity"] = result["similarity"]
        output_data.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8", newline="") as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    paired_file = Path(args.paired_file)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    raw_pairs = load_pairs(paired_file)
    paired_base_dir = paired_file.parent
    pairs = []
    for idx, raw in enumerate(raw_pairs, start=1):
        standardized = standardize_pair(raw, idx)
        image_path = Path(standardized["image_path"])
        if not image_path.is_absolute():
            image_path = image_dir / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        text_value = load_text_source(standardized["text"], paired_base_dir)
        pairs.append({"image_path": image_path, "text": text_value})

    processor = ChineseCLIPProcessor.from_pretrained(args.model_name)
    model = ChineseCLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()

    results = compute_pair_similarity_scores(
        pairs=pairs,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )

    summary = summarize(results)
    print("CLIP image-text alignment results:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.output_file:
        write_output_json(raw_pairs, results, Path(args.output_file))
        print(f"Saved paired JSON with clip_similarity to: {args.output_file}")


if __name__ == "__main__":
    main()
