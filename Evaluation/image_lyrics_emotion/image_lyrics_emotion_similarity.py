#!/usr/bin/env python3
"""Evaluate emotion similarity between an image and lyrics text.

This script compares image emotion predictions from CLIP-E with text emotion
predictions from a selectable text emotion model. It embeds each emotion label
with a sentence-transformers model, builds weighted emotion vectors, and
computes cosine similarity between the image and lyrics emotion representations.

Example:
    python Evaluation/image_lyrics_emotion/image_lyrics_emotion_similarity.py \
        --image-path Images/caption.jpg \
        --text "A calm lakeside scene with falling leaves" \
        --text-emotion-model johnson \
        --embedding-model-name sentence-transformers/all-MiniLM-L6-v2

To use full prediction sets rather than top-k, omit `--top-k-image` and
`--top-k-text`.
"""

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from paths import PROJECT_ROOT


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_image_emotion_predictor_script(verbose: bool = True) -> Path:
    clip_e_path = PROJECT_ROOT / "Emotion" / "CLIP-E" / "clip-e-ce.py"
    if not clip_e_path.exists():
        raise FileNotFoundError(f"Expected CLIP-E script at: {clip_e_path}")
    return clip_e_path


TEXT_EMOTION_MODEL_DEFS = [
    {
        "MODEL_ID": "zai-org/GLM-5",
        "text_emotion_model": "llm-online",
        "is_online": True,
    },
    {
        "MODEL_ID": "Johnson8187/Chinese-Emotion",
        "text_emotion_model": "johnson",
        "is_online": False,
    },
    {
        "MODEL_ID": "SchuylerH/bert-multilingual-go-emtions",
        "text_emotion_model": "go-emotion",
        "is_online": False,
    },
]


def get_text_emotion_model_def(model_name: str) -> dict:
    for model_def in TEXT_EMOTION_MODEL_DEFS:
        if model_def["text_emotion_model"] == model_name:
            return model_def
    raise ValueError(f"Unknown text emotion model: {model_name}")


def get_text_emotion_model_keys() -> List[str]:
    return [model_def["text_emotion_model"] for model_def in TEXT_EMOTION_MODEL_DEFS]


def get_text_emotion_model_display_name(model_name: str) -> str:
    model_def = get_text_emotion_model_def(model_name)
    model_id = model_def.get("MODEL_ID", "")
    if not model_id:
        return model_name
    suffix = "online" if model_def.get("is_online", False) else "local"
    return f"{model_id} ({suffix})"


def _resolve_text_emotion_model_definition(model_name: str):
    model_def = get_text_emotion_model_def(model_name)
    if model_name == "go-emotion":
        model_path = PROJECT_ROOT / "Emotion" / "Text2Emotion" / "bert-go-emotion.py"
        module_name = "bert_go_emotion"
        predictor_class = "BertGoEmotion"
    elif model_name == "johnson":
        model_path = PROJECT_ROOT / "Emotion" / "Text2Emotion" / "johnson_chinese_emotion.py.py"
        module_name = "johnson_chinese_emotion"
        predictor_class = "JohnsonChineseEmotion"
    elif model_name == "llm-online":
        model_path = PROJECT_ROOT / "Emotion" / "Text2Emotion" / "llm_text_emotion.py"
        module_name = "llm_text_emotion"
        predictor_class = "HuggingFaceLLMTextEmotion"
    else:
        raise ValueError(f"Unknown text emotion model: {model_name}")
    return model_path, module_name, predictor_class


def load_text_emotion_predictor(model_name: str = "go-emotion", verbose: bool = True):
    model_path, module_name, predictor_class = _resolve_text_emotion_model_definition(model_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Expected text emotion model script at: {model_path}")

    predictor_module = load_module_from_path(model_path, module_name)
    return getattr(predictor_module, predictor_class)(verbose=verbose)


def get_text_emotion_predictor_class(model_name: str):
    model_path, module_name, predictor_class = _resolve_text_emotion_model_definition(model_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Expected text emotion model script at: {model_path}")

    predictor_module = load_module_from_path(model_path, module_name)
    return getattr(predictor_module, predictor_class)


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def load_text(text: Optional[str], text_file: Optional[Path]) -> str:
    if text_file is not None:
        return text_file.read_text(encoding="utf-8").strip()
    if text is not None:
        return text
    raise ValueError("Either text or text_file must be provided.")


def create_label_embedder(
    model_name: str,
    device: Optional[str] = None,
):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for emotion label embeddings. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(model_name, device=device_name)


def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )


def embed_labels(
    embedder,
    labels: Sequence[str],
    device: Optional[str] = None,
) -> List[np.ndarray]:
    if hasattr(embedder, "encode"):
        embeddings = embedder.encode(
            list(labels),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [np.asarray(e, dtype=np.float32) for e in embeddings]

    raise RuntimeError("Unsupported embedder type; expected SentenceTransformer instance.")


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12)


def build_weighted_vector(
    predictions: Iterable[Dict[str, Union[str, float]]],
    label_embeddings: Dict[str, np.ndarray],
) -> np.ndarray:
    embedding_size = next(iter(label_embeddings.values())).shape[0]
    vector = np.zeros(embedding_size, dtype=np.float32)
    total_weight = 0.0

    # Use only the model label key for similarity; english_label is ignored here.
    for item in predictions:
        label = str(item["label"])
        score = float(item["score"])
        if label not in label_embeddings:
            continue
        vector += label_embeddings[label] * score
        total_weight += score

    if total_weight > 0:
        vector /= total_weight
    return vector


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_vector(a)
    b = normalize_vector(b)
    return float(np.dot(a, b))


def _parse_clip_e_output(output: str) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        label, score_text = line.split(":", 1)
        results.append({"label": label.strip(), "score": float(score_text.strip())})
    return results


def predict_image_emotions(
    clip_e_script: Path,
    image: Image.Image,
    top_k: Optional[int],
    model_type: str,
) -> List[Dict[str, float]]:
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    binary_image = image_bytes.getvalue()

    script_path = shlex.quote(str(clip_e_script))
    cmd = f"conda activate clip-e && python {script_path} --stdin-bytes --model-type {model_type}"
    if top_k is not None:
        cmd += f" --top-n {top_k}"
    cmd += " && conda deactivate"

    proc = subprocess.run(
        ["bash", "-lic", cmd],
        input=binary_image,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Clip-E subprocess failed:\n"
            + proc.stderr.decode("utf-8", errors="ignore")
        )

    output = proc.stdout.decode("utf-8", errors="ignore")
    return _parse_clip_e_output(output)


def predict_text_emotions(
    predictor,
    text: str,
    top_k: Optional[int],
) -> List[Dict[str, float]]:
    return predictor.predict_top_n(text, n=top_k)


def get_max_image_emotion_classes(model_type: str) -> int:
    if model_type == "25cat":
        return 25
    if model_type == "6cat":
        return 6
    if model_type == "binary":
        return 2
    raise ValueError(f"Unknown image model type: {model_type}")


def get_max_text_emotion_classes(model_name: str) -> int:
    predictor_class = get_text_emotion_predictor_class(model_name)
    if not hasattr(predictor_class, "MAX_EMOTION_CLASSES"):
        raise AttributeError(
            f"Predictor class {predictor_class.__name__} must define MAX_EMOTION_CLASSES."
        )
    return int(getattr(predictor_class, "MAX_EMOTION_CLASSES"))


def evaluate_emotion_similarity(
    image: Image.Image,
    lyrics_text: str,
    top_k_image: Optional[int],
    top_k_text: Optional[int],
    image_model_type: str,
    text_emotion_model: str,
    embedding_model_name: str,
    embedding_device: Optional[str],
    verbose: bool = True,
) -> Dict[str, object]:
    clip_e_script = load_image_emotion_predictor_script(verbose=verbose)
    text_predictor = load_text_emotion_predictor(text_emotion_model, verbose=verbose)

    image_predictions = predict_image_emotions(
        clip_e_script,
        image=image,
        top_k=top_k_image,
        model_type=image_model_type,
    )
    text_predictions = predict_text_emotions(text_predictor, lyrics_text, top_k=top_k_text)

    # Take union and build a unified label set and embed all labels
    label_set = {item["label"] for item in image_predictions} | {
        item["label"] for item in text_predictions
    }
    embedder = create_label_embedder(embedding_model_name, device=embedding_device)
    embeddings = embed_labels(embedder, sorted(label_set), device=embedding_device)
    label_embeddings = dict(zip(sorted(label_set), embeddings))

    image_vector = build_weighted_vector(image_predictions, label_embeddings)
    text_vector = build_weighted_vector(text_predictions, label_embeddings)
    similarity = compute_cosine_similarity(image_vector, text_vector)

    return {
        "similarity": similarity,
        "image_predictions": image_predictions,
        "text_predictions": text_predictions,
        "image_model_type": image_model_type,
        "embedding_model_name": embedding_model_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute emotion similarity between an image and lyrics text."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
        help="Path to the image file to evaluate.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Lyrics text to compare against the image.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Optional lyrics file path. If provided, it overrides --text.",
    )
    parser.add_argument(
        "--top-k-image",
        type=int,
        default=None,
        help="Number of top image emotion labels to use. Omit for full predictions.",
    )
    parser.add_argument(
        "--top-k-text",
        type=int,
        default=None,
        help="Number of top text emotion labels to use. Omit for full predictions.",
    )
    parser.add_argument(
        "--image-model-type",
        choices=["25cat", "6cat", "binary"],
        default="25cat",
        help="CLIP-E emotion model type for image prediction.",
    )
    parser.add_argument(
        "--text-emotion-model",
        choices=["llm-online", "johnson", "go-emotion"],
        default="johnson",
        help="Text emotion model to use for lyrics emotion prediction.",
    )
    # By experience, "johnson" appears to be more accurate, although it only
    #   gives 8 emotion categories.
    # "go-emotion" gives 28 categories but may be less precise

    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Sentence embedding model name or path.",
    )
    # Other choices: sentence-transformers/all-mpnet-base-v2,
    #   sentence-transformers/paraphrase-multilingual-mpnet-base-v2,
    #   sentence-transformers/all-MiniLM-L6-v2
    # Chinese grounding is better supported by paraphrase-multilingual-mpnet-base-v2.
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for embedding model (cpu or cuda).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional JSON file to save the similarity result.",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose loading messages.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.text_file is None and args.text is None:
        raise ValueError("Provide either --text or --text-file.")
    lyrics = load_text(args.text, args.text_file)
    image = load_image(args.image_path)

    results = evaluate_emotion_similarity(
        image=image,
        lyrics_text=lyrics,
        top_k_image=args.top_k_image,
        top_k_text=args.top_k_text,
        image_model_type=args.image_model_type,
        text_emotion_model=args.text_emotion_model,
        embedding_model_name=args.embedding_model_name,
        embedding_device=args.device,
        verbose=not args.no_verbose,
    )

    print("Image emotion predictions:")
    for item in results["image_predictions"]:
        print(f"  {item['label']}: {item['score']:.4f}")

    print("\nText emotion predictions:")
    for item in results["text_predictions"]:
        print(f"  {item['label']}: {item['score']:.4f}")

    print(f"\nEmotion similarity: {results['similarity']:.4f}")

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
