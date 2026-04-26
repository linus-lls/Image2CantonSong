# CLIP Image-Text Alignment Evaluation

This folder contains a simple CLIP-based evaluator that computes cosine similarity between paired images and text captions.

## Requirements

- Python with `torch`, `transformers`, and `Pillow`
- The existing project environment already includes `transformers` and `torch`.

## Script

- `clip_image_text_alignment.py`

## Input Format

The script accepts a paired file in either:

- CSV with header fields including `image_path` and `text` (or synonyms such as `image`, `filename`, `caption`, `description`)
- JSONL where each object contains corresponding `image_path` and `text` fields

The `text` field should contain the text directly.

The image paths may be absolute or relative to `--image-dir`.

Example CSV:

```csv
image_path,text
img001.jpg,An illustration of a person singing Cantonese lyrics.
img002.jpg,A microphone and stage lights.
```

Example JSONL:

```jsonl
{"image_path": "img001.jpg", "text": "An illustration of a person singing Cantonese lyrics."}
{"image_path": "img002.jpg", "text": "A microphone and stage lights."}
```

## Usage

```bash
python Evaluation/image_lyrics_alignment/clip_image_text_alignment.py \
  --paired-file /path/to/paired.csv
```

The script defaults to `--image-dir` = `IMAGES`, `--paired-file` = `IPP/image_prompt_pairs.json` and `--output-file` = `EVAL/image_lyrics_alignment/outputs/clip_similarity.json`.

Optional arguments:

- `--model-name`: Chinese CLIP model name or path (default: `OFA-Sys/chinese-clip-vit-base-patch16`)
- `--batch-size`: Batch size for embedding computation (default: 32)
- `--device`: Device to run on (`cpu` or `cuda`)
- `--normalize`: Normalize image/text embeddings before similarity

## Python API

The script exposes a reusable Python function via import:

```python
from Evaluation.image_lyrics_alignment.clip_image_text_alignment import (
    score_image_text_similarity,
)

with open("/path/to/image.jpg", "rb") as f:
    image_bytes = f.read()

payload = {
    "lyrics_text": "你的歌詞放這裡..."
}

score = score_image_text_similarity(
    image_bytes=image_bytes,
    json_input=payload,
)
print("similarity:", score)
```

The `json_input` can be either a JSON string or a Python dict. The payload should contain text directly.

## Output

The script prints summary metrics including:

- `num_pairs`
- `mean_similarity`
- `median_similarity`
- counts above threshold values

If `--output-file` is provided, the script saves the original input objects with an added `clip_similarity` field.
