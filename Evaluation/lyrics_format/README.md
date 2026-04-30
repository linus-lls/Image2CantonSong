# Lyrics Format Evaluation

This folder contains a hybrid evaluator for checking whether generated lyrics follow the expected song lyric format.

The evaluator focuses on **format only**. It does **not** evaluate lyric meaning, creativity, rhyme, emotion, or literary quality.

The expected format is based on section tags and blank-line structure, such as:

```text
[verse]
line
line
line

[chorus]
line
line
line

[bridge]
line
line

[outro]
line
line

[end]
```

## Evaluation Logic

The final score combines three components:

```text
final_score =
    rule_weight * rule_format_score
  + transformer_weight * transformer_format_similarity_score
  + sequence_weight * sequence_structure_score
```

Default weights:

```text
rule_weight = 0.50
transformer_weight = 0.30
sequence_weight = 0.20
```

### 1. Rule-based Format Score

This component checks strict character-level formatting rules, including:

- Section tags must appear alone on their own line.
- Supported tags include `[verse]`, `[chorus]`, `[bridge]`, `[outro]`, and `[end]`.
- The first non-empty line should be a section tag.
- Different sections should be separated by exactly one blank line.
- There should be no blank line immediately after a section tag.
- There should be no blank line inside a section.
- The final non-empty line should be `[end]`.
- There should be no lyric content after `[end]`.

Example of correct format:

```text
[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light

[end]
```

Example of incorrect format:

```text
[verse]

Staring at the sunset, colors paint the sky
```

The above is incorrect because there is an unnecessary blank line immediately after `[verse]`.

Another incorrect example:

```text
[verse]
Staring at the sunset, colors paint the sky
[chorus]
Every road you take, I'll be one step behind
```

The above is incorrect because there should be a blank line before `[chorus]`.

### 2. Transformer Format Similarity Score

This component uses a Transformer model to compare the generated lyric format with a reference format.

To avoid judging lyric content, the evaluator first converts the lyrics into a format-only signature.

For example:

```text
[verse]
hello
world

[chorus]
sing again

[end]
```

is converted into:

```text
TAG_VERSE
LYRIC_LINE
LYRIC_LINE
BLANK
TAG_CHORUS
LYRIC_LINE
BLANK
TAG_END
```

The Transformer model compares this format signature with the reference format signature.

Therefore, this score measures structural similarity instead of lyric meaning.

### 3. Sequence Structure Score

This component compares the section order and line-count pattern.

For example, a lyric may be represented as:

```text
TAG_VERSE:4 | TAG_CHORUS:6 | TAG_VERSE:4 | TAG_CHORUS:6 | TAG_BRIDGE:4 | TAG_OUTRO:4 | TAG_END
```

This compact structure is compared with the reference structure using sequence similarity.

## Script

```text
lyrics_format_transformer_score.py
```

## Requirements

The script requires:

- Python
- `torch`
- `transformers`

The existing project environment should already include `torch` and `transformers`.

Default Transformer model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

You can replace it with another Hugging Face model name or a local model path.

## Input Format

The script accepts lyrics from:

- CSV
- JSON
- JSONL
- TXT

For CSV, JSON, and JSONL files, the text field can use one of the following names:

- `lyrics_text`
- `lyrics`
- `text`
- `song_lyrics`
- `generated_lyrics`

Example CSV:

```csv
id,lyrics_text
1,"[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light

[end]"
```

Example JSONL:

```jsonl
{"lyrics_text": "[verse]\nStaring at the sunset, colors paint the sky\nThoughts of you keep swirling, can't deny\n\n[chorus]\nEvery road you take, I'll be one step behind\nEvery dream you chase, I'm reaching for the light\n\n[end]"}
```

Example TXT:

```text
[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light

[end]
```

## Usage

### Use the default reference format

```bash
python Evaluation/lyrics_format_evaluation/lyrics_format_transformer_score.py \
  --input-file IPP/generated_lyrics.json \
  --output-file EVAL/lyrics_format_evaluation/outputs/lyrics_format_transformer_scores.json
```

### Use a custom reference format

If you have a reference format file:

```text
Evaluation/lyrics_format_evaluation/reference_lyrics_format.txt
```

run:

```bash
python Evaluation/lyrics_format_evaluation/lyrics_format_transformer_score.py \
  --input-file IPP/generated_lyrics.json \
  --reference-file Evaluation/lyrics_format_evaluation/reference_lyrics_format.txt \
  --output-file EVAL/lyrics_format_evaluation/outputs/lyrics_format_transformer_scores.json
```

### Specify the lyrics text field

```bash
python Evaluation/lyrics_format_evaluation/lyrics_format_transformer_score.py \
  --input-file IPP/generated_lyrics.json \
  --text-field lyrics_text
```

### Change the Transformer model

```bash
python Evaluation/lyrics_format_evaluation/lyrics_format_transformer_score.py \
  --input-file IPP/generated_lyrics.json \
  --model-name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Use CPU

```bash
python Evaluation/lyrics_format_evaluation/lyrics_format_transformer_score.py \
  --input-file IPP/generated_lyrics.json \
  --device cpu
```

### Use CUDA

```bash
python Evaluation/lyrics_format_evaluation/lyrics_format_transformer_score.py \
  --input-file IPP/generated_lyrics.json \
  --device cuda
```

### Adjust score weights

For example, to make strict formatting more important:

```bash
python Evaluation/lyrics_format_evaluation/lyrics_format_transformer_score.py \
  --input-file IPP/generated_lyrics.json \
  --rule-weight 0.60 \
  --transformer-weight 0.25 \
  --sequence-weight 0.15
```

## Python API

The script exposes a reusable Python function:

```python
from Evaluation.lyrics_format_evaluation.lyrics_format_transformer_score import (
    score_lyrics_format_hybrid,
)

payload = {
    "lyrics_text": """
[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light

[end]
"""
}

result = score_lyrics_format_hybrid(
    json_input=payload,
    return_details=True,
)

print("score:", result["lyrics_format_score"])
print("grade:", result["grade"])
print("components:", result["components"])
print("warnings:", result["warnings"])
```

## Output

The script prints summary metrics, including:

- `num_items`
- `mean_score`
- `median_score`
- `min_score`
- `max_score`
- `count_above_90`
- `count_above_75`
- `count_above_60`

If `--output-file` is provided, the script saves the original input records with additional fields:

```text
lyrics_format_score
lyrics_format_grade
lyrics_format_components
lyrics_format_metrics
lyrics_format_warnings
```

Example output item:

```json
{
  "lyrics_text": "[verse]\nStaring at the sunset...\n\n[chorus]\nEvery road you take...\n\n[end]",
  "lyrics_format_score": 88.42,
  "lyrics_format_grade": "good",
  "lyrics_format_components": {
    "rule_format_score": 95.0,
    "transformer_format_similarity_score": 84.3,
    "sequence_structure_score": 76.5,
    "rule_weight": 0.5,
    "transformer_weight": 0.3,
    "sequence_weight": 0.2
  },
  "lyrics_format_metrics": {
    "num_lines": 9,
    "num_tags": 3,
    "num_blank_lines": 2,
    "num_lyric_lines": 4,
    "tags": ["verse", "chorus", "end"],
    "has_verse": true,
    "has_chorus": true,
    "has_bridge": false,
    "has_outro": false,
    "has_end": true,
    "compact_structure_signature": "TAG_VERSE:2 | TAG_CHORUS:2 | TAG_END"
  },
  "lyrics_format_warnings": []
}
```

## Score Grades

| Score Range | Grade |
|---|---|
| 90 - 100 | excellent |
| 75 - 89 | good |
| 60 - 74 | acceptable |
| 40 - 59 | weak |
| 0 - 39 | poor |

## Notes

This evaluator is intended for checking whether generated lyrics are suitable for downstream song generation models.

It is especially useful when the generation model expects clearly separated lyric sections, such as:

```text
[verse]

[chorus]

[bridge]

[outro]

[end]
```

The evaluator does not judge whether the lyrics are beautiful, emotional, or musically strong. It only checks whether the lyrics follow the expected structural format.
