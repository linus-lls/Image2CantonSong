# Genre Source Evaluation

This folder contains an evaluator for checking whether generated genre tags are sourced from an allowed genre list.

The evaluator focuses on **list membership only**. It does not judge whether the genre is musically appropriate, creative, or consistent with the lyrics/image.

## Script

```text
genre_source_eval.py
```

## Reference Tag List

The evaluator reads a JSON file containing a `genre` list, for example:

```json
{
  "genre": [
    "Pop",
    "rock",
    "electronic",
    "Classical"
  ]
}
```

The default expected tag list file is:

```text
top_200_tags.json
```

You can also pass a custom file using:

```bash
--tag-list-file /path/to/top_200_tags.json
```

## Input Format

The script accepts:

- CSV
- JSON
- JSONL
- TXT

Supported genre fields include:

- `genre`
- `genres`
- `genre_text`
- `genre_tags`
- `music_genre`
- `music_genres`

If no direct genre field is found, the evaluator can extract genre content from prompt-like fields such as:

- `style`
- `prompt`
- `music_prompt`
- `description`
- `text`
- `caption`

For prompt-like fields, the text should contain an explicit genre header, such as:

```text
genre: Pop, rock
```

or:

```text
Genre = electronic; Mood = happy; Instrument = guitar
```

## Example CSV

```csv
id,genre
1,"Pop, rock"
2,"electronic, unknown-style"
```

## Example JSONL

```jsonl
{"genre": ["Pop", "rock"]}
{"prompt": "genre: electronic, jazz; mood: happy; instrument: guitar"}
```

## Usage

```bash
python Evaluation/genre_source_evaluation/genre_source_eval.py \
  --input-file IPP/generated_genres.json \
  --tag-list-file top_200_tags.json \
  --output-file EVAL/genre_source_evaluation/outputs/genre_source_scores.json
```

## Specify the Genre Field

```bash
python Evaluation/genre_source_evaluation/genre_source_eval.py \
  --input-file IPP/generated_genres.json \
  --genre-field genre
```

## Case-sensitive Matching

By default, matching is case-insensitive.

For example, `Pop`, `pop`, and `POP` are treated as the same normalized tag.

To enable strict case-sensitive matching:

```bash
python Evaluation/genre_source_evaluation/genre_source_eval.py \
  --input-file IPP/generated_genres.json \
  --case-sensitive
```

## Fuzzy Matching

By default, fuzzy matching is disabled.

This means a genre is considered valid only if it matches the allowed list after normalization.

To allow high-similarity fuzzy matches:

```bash
python Evaluation/genre_source_evaluation/genre_source_eval.py \
  --input-file IPP/generated_genres.json \
  --allow-fuzzy \
  --fuzzy-cutoff 0.88
```

## Python API

```python
from Evaluation.genre_source_evaluation.genre_source_eval import (
    evaluate_genre_source,
)

payload = {
    "genre": "Pop, rock, unknown-style"
}

result = evaluate_genre_source(
    json_input=payload,
    tag_list_file="top_200_tags.json",
    return_details=True,
)

print(result["genre_source_score"])
print(result["all_genres_from_list"])
print(result["valid_genres"])
print(result["invalid_genres"])
print(result["suggestions"])
```

## Output

The script prints summary metrics including:

- `num_items`
- `mean_score`
- `median_score`
- `min_score`
- `max_score`
- `num_all_genres_from_list`
- `ratio_all_genres_from_list`
- `count_above_100`
- `count_above_80`
- `count_above_50`

If `--output-file` is provided, the script saves the original records with additional fields:

```text
genre_source_score
genre_source_grade
all_genres_from_list
valid_genres
invalid_genres
matched_canonical_genres
genre_source_suggestions
genre_source_warnings
```

## Score Logic

The score is calculated as:

```text
genre_source_score = valid_genre_count / total_genre_count * 100
```

Example:

```text
Input genres:
Pop, rock, unknown-style

Allowed:
Pop, rock

Result:
valid_genres = ["Pop", "rock"]
invalid_genres = ["unknown-style"]
genre_source_score = 66.67
```

## Grades

| Score | Grade |
|---:|---|
| 100 | pass |
| 80 - 99 | mostly_pass |
| 50 - 79 | partial |
| 0 - 49 | fail |

## Notes

This evaluator is useful for checking whether generated music style metadata remains within a predefined genre vocabulary.

It is suitable for pipelines where genre tags must come from a fixed list before being passed to downstream music generation or retrieval modules.
