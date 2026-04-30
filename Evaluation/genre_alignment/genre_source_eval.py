#!/usr/bin/env python3
"""Evaluate whether generated genre tags come from an allowed genre list.

This script checks whether the genre content in CSV / JSON / JSONL / TXT inputs
is sourced from the `genre` list inside a tag-list JSON file, such as
`top_200_tags.json`.

The evaluation focuses on list membership only. It does not judge music quality,
genre appropriateness, or semantic creativity.

Supported input fields include:
- genre
- genres
- genre_text
- genre_tags
- music_genre
- style
- prompt
- description

If a structured genre field is available, it is used directly.
If only a prompt-like text field is available, the script tries to extract the
genre part from patterns such as:
- genre: Pop, rock
- genres: Pop | rock
- Genre = Pop; Mood = happy
"""

import argparse
import csv
import json
import re
import statistics
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

try:
    from paths import EVAL, IPP
except Exception:
    EVAL = repo_root / "EVAL"
    IPP = repo_root / "IPP"


GENRE_FIELD_CANDIDATES = [
    "genre",
    "genres",
    "genre_text",
    "genre_tags",
    "music_genre",
    "music_genres",
]

PROMPT_FIELD_CANDIDATES = [
    "style",
    "prompt",
    "music_prompt",
    "description",
    "text",
    "caption",
]

DEFAULT_TAG_LIST_FILE = repo_root / "top_200_tags.json"

GENRE_HEADER_PATTERN = re.compile(
    r"(?:^|[\n;,|])\s*(genre|genres|music\s*genre|music\s*genres)\s*[:=]\s*",
    flags=re.IGNORECASE,
)

NEXT_FIELD_PATTERN = re.compile(
    r"\s*(?:mood|instrument|instruments|gender|timbre|tempo|lyrics|lyric|vocal|voice)\s*[:=]",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether genre tags are sourced from an allowed genre list."
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=str(IPP / "generated_genres.json"),
        help="Input CSV, JSON, JSONL, or TXT file containing genre content.",
    )
    parser.add_argument(
        "--tag-list-file",
        type=str,
        default=str(DEFAULT_TAG_LIST_FILE),
        help="JSON file containing a `genre` list, e.g. top_200_tags.json.",
    )
    parser.add_argument(
        "--genre-field",
        type=str,
        default=None,
        help="Optional explicit field name containing genre content.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(EVAL / "genre_source_evaluation" / "outputs" / "genre_source_scores.json"),
        help="Output JSON or JSONL file.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use case-sensitive matching. Default is case-insensitive.",
    )
    parser.add_argument(
        "--allow-fuzzy",
        action="store_true",
        help="Allow fuzzy matched tags to count as valid when similarity is high enough.",
    )
    parser.add_argument(
        "--fuzzy-cutoff",
        type=float,
        default=0.88,
        help="Fuzzy matching cutoff between 0 and 1. Default: 0.88.",
    )

    return parser.parse_args()


def normalize_tag(tag: str, case_sensitive: bool = False) -> str:
    """Normalize a tag for robust matching.

    This normalization handles common formatting variations:
    - leading / trailing spaces
    - repeated whitespace
    - underscores
    - hyphen spacing
    - optional case folding
    """

    tag = str(tag).strip()
    tag = tag.replace("_", " ")
    tag = re.sub(r"\s+", " ", tag)
    tag = re.sub(r"\s*-\s*", "-", tag)

    if not case_sensitive:
        tag = tag.lower()

    return tag


def load_allowed_genres(
    tag_list_file: Union[str, Path],
    case_sensitive: bool = False,
) -> Tuple[Set[str], Dict[str, str], List[str]]:
    """Load and normalize allowed genres from a JSON file.

    Returns
    -------
    allowed_normalized:
        Set of normalized genre names.
    canonical_map:
        Mapping from normalized genre name to one canonical original spelling.
    raw_genres:
        Original genre list from the JSON file.
    """

    tag_list_file = Path(tag_list_file)

    with tag_list_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or "genre" not in payload:
        raise ValueError("Tag list JSON must be an object containing a `genre` list.")

    raw_genres = payload["genre"]

    if not isinstance(raw_genres, list):
        raise ValueError("The `genre` field in tag list JSON must be a list.")

    allowed_normalized = set()
    canonical_map = {}

    for item in raw_genres:
        norm = normalize_tag(str(item), case_sensitive=case_sensitive)
        allowed_normalized.add(norm)

        # Keep the first observed spelling as canonical output.
        if norm not in canonical_map:
            canonical_map[norm] = str(item).strip()

    return allowed_normalized, canonical_map, raw_genres


def load_records(input_file: Union[str, Path]) -> List[Dict[str, Any]]:
    input_file = Path(input_file)
    ext = input_file.suffix.lower()

    if ext == ".csv":
        with input_file.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    if ext == ".jsonl":
        records = []
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if ext == ".json":
        text = input_file.read_text(encoding="utf-8").strip()
        if not text:
            return []

        payload = json.loads(text)

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return payload["data"]
            return [payload]

        raise ValueError("JSON input must be an object, a list of objects, or {'data': [...]}.")

    if ext == ".txt":
        return [{"genre": input_file.read_text(encoding="utf-8")}]

    raise ValueError("Unsupported input format. Use CSV, JSON, JSONL, or TXT.")


def split_genre_string(value: str) -> List[str]:
    """Split a genre string into candidate genre tags.

    Examples
    --------
    "Pop, rock, hip-hop" -> ["Pop", "rock", "hip-hop"]
    "Pop | rock / jazz" -> ["Pop", "rock", "jazz"]
    """

    value = str(value).strip()

    if not value:
        return []

    # Remove common wrappers.
    value = value.strip("[](){}")
    value = value.replace("，", ",")
    value = value.replace("、", ",")
    value = value.replace("；", ";")

    # If the whole value looks like "genre: Pop, rock", keep only the value part.
    header_match = GENRE_HEADER_PATTERN.search(value)
    if header_match:
        value = value[header_match.end():]

    # Cut when another metadata field starts.
    next_field_match = NEXT_FIELD_PATTERN.search(value)
    if next_field_match:
        value = value[:next_field_match.start()]

    parts = re.split(r"[,;/|]+|\n+", value)

    cleaned = []
    for part in parts:
        part = part.strip().strip("\"'`")
        part = re.sub(r"^\s*[-*•]\s*", "", part)
        if part:
            cleaned.append(part)

    return cleaned


def extract_genres_from_prompt_like_text(text: str) -> List[str]:
    """Extract genre values from a prompt-like text field.

    This is intentionally conservative. It only extracts when a genre header
    exists, such as "genre:" or "music genre:".
    """

    text = str(text).strip()
    if not text:
        return []

    match = GENRE_HEADER_PATTERN.search(text)
    if not match:
        return []

    after_header = text[match.end():]

    next_field_match = NEXT_FIELD_PATTERN.search(after_header)
    if next_field_match:
        after_header = after_header[:next_field_match.start()]

    return split_genre_string(after_header)


def resolve_genres_from_record(
    record: Dict[str, Any],
    genre_field: Optional[str] = None,
) -> List[str]:
    """Resolve genre tags from a record.

    Priority:
    1. Explicit --genre-field
    2. Common structured genre fields
    3. Prompt-like fields where a "genre:" header can be extracted
    """

    if genre_field:
        value = record.get(genre_field, "")
        return normalize_input_genre_value(value)

    for key in record:
        if key.lower() in GENRE_FIELD_CANDIDATES:
            value = record.get(key, "")
            genres = normalize_input_genre_value(value)
            if genres:
                return genres

    for key in record:
        if key.lower() in PROMPT_FIELD_CANDIDATES:
            extracted = extract_genres_from_prompt_like_text(str(record.get(key, "")))
            if extracted:
                return extracted

    return []


def normalize_input_genre_value(value: Any) -> List[str]:
    """Normalize raw genre input into a list of genre strings."""

    if value is None:
        return []

    if isinstance(value, list):
        output = []
        for item in value:
            if isinstance(item, dict):
                # In case genre is represented as {"name": "Pop"}
                candidate = item.get("name") or item.get("genre") or item.get("tag")
                if candidate:
                    output.extend(split_genre_string(str(candidate)))
            else:
                output.extend(split_genre_string(str(item)))
        return output

    if isinstance(value, dict):
        candidate = value.get("name") or value.get("genre") or value.get("tag") or value.get("text")
        if candidate:
            return split_genre_string(str(candidate))
        return []

    return split_genre_string(str(value))


def evaluate_genre_source(
    json_input: Union[str, Dict[str, Any]],
    allowed_genres: Optional[Set[str]] = None,
    canonical_map: Optional[Dict[str, str]] = None,
    tag_list_file: Optional[Union[str, Path]] = None,
    genre_field: Optional[str] = None,
    case_sensitive: bool = False,
    allow_fuzzy: bool = False,
    fuzzy_cutoff: float = 0.88,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """Evaluate whether genre tags are sourced from the allowed genre list.

    Parameters
    ----------
    json_input:
        A dict, JSON string, or raw genre string.
    allowed_genres:
        Optional preloaded normalized allowed genre set.
    canonical_map:
        Optional mapping from normalized genre to canonical spelling.
    tag_list_file:
        JSON file containing a `genre` list. Used if allowed_genres is not provided.
    genre_field:
        Optional explicit field containing genre content.
    case_sensitive:
        Whether matching should be case-sensitive.
    allow_fuzzy:
        If True, high-similarity fuzzy matches count as valid.
    fuzzy_cutoff:
        Similarity threshold for fuzzy matching.
    return_details:
        If True, return detailed result; otherwise return only score.

    Returns
    -------
    float or dict
        0-100 score, or detailed evaluation result.
    """

    if allowed_genres is None or canonical_map is None:
        if tag_list_file is None:
            tag_list_file = DEFAULT_TAG_LIST_FILE
        allowed_genres, canonical_map, _ = load_allowed_genres(
            tag_list_file=tag_list_file,
            case_sensitive=case_sensitive,
        )

    if isinstance(json_input, str):
        try:
            record = json.loads(json_input)
            if not isinstance(record, dict):
                record = {"genre": json_input}
        except json.JSONDecodeError:
            record = {"genre": json_input}
    elif isinstance(json_input, dict):
        record = json_input
    else:
        raise TypeError("json_input must be a dict, JSON string, or raw genre string.")

    raw_genres = resolve_genres_from_record(record, genre_field=genre_field)

    warnings = []

    if not raw_genres:
        result = {
            "genre_source_score": 0.0,
            "genre_source_grade": "poor",
            "all_genres_from_list": False,
            "valid_genres": [],
            "invalid_genres": [],
            "matched_canonical_genres": [],
            "suggestions": {},
            "num_genres": 0,
            "num_valid": 0,
            "num_invalid": 0,
            "warnings": ["No genre content found."],
        }
        return result if return_details else 0.0

    valid_genres = []
    invalid_genres = []
    matched_canonical_genres = []
    suggestions = {}

    allowed_list = sorted(allowed_genres)

    for raw in raw_genres:
        norm = normalize_tag(raw, case_sensitive=case_sensitive)

        if norm in allowed_genres:
            valid_genres.append(raw)
            matched_canonical_genres.append(canonical_map.get(norm, raw))
            continue

        if allow_fuzzy:
            match = get_close_matches(norm, allowed_list, n=1, cutoff=fuzzy_cutoff)
            if match:
                matched_norm = match[0]
                valid_genres.append(raw)
                matched_canonical_genres.append(canonical_map.get(matched_norm, raw))
                suggestions[raw] = canonical_map.get(matched_norm, matched_norm)
                continue

        invalid_genres.append(raw)
        nearest = get_close_matches(norm, allowed_list, n=3, cutoff=0.6)
        if nearest:
            suggestions[raw] = [canonical_map.get(item, item) for item in nearest]

    num_genres = len(raw_genres)
    num_valid = len(valid_genres)
    num_invalid = len(invalid_genres)

    score = round((num_valid / num_genres) * 100.0, 2)

    if num_invalid > 0:
        warnings.append("Some genre tags are not found in the allowed genre list.")

    if score >= 100:
        grade = "pass"
    elif score >= 80:
        grade = "mostly_pass"
    elif score >= 50:
        grade = "partial"
    else:
        grade = "fail"

    result = {
        "genre_source_score": score,
        "genre_source_grade": grade,
        "all_genres_from_list": num_invalid == 0,
        "valid_genres": valid_genres,
        "invalid_genres": invalid_genres,
        "matched_canonical_genres": matched_canonical_genres,
        "suggestions": suggestions,
        "num_genres": num_genres,
        "num_valid": num_valid,
        "num_invalid": num_invalid,
        "warnings": warnings,
    }

    return result if return_details else score


def compute_batch_scores(
    records: List[Dict[str, Any]],
    allowed_genres: Set[str],
    canonical_map: Dict[str, str],
    genre_field: Optional[str],
    case_sensitive: bool,
    allow_fuzzy: bool,
    fuzzy_cutoff: float,
) -> List[Dict[str, Any]]:
    output_records = []

    for record in records:
        result = evaluate_genre_source(
            json_input=record,
            allowed_genres=allowed_genres,
            canonical_map=canonical_map,
            genre_field=genre_field,
            case_sensitive=case_sensitive,
            allow_fuzzy=allow_fuzzy,
            fuzzy_cutoff=fuzzy_cutoff,
            return_details=True,
        )

        item = dict(record)
        item["genre_source_score"] = result["genre_source_score"]
        item["genre_source_grade"] = result["genre_source_grade"]
        item["all_genres_from_list"] = result["all_genres_from_list"]
        item["valid_genres"] = result["valid_genres"]
        item["invalid_genres"] = result["invalid_genres"]
        item["matched_canonical_genres"] = result["matched_canonical_genres"]
        item["genre_source_suggestions"] = result["suggestions"]
        item["genre_source_warnings"] = result["warnings"]

        output_records.append(item)

    return output_records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float(item["genre_source_score"]) for item in records]

    if not scores:
        return {}

    num_pass = sum(item["all_genres_from_list"] for item in records)

    return {
        "num_items": len(scores),
        "mean_score": round(statistics.mean(scores), 2),
        "median_score": round(statistics.median(scores), 2),
        "min_score": round(min(scores), 2),
        "max_score": round(max(scores), 2),
        "num_all_genres_from_list": num_pass,
        "ratio_all_genres_from_list": round(num_pass / len(scores), 4),
        "count_above_100": sum(score >= 100 for score in scores),
        "count_above_80": sum(score >= 80 for score in scores),
        "count_above_50": sum(score >= 50 for score in scores),
    }


def write_output_json(records: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8", newline="") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        output_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()

    records = load_records(args.input_file)

    allowed_genres, canonical_map, raw_genres = load_allowed_genres(
        tag_list_file=args.tag_list_file,
        case_sensitive=args.case_sensitive,
    )

    scored_records = compute_batch_scores(
        records=records,
        allowed_genres=allowed_genres,
        canonical_map=canonical_map,
        genre_field=args.genre_field,
        case_sensitive=args.case_sensitive,
        allow_fuzzy=args.allow_fuzzy,
        fuzzy_cutoff=args.fuzzy_cutoff,
    )

    summary = summarize(scored_records)

    print("Genre source evaluation results:")
    print(f"Allowed genre entries loaded: {len(raw_genres)}")
    print(f"Unique normalized allowed genres: {len(allowed_genres)}")

    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.output_file:
        write_output_json(scored_records, args.output_file)
        print(f"Saved genre source scores to: {args.output_file}")


if __name__ == "__main__":
    main()
