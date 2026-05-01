#!/usr/bin/env python3
"""Evaluate lyrics format using rule-based checks and Transformer format similarity.

This evaluator focuses on lyrics FORMAT only:
- section tags such as [verse], [chorus], [bridge], [outro], [end]
- blank lines between sections
- no blank lines inside sections
- section order and structural similarity
- Transformer similarity based on normalized format signatures

It does NOT evaluate lyric meaning, creativity, rhyme, or semantic quality.
"""

import argparse
import csv
import json
import re
import sys
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModel


repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

try:
    from paths import EVAL, IPP
except Exception:
    EVAL = repo_root / "EVAL"
    IPP = repo_root / "IPP"


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TEXT_FIELD_CANDIDATES = [
    "lyrics_text",
    "lyrics",
    "text",
    "song_lyrics",
    "generated_lyrics",
]

VALID_TAGS = {
    "verse",
    "chorus",
    "bridge",
    "outro",
    "end",
    "intro",
    "pre-chorus",
    "prechorus",
    "hook",
}

TAG_PATTERN = re.compile(r"^\[([A-Za-z\-]+)\]$")


DEFAULT_REFERENCE_FORMAT = """[verse]
line
line
line
line

[chorus]
line
line
line
line
line
line

[verse]
line
line
line
line

[chorus]
line
line
line
line
line
line

[bridge]
line
line
line
line

[outro]
line
line
line
line

[end]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate lyrics format with rule checks and Transformer format similarity."
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=str(IPP / "generated_lyrics.json"),
        help="Input CSV, JSON, JSONL, or TXT file containing lyrics.",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help="Optional reference lyrics format file. If omitted, a built-in reference format is used.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Optional field name containing lyrics text.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face Transformer model name or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Transformer embedding computation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on: cpu or cuda.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(EVAL / "lyrics_format_evaluation" / "outputs" / "lyrics_format_transformer_scores.json"),
        help="Output JSON or JSONL file.",
    )
    parser.add_argument(
        "--rule-weight",
        type=float,
        default=0.50,
        help="Weight for strict rule-based format score.",
    )
    parser.add_argument(
        "--transformer-weight",
        type=float,
        default=0.30,
        help="Weight for Transformer format signature similarity score.",
    )
    parser.add_argument(
        "--sequence-weight",
        type=float,
        default=0.20,
        help="Weight for section sequence similarity score.",
    )

    return parser.parse_args()


def load_records(input_file: Path) -> List[Dict[str, Any]]:
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
        return [{"lyrics_text": input_file.read_text(encoding="utf-8")}]

    raise ValueError("Unsupported input format. Use CSV, JSON, JSONL, or TXT.")


def write_output_json(records: List[Dict[str, Any]], output_path: Path) -> None:
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


def resolve_text_from_payload(
    payload: Union[str, Dict[str, Any]],
    text_field: Optional[str] = None,
) -> str:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload

    if not isinstance(payload, dict):
        raise ValueError("json_input must be a JSON string, raw lyrics string, or dict.")

    if text_field:
        return str(payload.get(text_field, "") or "")

    text_key = next(
        (k for k in payload if k.lower() in TEXT_FIELD_CANDIDATES),
        None,
    )

    if text_key is None:
        raise ValueError(f"Payload is missing a lyrics text field; got {list(payload.keys())}")

    return str(payload[text_key])


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def is_tag_line(line: str) -> bool:
    return bool(TAG_PATTERN.match(line.strip()))


def get_tag(line: str) -> str:
    match = TAG_PATTERN.match(line.strip())
    if not match:
        return ""
    return match.group(1).lower()


def extract_tags(lines: List[str]) -> List[str]:
    return [get_tag(line) for line in lines if is_tag_line(line)]


def extract_section_line_counts(lyrics: str) -> Dict[str, int]:
    """
    Extract section line counts.

    Example:
    [verse]
    line
    line

    [chorus]
    line

    [end]

    -> {"verse": 2, "chorus": 1, "end": 0}
    """

    lyrics = normalize_line_endings(lyrics)
    lines = lyrics.split("\n")

    section_counts = {}
    current_tag = None

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            continue

        if is_tag_line(stripped):
            current_tag = get_tag(stripped)
            section_counts[current_tag] = 0
        else:
            if current_tag is not None:
                section_counts[current_tag] += 1

    return section_counts


def extract_required_tags_from_reference(reference_lyrics: str) -> List[str]:
    """
    Extract required tags from reference lyrics.

    4 / 8 line reference:
    ["verse", "chorus", "end"]

    16 line reference:
    ["verse", "chorus", "bridge", "outro", "end"]
    """

    tags = extract_tags(normalize_line_endings(reference_lyrics).split("\n"))

    # Remove duplicates but keep order.
    output = []
    for tag in tags:
        if tag not in output:
            output.append(tag)

    return output


def build_format_signature(lyrics: str) -> str:
    """
    Convert lyrics into a structure-only signature.

    Example:

    [verse]
    hello
    world

    [chorus]
    sing

    [end]

    becomes:

    TAG_VERSE
    LYRIC_LINE
    LYRIC_LINE
    BLANK
    TAG_CHORUS
    LYRIC_LINE
    BLANK
    TAG_END
    """

    lyrics = normalize_line_endings(lyrics)
    lines = lyrics.split("\n")

    signature_tokens = []

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            signature_tokens.append("BLANK")
        elif is_tag_line(stripped):
            tag = get_tag(stripped)
            signature_tokens.append(f"TAG_{tag.upper()}")
        else:
            signature_tokens.append("LYRIC_LINE")

    return "\n".join(signature_tokens)


def build_compact_structure_signature(lyrics: str) -> str:
    """
    Build a compact section-level structure signature.

    Example:
    TAG_VERSE:4 | TAG_CHORUS:6 | TAG_BRIDGE:4 | TAG_OUTRO:4 | TAG_END
    """

    lyrics = normalize_line_endings(lyrics)
    lines = lyrics.split("\n")

    sections = []
    current_tag = None
    current_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            continue

        if is_tag_line(stripped):
            if current_tag is not None:
                sections.append((current_tag, current_count))

            current_tag = get_tag(stripped)
            current_count = 0
        else:
            if current_tag is not None:
                current_count += 1

    if current_tag is not None:
        sections.append((current_tag, current_count))

    parts = []
    for tag, count in sections:
        if tag == "end":
            parts.append("TAG_END")
        else:
            parts.append(f"TAG_{tag.upper()}:{count}")

    return " | ".join(parts)


def compute_rule_format_score(
    lyrics: str,
    reference_lyrics: Optional[str] = None,
) -> Tuple[float, List[str], Dict[str, Any]]:

    """
    Strict character-level / blank-line / tag format score.

    This keeps the original exact-format comparison logic.
    """

    warnings = []

    lyrics = normalize_line_endings(lyrics)
    if not lyrics:
        return 0.0, ["No lyrics text found."], {}

    lines = lyrics.split("\n")

    score = 100.0

    tag_lines = []
    blank_lines = []
    lyric_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "":
            blank_lines.append(i)
        elif is_tag_line(stripped):
            tag_lines.append((i, get_tag(stripped)))
        else:
            lyric_lines.append((i, stripped))

    tags = [tag for _, tag in tag_lines]

    if reference_lyrics is None:
        reference_lyrics = DEFAULT_REFERENCE_FORMAT

    required_tags = extract_required_tags_from_reference(reference_lyrics)
    expected_line_counts = extract_section_line_counts(reference_lyrics)
    actual_line_counts = extract_section_line_counts(lyrics)

    if not tag_lines:
        score -= 35
        warnings.append("No valid section tags found.")
        
    for required_tag in required_tags:
        if required_tag not in tags:
            penalty = 15 if required_tag == "end" else 10
            score -= penalty
            warnings.append(f"Missing required [{required_tag}] section.")

    invalid_tags = [tag for tag in tags if tag not in VALID_TAGS]
    if invalid_tags:
        penalty = min(15, len(invalid_tags) * 5)
        score -= penalty
        warnings.append(f"Invalid section tags found: {invalid_tags}")

    # malformed tag or tag mixed with text
    for i, line in enumerate(lines):
        stripped = line.strip()

        if "[" in stripped or "]" in stripped:
            if not is_tag_line(stripped):
                score -= 5
                warnings.append(
                    f"Line {i + 1} contains malformed tag or tag mixed with text."
                )

    # first non-empty line should be tag
    first_non_empty = next(
        (i for i, line in enumerate(lines) if line.strip() != ""),
        None,
    )

    if first_non_empty is not None and not is_tag_line(lines[first_non_empty]):
        score -= 10
        warnings.append("First non-empty line should be a section tag.")

    # final non-empty line should be [end]
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip() != ""]

    if non_empty_indices:
        last_non_empty = non_empty_indices[-1]
        if lines[last_non_empty].strip().lower() != "[end]":
            score -= 15
            warnings.append("Final non-empty line should be [end].")

    # no content after [end]
    if "end" in tags:
        end_index = next(i for i, tag in tag_lines if tag == "end")
        after_end = [
            line for line in lines[end_index + 1:]
            if line.strip() != ""
        ]
        if after_end:
            score -= 15
            warnings.append("There should be no content after [end].")

    # blank line rules
    for i in blank_lines:
        prev_line = lines[i - 1].strip() if i > 0 else ""
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if prev_line == "" or next_line == "":
            score -= 5
            warnings.append(f"Line {i + 1} has repeated or unnecessary blank lines.")
            continue

        if not is_tag_line(next_line):
            score -= 5
            warnings.append(
                f"Line {i + 1} is a blank line not followed by a section tag."
            )

    # missing blank line before a new section tag
    for idx, tag in tag_lines:
        if idx == 0:
            continue

        prev_line = lines[idx - 1].strip()

        if prev_line != "":
            score -= 5
            warnings.append(f"Missing blank line before [{tag}] at line {idx + 1}.")

    # blank line immediately after tag
    for idx, tag in tag_lines:
        if tag == "end":
            continue

        if idx + 1 < len(lines) and lines[idx + 1].strip() == "":
            score -= 5
            warnings.append(
                f"Unexpected blank line immediately after [{tag}] at line {idx + 1}."
            )

    # empty sections
    for pos, (idx, tag) in enumerate(tag_lines):
        if tag == "end":
            continue

        next_tag_idx = len(lines)
        if pos + 1 < len(tag_lines):
            next_tag_idx = tag_lines[pos + 1][0]

        section_content = [
            line for line in lines[idx + 1:next_tag_idx]
            if line.strip() != ""
        ]

        if not section_content:
            score -= 10
            warnings.append(f"Section [{tag}] at line {idx + 1} is empty.")
            
        for tag, expected_count in expected_line_counts.items():
            if tag == "end":
                continue

            actual_count = actual_line_counts.get(tag)

            if actual_count is None:
                continue

            if actual_count != expected_count:
                score -= 5
                warnings.append(
                    f"Section [{tag}] has {actual_count} lyric lines; expected {expected_count}."
                )
                

    metrics = {
        "num_lines": len(lines),
        "num_tags": len(tag_lines),
        "num_blank_lines": len(blank_lines),
        "num_lyric_lines": len(lyric_lines),
        "tags": tags,
        "has_verse": "verse" in tags,
        "has_chorus": "chorus" in tags,
        "has_bridge": "bridge" in tags,
        "has_outro": "outro" in tags,
        "has_end": "end" in tags,
        "invalid_tags": invalid_tags,
    }

    return round(max(0.0, min(score, 100.0)), 2), warnings, metrics


def compute_sequence_structure_score(candidate_lyrics: str, reference_lyrics: str) -> float:
    """
    Compare tag order and section line-count pattern using SequenceMatcher.

    This is not Transformer-based. It is a lightweight structural similarity score.
    """

    candidate_signature = build_compact_structure_signature(candidate_lyrics)
    reference_signature = build_compact_structure_signature(reference_lyrics)

    ratio = SequenceMatcher(
        None,
        candidate_signature,
        reference_signature,
    ).ratio()

    return round(ratio * 100, 2)


def mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    return summed / counts


def load_transformer_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device


def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
    normalize: bool = True,
) -> torch.Tensor:
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs["attention_mask"])

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def compute_transformer_format_scores(
    candidate_lyrics_list: List[str],
    reference_lyrics: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 16,
) -> List[float]:
    """
    Compute Transformer similarity based only on normalized format signatures.

    Lyric content is replaced with LYRIC_LINE, so the model compares structure,
    not actual words.
    """

    reference_signature = build_format_signature(reference_lyrics)
    candidate_signatures = [
        build_format_signature(text)
        for text in candidate_lyrics_list
    ]

    all_texts = [reference_signature] + candidate_signatures

    embeddings = encode_texts(
        texts=all_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        normalize=True,
    )

    reference_embedding = embeddings[0:1]
    candidate_embeddings = embeddings[1:]

    sims = torch.nn.functional.cosine_similarity(
        candidate_embeddings,
        reference_embedding.expand_as(candidate_embeddings),
        dim=1,
    )

    # cosine similarity usually ranges from -1 to 1.
    # Convert to 0-100.
    scores = ((sims + 1.0) / 2.0 * 100.0).clamp(0, 100)

    return [round(float(score), 2) for score in scores.tolist()]


def grade_score(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "acceptable"
    if score >= 40:
        return "weak"
    return "poor"


def score_lyrics_format_hybrid(
    json_input: Union[str, Dict[str, Any]],
    reference_lyrics: Optional[str] = None,
    text_field: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    model: Optional[AutoModel] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None,
    rule_weight: float = 0.50,
    transformer_weight: float = 0.30,
    sequence_weight: float = 0.20,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    Python API for single lyrics scoring.

    This function can be imported by other modules.

    The score focuses on format only:
    - strict rule score
    - Transformer similarity between format signatures
    - section sequence similarity
    """

    lyrics = resolve_text_from_payload(json_input, text_field=text_field)
    reference_lyrics = reference_lyrics or DEFAULT_REFERENCE_FORMAT

    total_weight = rule_weight + transformer_weight + sequence_weight
    if total_weight <= 0:
        raise ValueError("At least one score weight must be positive.")

    rule_weight = rule_weight / total_weight
    transformer_weight = transformer_weight / total_weight
    sequence_weight = sequence_weight / total_weight

    rule_score, warnings, metrics = compute_rule_format_score(
        lyrics,
        reference_lyrics=reference_lyrics,
    )
    sequence_score = compute_sequence_structure_score(lyrics, reference_lyrics)

    if tokenizer is None or model is None:
        tokenizer, model, device = load_transformer_model(
            model_name=model_name,
            device=device,
        )
    else:
        device = torch.device(device if device else next(model.parameters()).device)

    transformer_score = compute_transformer_format_scores(
        candidate_lyrics_list=[lyrics],
        reference_lyrics=reference_lyrics,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=1,
    )[0]

    final_score = (
        rule_weight * rule_score
        + transformer_weight * transformer_score
        + sequence_weight * sequence_score
    )

    final_score = round(float(final_score), 2)

    result = {
        "lyrics_format_score": final_score,
        "grade": grade_score(final_score),
        "components": {
            "rule_format_score": rule_score,
            "transformer_format_similarity_score": transformer_score,
            "sequence_structure_score": sequence_score,
            "rule_weight": round(rule_weight, 4),
            "transformer_weight": round(transformer_weight, 4),
            "sequence_weight": round(sequence_weight, 4),
        },
        "metrics": {
            **metrics,
            "format_signature": build_format_signature(lyrics),
            "compact_structure_signature": build_compact_structure_signature(lyrics),
            "reference_compact_structure_signature": build_compact_structure_signature(reference_lyrics),
        },
        "warnings": warnings,
    }

    return result if return_details else final_score


def compute_batch_scores(
    records: List[Dict[str, Any]],
    reference_lyrics: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    text_field: Optional[str],
    rule_weight: float,
    transformer_weight: float,
    sequence_weight: float,
) -> List[Dict[str, Any]]:
    total_weight = rule_weight + transformer_weight + sequence_weight
    if total_weight <= 0:
        raise ValueError("At least one score weight must be positive.")

    rule_weight = rule_weight / total_weight
    transformer_weight = transformer_weight / total_weight
    sequence_weight = sequence_weight / total_weight

    lyrics_list = [
        resolve_text_from_payload(record, text_field=text_field)
        for record in records
    ]

    transformer_scores = compute_transformer_format_scores(
        candidate_lyrics_list=lyrics_list,
        reference_lyrics=reference_lyrics,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
    )

    output_records = []

    for record, lyrics, transformer_score in zip(records, lyrics_list, transformer_scores):
        rule_score, warnings, metrics = compute_rule_format_score(lyrics, reference_lyrics=reference_lyrics)
        sequence_score = compute_sequence_structure_score(lyrics, reference_lyrics)

        final_score = (
            rule_weight * rule_score
            + transformer_weight * transformer_score
            + sequence_weight * sequence_score
        )

        final_score = round(float(final_score), 2)

        item = dict(record)
        item["lyrics_format_score"] = final_score
        item["lyrics_format_grade"] = grade_score(final_score)
        item["lyrics_format_components"] = {
            "rule_format_score": rule_score,
            "transformer_format_similarity_score": transformer_score,
            "sequence_structure_score": sequence_score,
            "rule_weight": round(rule_weight, 4),
            "transformer_weight": round(transformer_weight, 4),
            "sequence_weight": round(sequence_weight, 4),
        }
        item["lyrics_format_metrics"] = {
            **metrics,
            "format_signature": build_format_signature(lyrics),
            "compact_structure_signature": build_compact_structure_signature(lyrics),
            "reference_compact_structure_signature": build_compact_structure_signature(reference_lyrics),
        }
        item["lyrics_format_warnings"] = warnings

        output_records.append(item)

    return output_records


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [item["lyrics_format_score"] for item in records]

    if not scores:
        return {}

    return {
        "num_items": len(scores),
        "mean_score": round(statistics.mean(scores), 2),
        "median_score": round(statistics.median(scores), 2),
        "min_score": round(min(scores), 2),
        "max_score": round(max(scores), 2),
        "count_above_90": sum(score >= 90 for score in scores),
        "count_above_75": sum(score >= 75 for score in scores),
        "count_above_60": sum(score >= 60 for score in scores),
    }


def main() -> None:
    args = parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    records = load_records(input_file)

    if args.reference_file:
        reference_lyrics = Path(args.reference_file).read_text(encoding="utf-8")
    else:
        reference_lyrics = DEFAULT_REFERENCE_FORMAT

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    tokenizer, model, device = load_transformer_model(
        model_name=args.model_name,
        device=device,
    )

    scored_records = compute_batch_scores(
        records=records,
        reference_lyrics=reference_lyrics,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        text_field=args.text_field,
        rule_weight=args.rule_weight,
        transformer_weight=args.transformer_weight,
        sequence_weight=args.sequence_weight,
    )

    summary = summarize(scored_records)

    print("Lyrics format evaluation results:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.output_file:
        write_output_json(scored_records, output_file)
        print(f"Saved lyrics format scores to: {args.output_file}")


if __name__ == "__main__":
    main()