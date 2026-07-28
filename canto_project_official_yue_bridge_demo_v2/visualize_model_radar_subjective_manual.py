#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize selected model result CSVs on one radar chart,
including manually provided subjective evaluation scores.

Expected per-model CSV format:
image_name,clip_alignment,emotion_similarity,lyrics_format_score,quality_overall

Subjective scores are NOT read from CSV. They are provided by command arguments.

Examples:
python visualize_model_radar_subjective_manual.py \
  --input-dir outputs/batch_eval_50 \
  --models intern_rag intern \
  --subjective-scores intern_rag=85 intern=72 \
  --output outputs/batch_eval_50/model_radar_subjective_manual.png \
  --summary-output outputs/batch_eval_50/model_metric_summary_subjective_manual.csv

python visualize_model_radar_subjective_manual.py \
  --input-dir outputs/batch_eval_50 \
  --models intern_rag intern qwen \
  --subjective-values 85 72 78 \
  --output outputs/batch_eval_50/model_radar_subjective_all.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_METRIC_COLS = [
    "clip_alignment",
    "emotion_similarity",
    "lyrics_format_score",
    "quality_overall",
]

SUBJECTIVE_COL = "subjective_eval"
METRIC_COLS = BASE_METRIC_COLS + [SUBJECTIVE_COL]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected model result CSVs with manual subjective scores."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing results_intern.csv, results_intern_rag.csv, etc.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["intern_rag", "intern"],
        help="Model keys to visualize. Example: intern_rag intern qwen",
    )

    parser.add_argument(
        "--subjective-scores",
        nargs="*",
        default=None,
        help=(
            "Manual subjective scores by model. "
            "Format: model=score model=score. "
            "Example: --subjective-scores intern_rag=85 intern=72 qwen=78"
        ),
    )

    parser.add_argument(
        "--subjective-values",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Alternative subjective scores in the same order as --models. "
            "Example: --models intern_rag intern --subjective-values 85 72"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output radar chart path, e.g. model_radar_subjective.png.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional path to save aggregated mean scores as CSV.",
    )

    return parser.parse_args()


def parse_subjective_score_items(items: list[str] | None) -> dict[str, float]:
    score_map: dict[str, float] = {}

    if not items:
        return score_map

    for item in items:
        item = str(item).strip()
        if not item:
            continue

        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid subjective score item: {item}. "
                "Use model=score, for example intern_rag=85."
            )

        key = key.strip()
        value = value.strip().replace("%", "")

        if not key:
            raise ValueError(f"Invalid empty model key in item: {item}")

        try:
            score_map[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid score value in item: {item}") from exc

    return score_map


def build_subjective_score_map(args: argparse.Namespace) -> dict[str, float]:
    score_map = parse_subjective_score_items(args.subjective_scores)

    if args.subjective_values is not None and len(args.subjective_values) > 0:
        if score_map:
            raise ValueError("Use either --subjective-scores or --subjective-values, not both.")

        if len(args.subjective_values) != len(args.models):
            raise ValueError(
                "--subjective-values length must match --models length. "
                f"Got {len(args.subjective_values)} values for {len(args.models)} models."
            )

        score_map = {
            model_key: float(score)
            for model_key, score in zip(args.models, args.subjective_values)
        }

    # Normalize 0-100 to 0-1 if needed.
    if score_map:
        max_score = max(score_map.values())
        if max_score > 1.5:
            score_map = {
                model_key: score / 100.0
                for model_key, score in score_map.items()
            }

    return score_map


def load_model_csvs(input_dir: Path, models: list[str]) -> pd.DataFrame:
    all_dfs = []

    for model_key in models:
        csv_path = input_dir / f"results_{model_key}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Model CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = {"image_name", *BASE_METRIC_COLS}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

        df = df.copy()
        df["model_key"] = model_key
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def normalize_base_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in BASE_METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["lyrics_format_score"].dropna().max() > 1.5:
        df["lyrics_format_score"] = df["lyrics_format_score"] / 100.0

    return df


def aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    mean_summary = (
        df.groupby("model_key", as_index=False)[BASE_METRIC_COLS]
        .mean()
        .sort_values("model_key")
        .reset_index(drop=True)
    )

    valid_counts = (
        df.groupby("model_key")[BASE_METRIC_COLS]
        .count()
        .add_suffix("_valid_count")
        .reset_index()
    )

    total_counts = (
        df.groupby("model_key")
        .size()
        .reset_index(name="total_records")
    )

    summary = (
        mean_summary
        .merge(valid_counts, on="model_key", how="left")
        .merge(total_counts, on="model_key", how="left")
    )

    return summary


def apply_subjective_scores(
    summary: pd.DataFrame,
    score_map: dict[str, float],
    models: list[str],
) -> pd.DataFrame:
    summary = summary.copy()

    unknown_models = sorted(set(score_map) - set(models))
    if unknown_models:
        raise ValueError(
            "Subjective scores contain model keys not listed in --models: "
            f"{unknown_models}"
        )

    summary[SUBJECTIVE_COL] = summary["model_key"].map(score_map)
    summary[f"{SUBJECTIVE_COL}_valid_count"] = summary[SUBJECTIVE_COL].notna().astype(int)

    return summary


def plot_radar(summary: pd.DataFrame, output_path: Path) -> None:
    labels = [
        "CLIP Alignment",
        "Emotion Similarity",
        "Lyrics Format",
        "Lyrics Quality",
        "Subjective Eval",
    ]

    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.5, 8.5))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)

    for _, row in summary.iterrows():
        values = [row[col] for col in METRIC_COLS]
        values = [0 if pd.isna(v) else v for v in values]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=row["model_key"])
        ax.fill(angles, values, alpha=0.20)

    ax.set_title("Model Comparison Radar Chart with Subjective Evaluation", pad=22, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    subjective_score_map = build_subjective_score_map(args)

    df = load_model_csvs(args.input_dir, args.models)
    df = normalize_base_metrics(df)

    summary = aggregate_by_model(df)
    summary = apply_subjective_scores(
        summary=summary,
        score_map=subjective_score_map,
        models=args.models,
    )

    print("Aggregated mean scores by model:")
    print(summary)

    print("\nManual subjective scores used:")
    if subjective_score_map:
        for model_key in args.models:
            value = subjective_score_map.get(model_key, np.nan)
            print(f"  {model_key}: {value if not pd.isna(value) else 'missing'}")
    else:
        print("  None. Subjective Eval will be plotted as 0.")

    print("\nValid metric counts:")
    count_cols = [
        "model_key",
        "total_records",
        "clip_alignment_valid_count",
        "emotion_similarity_valid_count",
        "lyrics_format_score_valid_count",
        "quality_overall_valid_count",
        "subjective_eval_valid_count",
    ]
    print(summary[count_cols])

    if summary["subjective_eval_valid_count"].sum() == 0:
        print(
            "\nWarning: No manual subjective scores provided. "
            "The subjective dimension will be plotted as 0."
        )

    missing_subjective = summary.loc[
        summary["subjective_eval"].isna(),
        "model_key",
    ].tolist()
    if missing_subjective:
        print(
            "\nWarning: Missing subjective scores for models: "
            + ", ".join(missing_subjective)
            + ". They will be plotted as 0."
        )

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False, encoding="utf-8-sig")
        print(f"Saved summary CSV to: {args.summary_output}")

    plot_radar(summary, args.output)
    print(f"Saved radar chart to: {args.output}")


if __name__ == "__main__":
    main()
