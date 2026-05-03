from __future__ import annotations
from io import BytesIO
from pathlib import Path
import os
import sys
import traceback
import json
import argparse
import importlib.util
import streamlit as st
from PIL import Image

from state_utils import init_state, hard_reset
from schemas import LyricsPromptBundle
from generator import generate_from_image, generate_song_auto
from modules.mm_direct_gen import unload_mm_models, build_lyrics_format_instruction, normalize_lyrics_format

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Ensure repo root is in path for imports
from paths import PROJECT_ROOT, DEMO, EVAL

st.set_page_config(page_title="Image2CantonSong", page_icon="🎵", layout="wide")
init_state()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--debug", action="store_true")
args, _ = parser.parse_known_args()
debug_mode = args.debug

LYRIC_LENGTH_TO_SEGMENTS = {
    4: 2,
    8: 3,
    16: 5,
}
STYLE_PRESETS = {
    "Cantonese Ballad": (
        "female Cantonese Melancholic Classical airy vocal "
        "Piano bright vocal Pop Nostalgic Violin"
    )
}

TAG_CATEGORIES = ["genre", "instrument", "mood", "gender", "timbre"]

MANDATORY_TAG_LIST_STYLE_TAGS = ["Cantonese"]


def ensure_mandatory_style_tags(style_prompt: str) -> str:
    """Ensure mandatory tags are always included in tag-list style mode."""
    style_prompt = style_prompt.strip()

    existing_lower = {
        token.strip().lower()
        for token in style_prompt.split()
        if token.strip()
    }

    final_tags = []

    for tag in MANDATORY_TAG_LIST_STYLE_TAGS:
        if tag.lower() not in existing_lower:
            final_tags.append(tag)

    if style_prompt:
        final_tags.append(style_prompt)

    return " ".join(final_tags).strip()


@st.cache_data
def load_top_200_tags() -> dict:
    """Load selectable style tags from top_200_tags.json."""
    candidate_paths = [
        PROJECT_ROOT / "top_200_tags.json",
        Path(__file__).resolve().parent / "top_200_tags.json",
        PROJECT_ROOT / "Evaluation" / "genre_alignment" /"top_200_tags.json",
    ]

    for path in candidate_paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        "Cannot find top_200_tags.json. Please place it under PROJECT_ROOT "
        "or the current demo folder."
    )


def unique_clean_tags(items: list[str]) -> list[str]:
    """Remove duplicate tags while preserving readable order."""
    seen = set()
    output = []

    for item in items:
        tag = str(item).strip()
        if not tag:
            continue

        key = tag.lower()
        if key not in seen:
            seen.add(key)
            output.append(tag)

    return output


def build_style_prompt_from_selected_tags(selected_by_category: dict[str, list[str]]) -> str:
    """Flatten selected tags into one YuE genre prompt string."""
    ordered_tags = []

    for category in TAG_CATEGORIES:
        ordered_tags.extend(selected_by_category.get(category, []))

    return " ".join(tag.strip() for tag in ordered_tags if tag.strip()).strip()


def sync_run_n_segments_to_line_count() -> None:
    """Update run_n_segments when lyric length changes.

    User can still manually edit run_n_segments afterwards.
    """
    line_count_value = int(st.session_state.get("line_count", 8))
    st.session_state["run_n_segments"] = LYRIC_LENGTH_TO_SEGMENTS.get(
        line_count_value,
        3,
    )
    

def load_example_prompt_bundle() -> LyricsPromptBundle:
    example_path = Path(__file__).resolve().parent / "examples" \
                        / "noosa_everglades_prompt_bundle.json"
    if not example_path.exists():
        raise FileNotFoundError(f"Example prompt bundle not found: {example_path}")
    raw = json.loads(example_path.read_text(encoding="utf-8"))
    return LyricsPromptBundle(
        title=raw.get("title", ""),
        lyrics_text=raw.get("lyrics_text", ""),
        genre_prompt=raw.get("genre_prompt", ""),
        language_tag=raw.get("language_tag", "Cantonese"),
        raw_meta=raw.get("raw_meta", {}),
    )


def load_image_text_similarity_module() -> object:
    module_path = EVAL / "image_lyrics_alignment" / "clip_image_text_alignment.py"
    spec = importlib.util.spec_from_file_location("clip_image_text_alignment", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load similarity module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_image_lyrics_emotion_similarity_module() -> object:
    module_path = EVAL / "image_lyrics_emotion" / "image_lyrics_emotion_similarity.py"
    spec = importlib.util.spec_from_file_location("image_lyrics_emotion_similarity", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load emotion similarity module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_lyrics_format_transformer_score_module() -> object:
    module_path = EVAL / "lyrics_format" / "lyrics_format_transformer_score.py"
    spec = importlib.util.spec_from_file_location("lyrics_format_transformer_score", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load lyrics format evaluation module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cantonese_lyrics_quality_module() -> object:
    module_path = EVAL / "lyrics_quality" / "lyrics_quality_evaluation.py"
    spec = importlib.util.spec_from_file_location("lyrics_quality_evaluation", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load lyrics quality module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strip_lyrics_section_tags(lyrics_text: str) -> str:
    """Remove section tags before Cantonese lyrics quality evaluation."""
    section_tags = {
        "[verse]",
        "[chorus]",
        "[bridge]",
        "[outro]",
        "[end]",
    }

    lines = []

    for line in lyrics_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.lower() in section_tags:
            continue

        lines.append(stripped)

    return "\n".join(lines)


def reset_evaluation_results() -> None:
    """Clear all evaluation results when a new lyrics prompt is generated."""
    keys_to_clear = [
        # Image-lyrics alignment
        "image_lyrics_alignment",
        "image_lyrics_alignment_error",

        # Image-lyrics emotion similarity
        "image_lyrics_emotion_similarity",
        "image_lyrics_emotion_similarity_error",
        "image_lyrics_emotion_predictions",
        "image_lyrics_emotion_results",

        # Lyrics format evaluation
        "lyrics_format_score_result",
        "lyrics_format_score_error",
        "show_lyrics_format_metrics",

        # Cantonese lyrics quality evaluation
        "cantonese_lyrics_quality_result",
        "cantonese_lyrics_quality_error",
        "show_cantonese_rhyme_debug",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

@st.dialog("Debug log")
def show_debug_log(log_text: str):
    st.code(log_text, language="python")

@st.dialog("Image-lyrics alignment debug log")
def show_image_lyrics_alignment_debug_log(log_text: str):
    st.code(log_text, language="python")

@st.dialog("Image-lyrics emotion debug log")
def show_image_lyrics_emotion_debug_log(log_text: str):
    st.code(log_text, language="python")
    
@st.dialog("Lyrics format evaluation debug log")
def show_lyrics_format_evaluation_debug_log(log_text: str):
    st.code(log_text, language="python")
    
@st.dialog("lyrics quality debug log")
def show_cantonese_lyrics_quality_debug_log(log_text: str):
    st.code(log_text, language="python")

if debug_mode:
    st.info("Debug mode enabled.")

st.title("🎵 Image2CantonSong")
st.caption("Image → multimodal lyrics draft → manual confirm → final generation by original YuE environment")

with st.sidebar:
    if st.button("Reset app state"):
        hard_reset()
        st.rerun()

    st.header("HuggingFace")
    hf_token = st.text_input("HF Token", type="password", placeholder="hf_...")
    if hf_token:
        st.caption("✅ Token provided — used to load private HF models/adapters.")
    else:
        st.caption("⚠️ Leave blank if all models are public.")

    st.header("Step 2 — Multimodal LLM")
    MM_MODEL_OPTIONS = [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "OpenGVLab/InternVL2-4B",
    ]
    mm_model = st.selectbox("Image-capable HF LLM", options=MM_MODEL_OPTIONS, index=0)

    # RAG — only available for InternVL2-4B
    use_rag = False
    rag_csv_path = ""
    rag_top_k = 3
    if mm_model == "OpenGVLab/InternVL2-4B":
        use_rag = st.checkbox(
            "Use RAG (corpus-augmented lyrics)",
            value=False,
            help="Retrieve similar Cantopop lyrics from the corpus and inject them as few-shot examples into the prompt.",
        )
        if use_rag:
            from modules.rag_retriever import DEFAULT_CSV_PATH as _DEFAULT_RAG_CSV
            rag_csv_path = st.text_input(
                "Corpus CSV path",
                value=str(_DEFAULT_RAG_CSV),
                help="Path to cantopop_corpus_final_583_yue.csv (defaults to repo root)",
            )
            rag_top_k = st.slider("RAG top-k examples", min_value=1, max_value=6, value=3)
            st.caption("📚 RAG injects the most similar lyrics as few-shot context to improve [verse]/[chorus] structure and style.")

    # st.subheader("Style source")

    style_source = st.radio(
        "Choose genre prompt source",
        [
            "Preset",
            "Select tags from list",
            "Generate by lyrics model",
        ],
        index=0,
        horizontal=False,
    )

    selected_style_tags = {}

    if style_source == "Preset":
        preset_name = st.selectbox(
            "Style preset",
            list(STYLE_PRESETS.keys()),
            index=0,
        )
        style = STYLE_PRESETS[preset_name]
        genre_prompt_mode = "preset"

        st.caption("Preset genre prompt:")
        st.code(style)

    elif style_source == "Select tags from list":
        tag_data = load_top_200_tags()

        st.caption("Mandatory tag: Cantonese")

        for category in TAG_CATEGORIES:
            options = unique_clean_tags(tag_data.get(category, []))

            selected_style_tags[category] = st.multiselect(
                f"{category.title()} tags",
                options=options,
                default=[],
                key=f"style_tags_{category}",
            )

        user_selected_style = build_style_prompt_from_selected_tags(selected_style_tags)

        # Always include Cantonese in tag-list mode.
        style = ensure_mandatory_style_tags(user_selected_style)
        genre_prompt_mode = "tag_list"

        if user_selected_style:
            st.caption("Selected genre prompt:")
            st.code(style)
        else:
            st.warning("Please select at least one style tag. Cantonese will still be included automatically. (It is recommended to include all the 5 components.)")
            st.caption("Current genre prompt:")
            st.code(style)

    else:
        style = ""
        genre_prompt_mode = "generated"
        st.caption("The multimodal lyrics model will generate genre_prompt directly.")
    
    if "line_count" not in st.session_state:
        st.session_state["line_count"] = 8

    if "run_n_segments" not in st.session_state:
        st.session_state["run_n_segments"] = LYRIC_LENGTH_TO_SEGMENTS[
            int(st.session_state["line_count"])
        ]

    user_style_hints = st.text_input("Optional style hints", value="male or female cantopop vocal, emotionally expressive")

    line_count = st.selectbox(
        "Lyric length",
        [4, 8, 16],
        key="line_count",
        on_change=sync_run_n_segments_to_line_count,
    )
    
    mm_temperature = st.number_input("MM temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
    mm_max_new_tokens = st.number_input("MM max_new_tokens", min_value=128, max_value=2048, value=2048, step=64)
    mm_run_on_cpu = st.checkbox("Run multimodal lyrics model on CPU", value=False)

    st.header("Step 4 — Original YuE")
    output_dir = st.text_input("Output dir", value="outputs")
    stage1_model = st.text_input("Stage 1 model", value="m-a-p/YuE-s1-7B-anneal-zh-cot")
    stage2_model = st.text_input("Stage 2 model", value="m-a-p/YuE-s2-1B-general")
    run_n_segments = st.number_input(
        "run_n_segments",
        min_value=1,
        max_value=12,
        step=1,
        key="run_n_segments",
    )
    stage2_batch_size = st.number_input("stage2_batch_size", min_value=1, max_value=32, value=4, step=1)
    yue_max_new_tokens = st.number_input("YuE max_new_tokens", min_value=128, max_value=12000, value=3000, step=64)
    repetition_penalty = st.number_input("repetition_penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.05)
    extra_cli_args = st.text_input("Extra YuE CLI args", value="")

    st.header("ICL (optional)")
    use_single_track_icl = st.checkbox("Use single-track ICL", value=False)
    use_dual_track_icl = st.checkbox("Use dual-track ICL", value=False)
    prompt_start_time = st.number_input("prompt_start_time", min_value=0, max_value=300, value=0, step=1)
    prompt_end_time = st.number_input("prompt_end_time", min_value=1, max_value=300, value=30, step=1)

st.subheader("Step 1 — Upload image")
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key=f"img_{st.session_state.get('uploader_version', 0)}")
if uploaded is not None:
    st.session_state["uploaded_image_bytes"] = uploaded.getvalue()
    st.session_state["uploaded_image_name"] = uploaded.name
    st.session_state["step_1_done"] = True
    st.image(
        Image.open(BytesIO(st.session_state["uploaded_image_bytes"])),
        caption=uploaded.name,
        use_container_width=True,
    )

with st.expander("Optional reference audio", expanded=False):
    single_ref = st.file_uploader("Single-track reference audio (for single-track ICL)", type=["mp3", "wav", "m4a", "flac"], key=f"single_ref_{st.session_state.get('uploader_version', 0)}")
    dual_vocal = st.file_uploader("Dual-track vocal reference", type=["mp3", "wav", "m4a", "flac"], key=f"dual_vocal_{st.session_state.get('uploader_version', 0)}")
    dual_instr = st.file_uploader("Dual-track instrumental reference", type=["mp3", "wav", "m4a", "flac"], key=f"dual_instr_{st.session_state.get('uploader_version', 0)}")

st.subheader("Step 2 — Generate lyrics & prompt")
if debug_mode and st.button("Use example prompt bundle"):
    st.session_state["last_error"] = ""
    st.session_state["last_debug_log"] = ""
    
    reset_evaluation_results()
    
    try:
        st.session_state["lyrics_prompt_raw"] = load_example_prompt_bundle()
        st.session_state["step_2_done"] = True
        st.session_state["step_3_done"] = False
        st.session_state["step_4_done"] = False
    except Exception:
        st.session_state["step_2_done"] = False
        st.session_state["last_error"] = traceback.format_exc()
        st.session_state["last_debug_log"] = st.session_state["last_error"]
    st.rerun()

if st.session_state["step_1_done"]:
    if st.button("Generate Lyrics & Prompt"):
        st.session_state["last_error"] = ""
        st.session_state["last_debug_log"] = ""
        
        # Clear previous evaluation results before generating new lyrics.
        reset_evaluation_results()

        try:
            # 清掉上一次殘留的 multimodal model / CUDA cache
            unload_mm_models()

            st.session_state["lyrics_prompt_raw"] = generate_from_image(
                st.session_state["uploaded_image_bytes"],
                model_id=mm_model,
                style=style,
                line_count=int(line_count),
                temperature=float(mm_temperature),
                max_new_tokens=int(mm_max_new_tokens),
                user_style_hints=user_style_hints,
                run_on_cpu=bool(mm_run_on_cpu),
                hf_token=hf_token,
                use_rag=bool(use_rag),
                rag_csv_path=rag_csv_path,
                rag_top_k=int(rag_top_k),
                genre_prompt_mode=genre_prompt_mode,
            )

            st.session_state["step_2_done"] = True
            st.session_state["step_3_done"] = False
            st.session_state["step_4_done"] = False

        except Exception:
            st.session_state["step_2_done"] = False
            st.session_state["last_error"] = traceback.format_exc()
            st.session_state["last_debug_log"] = st.session_state["last_error"]

        finally:
            # 無論成功或失敗，都釋放歌詞生成模型佔用的顯存
            try:
                unload_mm_models()
            except Exception:
                pass

        st.rerun()
if debug_mode:
    st.info("Upload an image and generate a prompt bundle from the image, "
            "or use the example prompt bundle button above to save run time.")

if st.session_state["last_error"]:
    st.error("Generation failed.")
    if st.button("View debug log", key="view_last_error_debug"):
        show_debug_log(st.session_state["last_debug_log"])

if st.session_state["step_2_done"]:
    rawb = st.session_state["lyrics_prompt_raw"]
    st.subheader("Step 3 — Confirm / edit")
    title = st.text_input("Title", value=rawb.title)
    lyrics_text = st.text_area("Lyrics", value=rawb.lyrics_text, height=260)
    genre_prompt = st.text_area("Genre prompt", value=rawb.genre_prompt)
    
    # metadata from multimodal model, for debugging and evaluation purposes
    # with st.expander("Step 2 metadata", expanded=False):
        # st.write("**Generation metadata from multimodal lyrics model:**")
        # st.json(rawb.raw_meta)

    with st.expander("Evaluation", expanded=False):
        eval_tabs = st.tabs([
            "Image-lyrics alignment (CLIP)",
            "Image-lyrics emotion similarity",
            "Lyrics format",
            "lyrics quality",
        ])
        with eval_tabs[0]:
            description = ("Chinese CLIP (Contrastive Language-Image Pre-Training), "
                "with ViT-B/16 as the image encoder and RoBERTa-wwm-base as the text encoder.")
            st.write(description)
            if st.button("Calculate image-lyrics alignment", key="eval_image_lyrics_alignment"):
                st.session_state["image_lyrics_alignment"] = None
                st.session_state["image_lyrics_alignment_error"] = ""
                try:
                    if not st.session_state.get("uploaded_image_bytes"):
                        raise ValueError("Please upload an image first.")
                    similarity_module = load_image_text_similarity_module()
                    score = similarity_module.score_image_text_similarity(
                        image_bytes=st.session_state["uploaded_image_bytes"],
                        json_input={"lyrics_text": lyrics_text.strip()},
                    )
                    st.session_state["image_lyrics_alignment"] = float(score)
                except Exception:
                    st.session_state["image_lyrics_alignment_error"] = traceback.format_exc()
                st.rerun()

            if st.session_state.get("image_lyrics_alignment") is not None:
                st.success(f"Image-lyrics alignment: {st.session_state['image_lyrics_alignment']:.4f}")
            elif st.session_state.get("image_lyrics_alignment_error"):
                st.error("Image-lyrics alignment calculation failed.")
                if st.button("View debug log", key="view_image_lyrics_alignment_debug"):
                    show_image_lyrics_alignment_debug_log(st.session_state["image_lyrics_alignment_error"])

        with eval_tabs[1]:
            emotion_module = load_image_lyrics_emotion_similarity_module()
            text_emotion_model_options = emotion_module.get_text_emotion_model_keys()

            def format_text_emotion_model(model_key: str) -> str:
                return emotion_module.get_text_emotion_model_display_name(model_key)

            text_emotion_model = st.selectbox(
                "Text emotion model",
                text_emotion_model_options,
                format_func=format_text_emotion_model,
                index=0,
                key="eval_text_emotion_model",
            )
            max_image_labels = emotion_module.get_max_image_emotion_classes("25cat")
            max_text_labels = emotion_module.get_max_text_emotion_classes(text_emotion_model)

            model_def = emotion_module.get_text_emotion_model_def(text_emotion_model)
            caption_text = ("")
            if model_def.get("is_online"):
                caption_text += " Requires environment variable HUGGINGFACE_API_TOKEN."

            st.caption(caption_text)
            top_k_image_eval = st.number_input(
                f"Top-k image emotions (up to {max_image_labels})",
                min_value=1,
                max_value=max_image_labels,
                value=max_image_labels,
                step=1,
                key="eval_top_k_image",
            )
            top_k_text_eval = st.number_input(
                f"Top-k text emotions (up to {max_text_labels})",
                min_value=1,
                max_value=max_text_labels,
                value=max_text_labels,
                step=1,
                key="eval_top_k_text",
            )
            if st.button("Calculate image-lyrics emotion similarity", key="eval_image_lyrics_emotion_similarity"):
                st.session_state["image_lyrics_emotion_similarity"] = None
                st.session_state["image_lyrics_emotion_similarity_error"] = ""
                st.session_state["image_lyrics_emotion_predictions"] = None
                try:
                    if not st.session_state.get("uploaded_image_bytes"):
                        raise ValueError("Please upload an image first.")
                    image = Image.open(BytesIO(st.session_state["uploaded_image_bytes"])).convert("RGB")

                    emotion_module = load_image_lyrics_emotion_similarity_module()
                    results = emotion_module.evaluate_emotion_similarity(
                        image=image,
                        lyrics_text=lyrics_text.strip(),
                        top_k_image=top_k_image_eval,
                        top_k_text=top_k_text_eval,
                        image_model_type="25cat",
                        text_emotion_model=text_emotion_model,
                        embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        embedding_device=None,
                        verbose=False,
                    )
                    st.session_state["image_lyrics_emotion_similarity"] = float(results["similarity"])
                    st.session_state["image_lyrics_emotion_results"] = results
                except Exception:
                    st.session_state["image_lyrics_emotion_similarity_error"] = traceback.format_exc()
                st.rerun()

            if st.session_state.get("image_lyrics_emotion_similarity") is not None:
                st.success(f"Image-lyrics emotion similarity: {st.session_state['image_lyrics_emotion_similarity']:.4f}")
                results = st.session_state.get("image_lyrics_emotion_results") or {}
                image_preds = results.get("image_predictions", [])
                text_preds = results.get("text_predictions", [])
                if image_preds:
                    st.write("**Image emotion predictions:**")
                    for item in image_preds:
                        label = item.get("label")
                        english = item.get("english_label")
                        if english and label and label.strip().lower() != english.strip().lower():
                            st.write(f"- {label} ({english}): {item['score']:.4f}")
                        else:
                            st.write(f"- {label}: {item['score']:.4f}")
                if text_preds:
                    st.write("**Text emotion predictions:**")
                    for item in text_preds:
                        label = item.get("label")
                        english = item.get("english_label")
                        if english and label and label.strip().lower() != english.strip().lower():
                            st.write(f"- {label} ({english}): {item['score']:.4f}")
                        else:
                            st.write(f"- {label}: {item['score']:.4f}")
            elif st.session_state.get("image_lyrics_emotion_similarity_error"):
                st.error("Image-lyrics emotion similarity calculation failed.")
                if st.button("View debug log", key="view_image_lyrics_emotion_debug"):
                    show_image_lyrics_emotion_debug_log(st.session_state["image_lyrics_emotion_similarity_error"])

        with eval_tabs[2]:
            st.write(
                "Hybrid lyrics format evaluation. This checks section tags, blank lines, "
                "and structural similarity against the reference lyric format. It does not judge lyric content quality."
            )

            lyrics_format_reference_path = PROJECT_ROOT / "Evaluation" / "lyrics_format" / "reference_lyrics_format.txt"

            col_rule, col_transformer, col_sequence = st.columns(3)

            with col_rule:
                lyrics_rule_weight = st.number_input(
                    "Rule weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.50,
                    step=0.05,
                    key="eval_lyrics_rule_weight",
                )

            with col_transformer:
                lyrics_transformer_weight = st.number_input(
                    "Transformer weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.30,
                    step=0.05,
                    key="eval_lyrics_transformer_weight",
                )

            with col_sequence:
                lyrics_sequence_weight = st.number_input(
                    "Sequence weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    step=0.05,
                    key="eval_lyrics_sequence_weight",
                )

            lyrics_format_model_name = st.text_input(
                "Transformer model",
                value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                key="eval_lyrics_format_model_name",
            )

            lyrics_format_device = st.selectbox(
                "Device",
                ["auto", "cpu", "cuda"],
                index=0,
                key="eval_lyrics_format_device",
            )

            if st.button("Calculate lyrics format score", key="eval_lyrics_format_score"):
                st.session_state["lyrics_format_score_result"] = None
                st.session_state["lyrics_format_score_error"] = ""

                try:
                    if not lyrics_text.strip():
                        raise ValueError("Lyrics text is empty.")

                    lyrics_format_module = load_lyrics_format_transformer_score_module()

                    structure_desc, reference_lyrics, lyrics_json_example = build_lyrics_format_instruction(
                        int(line_count)
                    )

                    device_arg = None if lyrics_format_device == "auto" else lyrics_format_device

                    result = lyrics_format_module.score_lyrics_format_hybrid(
                        json_input={"lyrics_text": lyrics_text.strip()},
                        reference_lyrics=reference_lyrics,
                        model_name=lyrics_format_model_name.strip(),
                        device=device_arg,
                        rule_weight=float(lyrics_rule_weight),
                        transformer_weight=float(lyrics_transformer_weight),
                        sequence_weight=float(lyrics_sequence_weight),
                        return_details=True,
                    )

                    st.session_state["lyrics_format_score_result"] = result

                except Exception:
                    st.session_state["lyrics_format_score_error"] = traceback.format_exc()

                st.rerun()

            lyrics_format_result = st.session_state.get("lyrics_format_score_result")

            if lyrics_format_result is not None:
                score_value = float(lyrics_format_result.get("lyrics_format_score", 0.0))
                grade = lyrics_format_result.get("grade", "unknown")

                st.success(f"Lyrics format score: {score_value:.2f} / 100 ({grade})")

                warnings = lyrics_format_result.get("warnings", [])
                if warnings:
                    st.warning("Format warnings:")
                    for warning in warnings:
                        st.write(f"- {warning}")
                else:
                    st.info("No format warnings detected.")

                metrics = lyrics_format_result.get("metrics", {})
                if metrics:
                    show_metrics = st.checkbox(
                        "Show lyrics format metrics",
                        value=False,
                        key="show_lyrics_format_metrics",
                    )
                    if show_metrics:
                        st.json(metrics)

            elif st.session_state.get("lyrics_format_score_error"):
                st.error("Lyrics format evaluation failed.")
                if st.button("View debug log", key="view_lyrics_format_debug"):
                    show_lyrics_format_evaluation_debug_log(
                        st.session_state["lyrics_format_score_error"]
                    )
                    
        with eval_tabs[3]:
            st.write(
                "Evaluate Cantonese lyric quality using tonal aesthetics, rhyme consistency, "
                "lexical diversity, structural regularity, semantic coherence, and naturalness."
            )

            st.caption(
                "Section tags such as [verse], [chorus], [bridge], [outro], and [end] "
                "will be removed before quality evaluation."
            )

            if st.button("Calculate Cantonese lyrics quality", key="eval_cantonese_lyrics_quality"):
                st.session_state["cantonese_lyrics_quality_result"] = None
                st.session_state["cantonese_lyrics_quality_error"] = ""

                try:
                    cleaned_lyrics_for_quality = strip_lyrics_section_tags(lyrics_text)

                    if not cleaned_lyrics_for_quality.strip():
                        raise ValueError("Lyrics text is empty after removing section tags.")

                    quality_module = load_cantonese_lyrics_quality_module()

                    result = quality_module.evaluate_cantonese_lyrics(
                        cleaned_lyrics_for_quality
                    )

                    st.session_state["cantonese_lyrics_quality_result"] = result

                except Exception:
                    st.session_state["cantonese_lyrics_quality_error"] = traceback.format_exc()

                st.rerun()

            quality_result = st.session_state.get("cantonese_lyrics_quality_result")

            if quality_result is not None:
                overall = float(quality_result.get("overall", 0.0))
                grade = quality_result.get("grade", "unknown")

                st.success(
                    f"Cantonese lyrics quality: {overall:.4f} / 1.0000 ({grade})"
                )

                scores = quality_result.get("scores", {})

                if scores:
                    st.write("**Metric scores:**")

                    metric_labels = {
                        "tonal": "Tonal aesthetics",
                        "rhyme": "Rhyme consistency",
                        "lexical": "Lexical diversity",
                        "structure": "Structural regularity",
                        "coherence": "Semantic coherence",
                        "natural": "Naturalness",
                    }

                    for key, label in metric_labels.items():
                        value = scores.get(key)

                        if value is None:
                            st.write(f"- {label}: N/A")
                        else:
                            st.write(f"- {label}: {float(value):.4f}")

                suggestions = quality_result.get("suggestions", [])

                if suggestions:
                    st.warning("Suggestions:")
                    for item in suggestions:
                        st.write(f"- {item}")
                else:
                    st.info("No major quality suggestions detected.")

                rhyme_debug = quality_result.get("rhyme_debug", {})

                show_rhyme_debug = st.checkbox(
                    "Show rhyme debug",
                    value=False,
                    key="show_cantonese_rhyme_debug",
                )

                if show_rhyme_debug:
                    st.json(rhyme_debug)

            elif st.session_state.get("cantonese_lyrics_quality_error"):
                st.error("Cantonese lyrics quality evaluation failed.")

                if st.button("View debug log", key="view_cantonese_lyrics_quality_debug"):
                    show_cantonese_lyrics_quality_debug_log(
                        st.session_state["cantonese_lyrics_quality_error"]
                    )


    if st.button("Confirm Lyrics & Prompt"):
        # lyrics_text_clean = normalize_lyrics_format(lyrics_text)

        st.session_state["lyrics_prompt_confirmed"] = LyricsPromptBundle(
            title=title.strip(),
            # lyrics_text=lyrics_text_clean,
            lyrics_text=lyrics_text,
            genre_prompt=genre_prompt.strip(),
            language_tag="Cantonese",
            raw_meta=rawb.raw_meta,
        )
        st.session_state["step_3_done"] = True
        st.session_state["step_4_done"] = False
        st.rerun()

if st.session_state["step_3_done"]:
    st.subheader("Step 4 — Generate with original YuE environment")
    if st.button("Generate Song (original YuE)"):
        st.session_state["last_error"] = ""
        st.session_state["last_debug_log"] = ""
        st.session_state["last_metadata_path"] = str(Path(output_dir) / "song_metadata.json")
        try:
            unload_mm_models()

            tmp_refs = Path(output_dir) / "icl_inputs"
            tmp_refs.mkdir(parents=True, exist_ok=True)

            single_ref_path = ""
            dual_vocal_path = ""
            dual_instr_path = ""

            if single_ref is not None:
                single_ref_path = str((tmp_refs / single_ref.name).resolve())
                Path(single_ref_path).write_bytes(single_ref.getvalue())
            if dual_vocal is not None:
                dual_vocal_path = str((tmp_refs / dual_vocal.name).resolve())
                Path(dual_vocal_path).write_bytes(dual_vocal.getvalue())
            if dual_instr is not None:
                dual_instr_path = str((tmp_refs / dual_instr.name).resolve())
                Path(dual_instr_path).write_bytes(dual_instr.getvalue())

            st.session_state["song_result"] = generate_song_auto(
                st.session_state["lyrics_prompt_confirmed"],
                out_dir=output_dir,
                stage1_model=stage1_model,
                stage2_model=stage2_model,
                run_n_segments=int(run_n_segments),
                stage2_batch_size=int(stage2_batch_size),
                max_new_tokens=int(yue_max_new_tokens),
                repetition_penalty=float(repetition_penalty),
                use_single_track_icl=bool(use_single_track_icl),
                use_dual_track_icl=bool(use_dual_track_icl),
                prompt_audio_path=single_ref_path,
                vocal_prompt_path=dual_vocal_path,
                instrumental_prompt_path=dual_instr_path,
                prompt_start_time=int(prompt_start_time),
                prompt_end_time=int(prompt_end_time),
                extra_cli_args=extra_cli_args,
            )
            st.session_state["step_4_done"] = True
        except Exception:
            st.session_state["step_4_done"] = False
            st.session_state["last_error"] = traceback.format_exc()
            meta_path = Path(st.session_state["last_metadata_path"])
            meta_txt = meta_path.read_text(encoding="utf-8") if meta_path.exists() else ""
            st.session_state["last_debug_log"] = st.session_state["last_error"] + ("\\n\\n=== song_metadata.json ===\\n" + meta_txt if meta_txt else "")
        st.rerun()

if st.session_state["step_4_done"]:
    result = st.session_state["song_result"]
    st.success("YuE song generated.")
    audio_path = Path(result.final_song_path)
    if audio_path.exists():
        st.audio(str(audio_path))
        with open(audio_path, "rb") as f:
            st.download_button("Download final YuE song", f.read(), file_name=audio_path.name)
    if result.generation_log.get("warning"):
        st.warning(result.generation_log["warning"])
    if Path(result.metadata_path).exists():
        with st.expander("YuE metadata", expanded=False):
            st.code(Path(result.metadata_path).read_text(encoding="utf-8"), language="json")
