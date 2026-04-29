from __future__ import annotations
from io import BytesIO
from pathlib import Path
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
from modules.mm_direct_gen import unload_mm_models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Ensure repo root is in path for imports
from paths import PROJECT_ROOT, DEMO, EVAL

st.set_page_config(page_title="Project Demo — Official YuE Bridge", page_icon="🎵", layout="wide")
init_state()

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--debug", action="store_true")
args, _ = parser.parse_known_args()
debug_mode = args.debug

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
        music_prompt=raw.get("music_prompt", ""),
        negative_prompt=raw.get("negative_prompt", ""),
        bpm=int(raw.get("bpm", 84)),
        key=raw.get("key", ""),
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

if debug_mode:
    st.info("Debug mode enabled.")

st.title("🎵 Project Demo — Official YuE Bridge")
st.caption("Image → multimodal lyrics draft → manual confirm → final generation by original YuE environment")

with st.sidebar:
    if st.button("Reset app state"):
        hard_reset()
        st.rerun()

    st.header("Step 2 — Multimodal LLM")
    mm_model = st.text_input("Image-capable HF LLM", value="Qwen/Qwen2.5-VL-3B-Instruct")
    style = st.selectbox("Style preset", ["cantopop-ballad", "city-pop", "dream-pop"], index=0)
    line_count = st.selectbox("Lyric length", [4, 8], index=1)
    mm_temperature = st.number_input("MM temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
    mm_max_new_tokens = st.number_input("MM max_new_tokens", min_value=128, max_value=2048, value=2048, step=64)
    mm_run_on_cpu = st.checkbox("Run multimodal lyrics model on CPU", value=False)
    user_style_hints = st.text_input("Optional style hints", value="male or female cantopop vocal, emotionally expressive")

    st.header("Step 4 — Original YuE")
    output_dir = st.text_input("Output dir", value="outputs")
    stage1_model = st.text_input("Stage 1 model", value="m-a-p/YuE-s1-7B-anneal-zh-cot")
    stage2_model = st.text_input("Stage 2 model", value="m-a-p/YuE-s2-1B-general")
    run_n_segments = st.number_input("run_n_segments", min_value=1, max_value=12, value=3, step=1)
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

st.subheader("Optional reference audio")
single_ref = st.file_uploader("Single-track reference audio (for single-track ICL)", type=["mp3", "wav", "m4a", "flac"], key=f"single_ref_{st.session_state.get('uploader_version', 0)}")
dual_vocal = st.file_uploader("Dual-track vocal reference", type=["mp3", "wav", "m4a", "flac"], key=f"dual_vocal_{st.session_state.get('uploader_version', 0)}")
dual_instr = st.file_uploader("Dual-track instrumental reference", type=["mp3", "wav", "m4a", "flac"], key=f"dual_instr_{st.session_state.get('uploader_version', 0)}")

st.subheader("Step 2 — Generate lyrics & prompt")
if debug_mode and st.button("Use example prompt bundle"):
    st.session_state["last_error"] = ""
    st.session_state["last_debug_log"] = ""
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
        try:
            st.session_state["lyrics_prompt_raw"] = generate_from_image(
                st.session_state["uploaded_image_bytes"],
                model_id=mm_model,
                style=style,
                line_count=int(line_count),
                temperature=float(mm_temperature),
                max_new_tokens=int(mm_max_new_tokens),
                user_style_hints=user_style_hints,
                run_on_cpu=bool(mm_run_on_cpu),
            )
            st.session_state["step_2_done"] = True
            st.session_state["step_3_done"] = False
            st.session_state["step_4_done"] = False
        except Exception:
            st.session_state["step_2_done"] = False
            st.session_state["last_error"] = traceback.format_exc()
            st.session_state["last_debug_log"] = st.session_state["last_error"]
        st.rerun()
elif debug_mode:
    st.info("Upload an image to generate a prompt bundle from the image, "
            "or use the example prompt bundle button above to save run time.")

if st.session_state["last_error"]:
    st.error("Generation failed.")
    with st.expander("Debug log", expanded=True):
        st.code(st.session_state["last_debug_log"], language="python")

if st.session_state["step_2_done"]:
    rawb = st.session_state["lyrics_prompt_raw"]
    st.subheader("Step 3 — Confirm / edit")
    title = st.text_input("Title", value=rawb.title)
    lyrics_text = st.text_area("Lyrics", value=rawb.lyrics_text, height=260)
    genre_prompt = st.text_area("Genre prompt", value=rawb.genre_prompt)
    music_prompt = st.text_area("Music prompt", value=rawb.music_prompt, height=120)
    negative_prompt = st.text_area("Negative prompt", value=rawb.negative_prompt, height=100)
    bpm = st.number_input("BPM", min_value=40, max_value=180, value=int(rawb.bpm), step=1)
    musical_key = st.text_input("Key", value=rawb.key)

    if st.button("Calculate image-lyrics similarity"):
        st.session_state["image_lyrics_similarity"] = None
        st.session_state["image_lyrics_similarity_error"] = ""
        try:
            if not st.session_state.get("uploaded_image_bytes"):
                raise ValueError("Please upload an image first.")
            similarity_module = load_image_text_similarity_module()
            score = similarity_module.score_image_text_similarity(
                image_bytes=st.session_state["uploaded_image_bytes"],
                json_input={"lyrics_text": lyrics_text.strip()},
            )
            st.session_state["image_lyrics_similarity"] = float(score)
        except Exception:
            st.session_state["image_lyrics_similarity_error"] = traceback.format_exc()
        st.rerun()

    if st.session_state.get("image_lyrics_similarity") is not None:
        st.success(f"Image-lyrics similarity: {st.session_state['image_lyrics_similarity']:.4f}")
    elif st.session_state.get("image_lyrics_similarity_error"):
        st.error("Image-lyrics similarity calculation failed.")
        with st.expander("Debug log", expanded=False):
            st.code(st.session_state["image_lyrics_similarity_error"], language="python")

    if st.button("Confirm Lyrics & Prompt"):
        st.session_state["lyrics_prompt_confirmed"] = LyricsPromptBundle(
            title=title.strip(),
            lyrics_text=lyrics_text.strip(),
            genre_prompt=genre_prompt.strip(),
            music_prompt=music_prompt.strip(),
            negative_prompt=negative_prompt.strip(),
            bpm=int(bpm),
            key=musical_key.strip(),
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
