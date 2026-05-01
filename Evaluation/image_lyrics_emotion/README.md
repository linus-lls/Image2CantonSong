# Image-Lyrics Emotion Similarity — `llm-online` Usage

This document explains how to use the new online Hugging Face LLM-backed text emotion predictor (`llm-online`) with the existing image-lyrics emotion similarity tooling.

Prerequisites
- Set up a Hugging Face API token (either `HF_TOKEN` or `HUGGINGFACE_API_TOKEN`).
- Choose a chat-capable model (example: `zai-org/GLM-5:novita`) and set it in `HUGGINGFACE_LLM_MODEL`.

Environment variables
- `HF_TOKEN` (preferred): your Hugging Face API key.
- `HUGGINGFACE_API_TOKEN` (fallback): alternate env var name accepted by the code.
- `HUGGINGFACE_LLM_MODEL`: model id to call, e.g. `zai-org/GLM-5:novita`.

Examples

Export token and model (bash):

```bash
export HF_TOKEN="your_hf_token_here"
export HUGGINGFACE_LLM_MODEL="zai-org/GLM-5:novita"
```

Run the evaluation script with the online LLM text emotion model:

```bash
python Evaluation/image_lyrics_emotion/image_lyrics_emotion_similarity.py \
  --image-path Images/caption.jpg \
  --text "A calm lakeside scene with falling leaves" \
  --text-emotion-model llm-online \
  --top-k-image 5 \
  --top-k-text 5 \
  --image-model-type 25cat \
  --embedding-model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

Or use a lyrics file:

```bash
python Evaluation/image_lyrics_emotion/image_lyrics_emotion_similarity.py \
  --image-path Images/caption.jpg \
  --text-file my_prompt/lyrics.txt \
  --text-emotion-model llm-online
```

Streamlit (UI)
- The Streamlit demo at `canto_project_official_yue_bridge_demo_v2/app.py` exposes `llm-online` in the "Text emotion model" dropdown.
- If `llm-online` is selected the UI caption notes that `HF_TOKEN`/`HUGGINGFACE_LLM_MODEL` must be set.

Notes
- The code reads the token from `HF_TOKEN` first, then `HUGGINGFACE_API_TOKEN` as a fallback.
- The LLM pipeline uses a three-stage, hierarchical prompt flow (emotional reading -> controlled mapping -> strict JSON output). The implementation is in `Emotion/Text2Emotion/llm_text_emotion.py`.
- `--top-k-text` controls how many text emotion labels are used when building the weighted emotion vector.

Files changed
- [Evaluation/image_lyrics_emotion/image_lyrics_emotion_similarity.py](Evaluation/image_lyrics_emotion/image_lyrics_emotion_similarity.py)
- [Emotion/Text2Emotion/llm_text_emotion.py](Emotion/Text2Emotion/llm_text_emotion.py)
- [canto_project_official_yue_bridge_demo_v2/app.py](canto_project_official_yue_bridge_demo_v2/app.py)

If you want, I can also add a short `examples/` script that demonstrates a minimal end-to-end run purely for text emotion outputs (without image embedding steps).