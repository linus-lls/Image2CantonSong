from __future__ import annotations
import gc
import json
import re
import subprocess
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModel, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from schemas import LyricsPromptBundle

# ── Set this to your HF Hub adapter repo ID after fine-tuning ─────────────
# e.g. "your-hf-username/internvl2-4b-cantopop-lora"
# Leave as None to use the base InternVL2-4B without the adapter.
INTERNVL_ADAPTER_ID: str | None = None  # LoRA adapter not ready yet; using base model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Ensure repo root is in path for imports
from paths import PROJECT_ROOT

_PROCESSOR_CACHE: Dict[str, object] = {}
_MODEL_CACHE: Dict[tuple, object] = {}

def _torch():
    import torch
    return torch

def unload_mm_models():
    torch = _torch()
    global _MODEL_CACHE
    for model in _MODEL_CACHE.values():
        try:
            model.to("cpu")
        except Exception:
            pass
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _norm_device(run_on_cpu: bool) -> str:
    torch = _torch()
    return "cpu" if run_on_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

def _is_internvl(model_id: str) -> bool:
    return "InternVL" in model_id or "internvl" in model_id.lower()


def _load_model(model_id: str, run_on_cpu: bool):
    import torch
    device = _norm_device(run_on_cpu)
    cache_key = (model_id, device)

    if _is_internvl(model_id):
        # InternVL2 uses AutoTokenizer + AutoModel with trust_remote_code
        if model_id not in _PROCESSOR_CACHE:
            tokenizer_src = INTERNVL_ADAPTER_ID if INTERNVL_ADAPTER_ID else model_id
            _PROCESSOR_CACHE[model_id] = AutoTokenizer.from_pretrained(
                tokenizer_src, trust_remote_code=True, use_fast=False
            )
        if cache_key not in _MODEL_CACHE:
            kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
            if device == "cuda":
                kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float32
            base_model = AutoModel.from_pretrained(model_id, **kwargs).to(device).eval()
            if INTERNVL_ADAPTER_ID:
                # Load fine-tuned LoRA adapter from HF Hub
                model = PeftModel.from_pretrained(base_model, INTERNVL_ADAPTER_ID)
                model = model.eval()
            else:
                model = base_model
            _MODEL_CACHE[cache_key] = model
    else:
        # Qwen2.5-VL uses AutoProcessor + Qwen2_5_VLForConditionalGeneration
        if model_id not in _PROCESSOR_CACHE:
            _PROCESSOR_CACHE[model_id] = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, use_fast=True
            )
        if cache_key not in _MODEL_CACHE:
            kwargs = {"trust_remote_code": True}
            if device == "cuda":
                kwargs["torch_dtype"] = torch.float16
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
            model = model.to(device)
            _MODEL_CACHE[cache_key] = model

    return _PROCESSOR_CACHE[model_id], _MODEL_CACHE[cache_key], device

def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))
    if "assistant" in text:
        text = text.split("assistant")[-1].strip()
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(starts):
        candidate = text[start:].strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No valid JSON object found. Raw tail:\\n{text[-3000:]}")

def generate_clip_e_mood(image: Image.Image) -> str:
    """Infer a top-2 mood label from the image using clip-e-ce.py."""
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    binary_image = image_bytes.getvalue()

    script_path = PROJECT_ROOT / "Emotion" / "CLIP-E" / "clip-e-ce.py"
    cmd = (
        "conda activate clip-e && "
        f" python {script_path} --stdin-bytes --model-type 25cat --top-n 2 --mood-only && "
        "conda deactivate"
    )
    proc = subprocess.run(
        ["bash", "-lic", cmd],
        input=binary_image,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        return "情緒模糊"

    lines = [line.strip() for line in proc.stdout.decode("utf-8", errors="ignore").splitlines() if line.strip()]
    labels = []
    for line in lines:
        if ":" in line:
            labels.append(line.split(":", 1)[0].strip())
        elif line.lower().startswith("top "):
            continue
        else:
            labels.append(line)
        if len(labels) >= 2:
            break
    return ", ".join(labels) if labels else "情緒模糊"

def generate_prompt(
    image: Image.Image,
    style: str,
    line_count: int = 8,
    user_style_hints: str = "",
    rag_few_shot_block: str = "",
) -> str:
    """Return the formatted prompt text for the multimodal model."""
    mood_text = generate_clip_e_mood(image)
    style_hint = user_style_hints.strip() or style or "無"
    rag_section = f"\n{rag_few_shot_block}\n" if rag_few_shot_block else ""
    return f"""{rag_section}
你是一個粵語流行歌作詞與音樂企劃助手。請直接觀看這張圖片，輸出一份基於圖片內容的歌曲方案。

硬性要求：
1. 歌詞必須使用繁體中文，適合香港粵語演唱。
2. 歌詞要明確呼應圖片中的人物、場景、動作、道具與氛圍。
3. 不要求口語化，但不能寫成明顯普通話朗讀腔。
4. 歌詞總長約 {line_count} 行，必須分成 [verse] 和 [chorus]。
5. 只允許輸出一個 JSON 物件，不要輸出任何額外文字。

JSON 格式如下：
{{
  "visual_anchor": "...",
  "title": "...",
  "lyrics_text": "[verse]\\n...\\n\\n[chorus]\\n...",
  "genre_prompt": "cantopop ballad piano guitar sentimental female airy vocal Cantonese",
  "music_prompt": "melodic singing, full accompaniment, expressive chorus, not narration",
  "negative_prompt": "Mandarin pronunciation, spoken word, narration, recitation, monotone",
  "bpm": 84,
  "key": "F major"
}}

補充風格提示：
{style_hint}

The image is likely of mood {mood_text}.
""".strip()


def generate_from_image(
    image_bytes: bytes,
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    style: str = "cantopop-ballad",
    line_count: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 448,
    user_style_hints: str = "",
    run_on_cpu: bool = False,
    hf_token: str | None = None,
    use_rag: bool = False,
    rag_csv_path: str = "",
    rag_top_k: int = 3,
) -> LyricsPromptBundle:
    torch = _torch()
    # Log in to HF Hub if token provided (needed for private adapter repos)
    if hf_token:
        try:
            from huggingface_hub import login as _hf_login
            _hf_login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1024, 1024))

    # ── RAG: retrieve similar lyrics and inject as few-shot context ──────
    rag_few_shot_block = ""
    if use_rag and _is_internvl(model_id):
        try:
            from modules.rag_retriever import init as _rag_init, build_few_shot_block
            _rag_init(rag_csv_path or None)
            rag_query = (user_style_hints.strip() or style or "cantopop ballad 粵語")
            rag_few_shot_block = build_few_shot_block(rag_query, top_k=rag_top_k)
        except Exception as _rag_err:
            warnings.warn(f"RAG retrieval failed, falling back to base prompt: {_rag_err}")

    prompt = generate_prompt(
        image, style,
        line_count=line_count,
        user_style_hints=user_style_hints,
        rag_few_shot_block=rag_few_shot_block,
    )
    processor, model, device = _load_model(model_id, run_on_cpu)

    if _is_internvl(model_id):
        # InternVL2 inference path
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        pixel_values = transform(image).unsqueeze(0).to(
            device, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        generation_config = dict(
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        question = f"<image>\n{prompt}"
        with torch.no_grad():
            decoded = model.chat(processor, pixel_values, question, generation_config)
    else:
        # Qwen2.5-VL inference path
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_len:]
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    try:
        payload = _extract_json(decoded)
    except Exception as e:
        raise RuntimeError(f"Multimodal LLM did not return valid JSON. Raw tail:\\n{decoded[-5000:]}") from e
    finally:
        if not _is_internvl(model_id):
            del inputs, outputs
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["genre_prompt"] = "cantopop ballad piano guitar sentimental female airy vocal Cantonese"
    payload["music_prompt"] = "melodic singing, full accompaniment, expressive chorus, not narration"
    payload["negative_prompt"] = "Mandarin pronunciation, spoken word, narration, recitation, monotone"

    lyrics_text = str(payload.get("lyrics_text", "")).strip()
    if "[verse]" not in lyrics_text.lower() or "[chorus]" not in lyrics_text.lower():
        warnings.warn(
            f"lyrics_text is missing [verse] or [chorus]. Output may be malformed. Raw tail:\\n{decoded[-3000:]}",
            UserWarning,
            stacklevel=2,
        )

    return LyricsPromptBundle(
        title=str(payload.get("title", "")).strip(),
        lyrics_text=lyrics_text,
        genre_prompt=str(payload.get("genre_prompt", "")).strip(),
        music_prompt=str(payload.get("music_prompt", "")).strip(),
        negative_prompt=str(payload.get("negative_prompt", "")).strip(),
        bpm=int(payload.get("bpm", 84)),
        key=str(payload.get("key", "F major")).strip(),
        language_tag="Cantonese",
        raw_meta={
            "visual_anchor": str(payload.get("visual_anchor", "")).strip(),
            "llm_backend": model_id,
            "device": device,
            "raw_generation_tail": decoded[-5000:],
        },
    )
