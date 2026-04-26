from __future__ import annotations
import gc
import json
import re
import warnings
from io import BytesIO
from typing import Dict, Any
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from schemas import LyricsPromptBundle

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

def _load_model(model_id: str, run_on_cpu: bool):
    torch = _torch()
    device = _norm_device(run_on_cpu)
    if model_id not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[model_id] = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
        )
    cache_key = (model_id, device)
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

def generate_from_image(
    image_bytes: bytes,
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    style: str = "cantopop-ballad",
    line_count: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 448,
    user_style_hints: str = "",
    run_on_cpu: bool = False,
) -> LyricsPromptBundle:
    torch = _torch()
    processor, model, device = _load_model(model_id, run_on_cpu)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((1024, 1024))

    prompt = f"""
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
{user_style_hints or "無"}
""".strip()

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
