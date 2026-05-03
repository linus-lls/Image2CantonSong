from __future__ import annotations
import ast
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

def unload_mm_models(clear_processor: bool = False):
    """Unload multimodal models and release CUDA memory."""
    torch = _torch()

    global _MODEL_CACHE
    global _PROCESSOR_CACHE

    for model in list(_MODEL_CACHE.values()):
        try:
            model.to("cpu")
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass

    _MODEL_CACHE.clear()

    if clear_processor:
        _PROCESSOR_CACHE.clear()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

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

def _repair_json(text: str) -> str:
    """Apply heuristic fixes for common model JSON-formatting errors."""
    # Remove spurious lone `},` lines that appear between key-value pairs.
    # Pattern: a closing quote of a value, then a line with just `},`, then the next key.
    # e.g.  "last lyric line"\n  },\n  "genre_prompt"  →  "last lyric line",\n  "genre_prompt"
    text = re.sub(r'("\s*)\n(\s*\},\s*\n\s*")', lambda m: m.group(1) + ',\n' + m.group(2).lstrip().lstrip('},').lstrip(), text)
    # Simpler pass: remove any line that is exactly `},` (with optional spaces) between two object entries
    text = re.sub(r'("[ \t]*)\n([ \t]*)\},[ \t]*\n([ \t]*")', r'\1,\n\3', text)
    return text

def _extract_json(text: str) -> Dict[str, Any]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        return json.loads(m.group(1))
    if "assistant" in text:
        text = text.split("assistant")[-1].strip()
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(starts):
        candidate = text[start:].strip()
        for repair in (lambda s: s, _repair_json):
            c = repair(candidate)
            # Try as-is
            try:
                return json.loads(c)
            except json.JSONDecodeError:
                pass
            # Try repairing truncated JSON by closing open brackets/quotes
            for suffix in ('"}', '"}\n}', '}', '\n}'):
                try:
                    return json.loads(c + suffix)
                except json.JSONDecodeError:
                    pass
    raise ValueError(f"No valid JSON object found. Raw tail:\\n{text[-3000:]}")

def _coerce_lyrics_text(value: Any) -> str:
    """
    Convert model-produced lyrics_text into a clean multiline string.

    Handles:
    - normal string
    - list of strings
    - stringified list, e.g. "['[verse]\\n...']"
    - double-escaped newlines, e.g. "\\n"
    """

    if value is None:
        return ""

    # Case 1: model returns ["..."] or ["line1", "line2"]
    if isinstance(value, list):
        parts = []
        for item in value:
            if item is None:
                continue
            parts.append(str(item))
        text = "\n".join(parts)

    # Case 2: model returns {"text": "..."} or similar
    elif isinstance(value, dict):
        for key in ("lyrics_text", "lyrics", "text", "content"):
            if key in value:
                return _coerce_lyrics_text(value[key])
        text = str(value)

    # Case 3: normal string, or stringified list
    else:
        text = str(value).strip()

        # Handle stringified list: "['[verse]\\n...']"
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return _coerce_lyrics_text(parsed)
            except Exception:
                pass

    # Convert literal backslash-n into real newline.
    # This fixes text like "[verse]\\n一句歌詞\\n\\n[chorus]..."
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")

    # Remove accidental surrounding quotes.
    text = text.strip()
    if len(text) >= 2 and (
        (text[0] == text[-1] == '"') or
        (text[0] == text[-1] == "'")
    ):
        text = text[1:-1].strip()

    return text


def normalize_lyrics_format(lyrics_text: str) -> str:
    """
    Normalize lyrics into displayable multiline format.
    This does not rewrite lyric content; it only repairs tags and spacing.
    """

    lyrics_text = _coerce_lyrics_text(lyrics_text)
    lyrics_text = lyrics_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Normalize supported section tags.
    tag_map = {
        "verse": "[verse]",
        "chorus": "[chorus]",
        "bridge": "[bridge]",
        "outro": "[outro]",
        "end": "[end]",
    }

    for tag, standard in tag_map.items():
        lyrics_text = re.sub(
            rf"^\s*\[{tag}\]\s*$",
            standard,
            lyrics_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    # Remove blank line immediately after section tag.
    lyrics_text = re.sub(
        r"(\[(?:verse|chorus|bridge|outro)\])\n\s*\n+",
        r"\1\n",
        lyrics_text,
        flags=re.IGNORECASE,
    )

    # Compress 3+ newlines into exactly one blank line.
    lyrics_text = re.sub(r"\n{3,}", "\n\n", lyrics_text)

    # If [end] exists, remove anything after it.
    end_match = re.search(r"^\[end\]\s*$", lyrics_text, flags=re.IGNORECASE | re.MULTILINE)
    if end_match:
        lyrics_text = lyrics_text[:end_match.end()]
    else:
        lyrics_text = lyrics_text.rstrip() + "\n\n[end]"

    return lyrics_text.strip()

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


def build_lyrics_format_instruction(line_count: int) -> tuple[str, str, str]:
    """
    Return lyrics format instruction and JSON example according to line_count.

    Rules:
    - 4 lines: verse + chorus + end
    - 8 lines: verse + chorus + end
    - 16 lines: verse + chorus + bridge + outro + end
    """

    if line_count <= 4:
        structure_desc = (
            "歌詞必須分成 [verse] 與 [chorus] 兩段，"
            "每段 2 行，共 4 行歌詞，最後以 [end] 結束。"
        )

        format_block = """[verse]
第一行歌詞
第二行歌詞

[chorus]
第三行歌詞
第四行歌詞

[end]"""

        json_example = (
            "[verse]\\n第一行繁體粵語歌詞\\n第二行繁體粵語歌詞"
            "\\n\\n[chorus]\\n第三行繁體粵語歌詞\\n第四行繁體粵語歌詞"
            "\\n\\n[end]"
        )

    elif line_count <= 8:
        structure_desc = (
            "歌詞必須分成 [verse] 與 [chorus] 兩段，"
            "每段 4 行，共 8 行歌詞，最後以 [end] 結束。"
        )

        format_block = """[verse]
第一行歌詞
第二行歌詞
第三行歌詞
第四行歌詞

[chorus]
第五行歌詞
第六行歌詞
第七行歌詞
第八行歌詞

[end]"""

        json_example = (
            "[verse]\\n第一行繁體粵語歌詞\\n第二行繁體粵語歌詞\\n第三行繁體粵語歌詞\\n第四行繁體粵語歌詞"
            "\\n\\n[chorus]\\n第五行繁體粵語歌詞\\n第六行繁體粵語歌詞\\n第七行繁體粵語歌詞\\n第八行繁體粵語歌詞"
            "\\n\\n[end]"
        )

    else:
        structure_desc = (
            "歌詞必須分成 [verse]、[chorus]、[bridge]、[outro] 四段，"
            "每段 4 行，共 16 行歌詞，最後以 [end] 結束。"
        )

        format_block = """[verse]
第一行歌詞
第二行歌詞
第三行歌詞
第四行歌詞

[chorus]
第五行歌詞
第六行歌詞
第七行歌詞
第八行歌詞

[bridge]
第九行歌詞
第十行歌詞
第十一行歌詞
第十二行歌詞

[outro]
第十三行歌詞
第十四行歌詞
第十五行歌詞
第十六行歌詞

[end]"""

        json_example = (
            "[verse]\\n第一行繁體粵語歌詞\\n第二行繁體粵語歌詞\\n第三行繁體粵語歌詞\\n第四行繁體粵語歌詞"
            "\\n\\n[chorus]\\n第五行繁體粵語歌詞\\n第六行繁體粵語歌詞\\n第七行繁體粵語歌詞\\n第八行繁體粵語歌詞"
            "\\n\\n[bridge]\\n第九行繁體粵語歌詞\\n第十行繁體粵語歌詞\\n第十一行繁體粵語歌詞\\n第十二行繁體粵語歌詞"
            "\\n\\n[outro]\\n第十三行繁體粵語歌詞\\n第十四行繁體粵語歌詞\\n第十五行繁體粵語歌詞\\n第十六行繁體粵語歌詞"
            "\\n\\n[end]"
        )

    return structure_desc, format_block, json_example


def generate_prompt(
    image: Image.Image,
    style: str,
    line_count: int = 8,
    user_style_hints: str = "",
    rag_few_shot_block: str = "",
    genre_prompt_mode: str = "generated",
) -> str:
    """Return the formatted prompt text for the multimodal model."""
    mood_text = generate_clip_e_mood(image)
    style_hint = user_style_hints.strip() or "無"
    style_prompt = style.strip()
    rag_section = f"\n{rag_few_shot_block}\n" if rag_few_shot_block else ""

    structure_desc, format_block, lyrics_json_example = build_lyrics_format_instruction(line_count)

    if genre_prompt_mode in {"preset", "tag_list"}:
        genre_prompt_instruction = (
            f'genre_prompt 必須完全等於："{style_prompt}"。'
            "不得改寫、不得翻譯、不得新增或刪除 tag。"
        )
        genre_prompt_example = style_prompt
    else:
        genre_prompt_instruction = (
            "genre_prompt 必須由你根據圖片、圖片情緒、歌曲標題和歌詞內容自行生成，"
            "不得留空，不得省略。"
            "genre_prompt 必須是一串英文 music tags，至少包含 8 個 tag，"
            "必須包含以下類型：vocal gender、language、mood、genre、instrument、timbre。"
            "例如：male Cantonese dramatic rock piano guitar dark vocal cinematic"
        )
        genre_prompt_example = (
            "female Cantonese melancholic pop piano airy vocal nostalgic"
        )

    return f"""{rag_section}
你是一位香港粵語流行歌作詞助手。請直接觀看這張圖片，輸出一個完整的 JSON 物件（不得截斷）。

【強制要求】
1. 所有中文必須使用繁體字（Traditional Chinese），嚴禁使用任何簡體字。
2. 歌詞語言為書面粵語，適合香港人演唱，不得使用普通話用語。
3. 歌詞須呼應圖片的人物、場景與氛圍。
4. {structure_desc}
5. lyrics_text 必須嚴格遵守指定歌詞格式。
6. 只輸出以下 JSON 物件，不得加入任何其他文字、解釋或 markdown。
7. JSON 必須完整，所有括號均須閉合。
8. JSON 字串中的換行必須使用 \\n 表示。
9. lyrics_text 的值必須是單一 JSON 字串，不得是 array/list，不得使用 [] 包住整段歌詞。
10. 不得把 lyrics_text 輸出成 Python list 字串，例如不得輸出 ["[verse]\\n..."] 或 ['[verse]\\n...']。
11. JSON 必須包含四個字段：visual_anchor、title、lyrics_text、genre_prompt，缺一不可。
12. genre_prompt 不得為空字串，不得省略，必須輸出英文 music tags。
13. {genre_prompt_instruction}

【lyrics_text 標準格式】
lyrics_text 必須嚴格使用以下結構：

{format_block}

【lyrics_text 格式細則】
1. 所有出現的 section 標識必須單獨成行。
2. 每個 section 標識後面必須直接接歌詞，不能有空行。
3. 不同 section 之間必須有且只有一個空行。
4. section 內部不能出現空行。
5. [end] 必須是 lyrics_text 的最後一個非空行。
6. [end] 後面不得再有任何歌詞或文字。
7. 4 句或 8 句短歌詞不得強行加入 [bridge] 或 [outro]。
8. 只有 16 句完整歌詞才使用 [bridge] 和 [outro]。
9. 不得輸出 [Verse]、[CHORUS]、【verse】、Verse:、verse: 等其他標識格式。
10. 不得使用 bullet points、編號、markdown 或解釋性文字。
11. 錯誤示例：["[verse]\\n第一行歌詞\\n第二行歌詞\\n\\n[chorus]\\n第三行歌詞\\n第四行歌詞\\n\\n[end]"]。
12. 正確示例："[verse]\\n第一行歌詞\\n第二行歌詞\\n\\n[chorus]\\n第三行歌詞\\n第四行歌詞\\n\\n[end]"。

【genre_prompt 格式細則】
1. genre_prompt 必須是一個英文 tag 字串。
2. 不得輸出 list、array、JSON object 或 bullet points。
3. 不得加入解釋文字。
4. 多個 tag 之間使用空格分隔。
5. 如果使用 preset 或 tag list，必須完全保留指定 tag，不得改寫。

請嚴格輸出以下 JSON 格式：

{{
  "visual_anchor": "用繁體中文描述圖片主要視覺元素",
  "title": "繁體中文歌名",
  "genre_prompt": "{genre_prompt_example}",
  "lyrics_text": "{lyrics_json_example}"
}}

補充風格提示：{style_hint}
圖片情緒：{mood_text}
""".strip()


def generate_from_image(
    image_bytes: bytes,
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    style: str = "",
    line_count: int = 8,
    temperature: float = 0.7,
    user_style_hints: str = "",
    run_on_cpu: bool = False,
    hf_token: str | None = None,
    use_rag: bool = False,
    rag_csv_path: str = "",
    rag_top_k: int = 3,
    max_new_tokens: int = 1024,
    genre_prompt_mode: str = "generated",
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
            rag_query = (style.strip() or "cantopop ballad 粵語")
            rag_few_shot_block = build_few_shot_block(rag_query, top_k=rag_top_k)
        except Exception as _rag_err:
            warnings.warn(f"RAG retrieval failed, falling back to base prompt: {_rag_err}")

    prompt = generate_prompt(
        image,
        style,
        line_count=line_count,
        user_style_hints=user_style_hints,
        rag_few_shot_block=rag_few_shot_block,
        genre_prompt_mode=genre_prompt_mode,
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
        # Prepend system instruction into the question (InternVL2 chat may or may not accept system_message kwarg)
        system_prefix = "【指令】你是香港粵語歌詞創作助手。所有中文必須使用繁體字。只輸出完整 JSON 物件，不得截斷，不得加入其他文字。\n\n"
        question = f"<image>\n{system_prefix}{prompt}"
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
        for name in [
            "inputs",
            "outputs",
            "generated_ids",
            "pixel_values",
        ]:
            if name in locals():
                try:
                    del locals()[name]
                except Exception:
                    pass

        gc.collect()

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    # payload["genre_prompt"] = "female Cantonese Melancholic Classical airy vocal Piano bright vocal Pop Nostalgic Violin"
    if genre_prompt_mode in {"preset", "tag_list"}:
        payload["genre_prompt"] = style.strip()
    else:
        payload["genre_prompt"] = str(payload.get("genre_prompt", "")).strip()

    lyrics_text = normalize_lyrics_format(payload.get("lyrics_text", ""))

    raw_genre_prompt = payload.get("genre_prompt", "")

    if isinstance(raw_genre_prompt, list):
        raw_genre_prompt = " ".join(str(x).strip() for x in raw_genre_prompt if str(x).strip())
    else:
        raw_genre_prompt = str(raw_genre_prompt).strip()

    if genre_prompt_mode in {"preset", "tag_list"}:
        final_genre_prompt = style.strip()
    else:
        final_genre_prompt = raw_genre_prompt

        # Fallback: avoid empty genre_prompt in Step 3
        if not final_genre_prompt:
            final_genre_prompt = (
                "female Cantonese Melancholic Classical airy vocal "
                "Piano bright vocal Pop Nostalgic Violin"
            )

    if "[verse]" not in lyrics_text.lower() or "[chorus]" not in lyrics_text.lower():
        warnings.warn(
            f"lyrics_text is missing [verse] or [chorus]. Output may be malformed. Raw tail:\\n{decoded[-3000:]}",
            UserWarning,
            stacklevel=2,
        )

    bundle = LyricsPromptBundle(
        title=str(payload.get("title", "")).strip(),
        lyrics_text=lyrics_text,
        genre_prompt=final_genre_prompt,
        language_tag="Cantonese",
        raw_meta={
            "visual_anchor": str(payload.get("visual_anchor", "")).strip(),
            "llm_backend": model_id,
            "device": device,

            # Step 2 style debug metadata
            "genre_prompt_mode": genre_prompt_mode,
            "style_prompt_input": style.strip(),
            "raw_genre_prompt_from_model": raw_genre_prompt,
            "final_genre_prompt": final_genre_prompt,
            "payload_keys": list(payload.keys()),

            # Raw debug
            "raw_payload": payload,
            "raw_generation_tail": decoded[-5000:],
        },
    )

    # Release multimodal model memory after lyrics generation.
    unload_mm_models(clear_processor=False)

    return bundle
