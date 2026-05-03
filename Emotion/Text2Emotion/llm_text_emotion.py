import json
import os
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

# TODO: Use info from the app instead of requiring environment variables.
HUGGINGFACE_API_TOKEN_ENV = "HF_TOKEN"
HUGGINGFACE_API_TOKEN_ALT_ENV = "HUGGINGFACE_API_TOKEN"
HUGGINGFACE_LLM_MODEL_ENV = "HUGGINGFACE_LLM_MODEL"

EMOTION_DEFINITIONS = """
affection: warm, caring, gentle love, not sexual
cheerfullness: light, upbeat happiness, playful joy
confusion: emotional uncertainty, being lost or puzzled
contentment: calm satisfaction, peaceful acceptance
disappointment: sadness caused by unmet expectations
disgust: strong aversion or revulsion
enthrallment: deep fascination or captivation
envy: pain from wanting what others have
exasperation: emotional exhaustion or fed-up frustration
gratitude: thankful appreciation
horror: fear mixed with shock or dread
irritabilty: persistent annoyance or impatience
lust: strong sexual desire
neglect: feeling ignored, abandoned, or unimportant
nervousness: anxiety, tension, anticipatory fear
optimism: hopefulness about the future
pride: positive self-regard or dignity
rage: intense, explosive anger
relief: emotional release after tension
sadness: emotional pain, sorrow, loss
shame: painful self-consciousness, moral embarrassment
suffering: prolonged emotional pain or anguish
surprise: sudden emotional reaction to the unexpected
sympathy: compassion for others’ pain
zest: energetic enthusiasm for life
"""

DEFAULT_HF_LLM_MODEL = os.environ.get(HUGGINGFACE_LLM_MODEL_ENV)


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("LLM returned empty text response.")

    # Attempt to parse the full text first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON object from the response.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Unable to extract JSON from LLM response.")

    candidate = text[start : end + 1]
    return json.loads(candidate)


def _resolve_hf_token(token: Optional[str]) -> Optional[str]:
    if token:
        return token
    return os.environ.get(HUGGINGFACE_API_TOKEN_ENV) or os.environ.get(HUGGINGFACE_API_TOKEN_ALT_ENV)


def call_huggingface_llm(
    messages: List[Dict[str, str]],
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 60,
) -> str:
    model_id = model_id or DEFAULT_HF_LLM_MODEL
    if not model_id:
        raise EnvironmentError(
            f"Hugging Face model ID must be set via {HUGGINGFACE_LLM_MODEL_ENV}."
        )
    token = _resolve_hf_token(token)
    if not token:
        raise EnvironmentError(
            f"Hugging Face API token must be set via {HUGGINGFACE_API_TOKEN_ENV} or {HUGGINGFACE_API_TOKEN_ALT_ENV}."
        )

    client = InferenceClient(api_key=token)
    completion = client.chat.completions.create(model=model_id, messages=messages)

    if hasattr(completion, "choices") and completion.choices:
        choice = completion.choices[0]
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        if message is not None:
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):
                content = message.get("content")
            if isinstance(content, str):
                return content.strip()

    if isinstance(completion, dict):
        choices = completion.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message") if isinstance(first, dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()

    raise RuntimeError(
        f"Unexpected Hugging Face chat completion format: {type(completion)}"
    )


def build_emotional_reading_prompt(lyrics: str) -> str:
    return f"""
You are analyzing Chinese song lyrics.

First, read the lyrics and describe:
- the overall emotional atmosphere
- emotional progression or contrasts
- emotional tone (gentle, intense, suppressed, explosive, etc.)

Do NOT use emotion labels.
Only use natural language description.

Lyrics:
{lyrics}
""".strip()


def build_emotion_mapping_prompt(emotional_analysis: str) -> str:
    return f"""
Based on the emotional analysis below, identify all applicable emotions
using ONLY the following labels and definitions.

Emotion definitions:
{EMOTION_DEFINITIONS}

Rules:
- Multiple emotions may coexist
- Assign an intensity score from 0–100 for each applicable emotion
- Do NOT include emotions without clear textual support

Emotional analysis:
{emotional_analysis}
""".strip()


def build_json_output_prompt(emotion_mapping_text: str) -> str:
    return f"""
Convert the emotion analysis into the following JSON format.

Output ONLY valid JSON. Do not explain.

Format:
{{
  "dominant_emotions": [
    {{"label": "<emotion>", "intensity": <number>}}
  ],
  "secondary_emotions": [
    {{"label": "<emotion>", "intensity": <number>}}
  ],
  "notes": "<short explanation of emotional structure>"
}}

Emotion analysis:
{emotion_mapping_text}
""".strip()


def analyze_lyric_emotion(lyrics: str, model_id: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
    prompt1 = build_emotional_reading_prompt(lyrics)
    analysis_text = call_huggingface_llm(
        [
            {"role": "system", "content": "You are a careful literary emotion analyst."},
            {"role": "user", "content": prompt1},
        ],
        model_id=model_id,
        token=token,
    )

    prompt2 = build_emotion_mapping_prompt(analysis_text)
    emotion_mapping = call_huggingface_llm(
        [
            {"role": "system", "content": "You are an expert in emotion categorization."},
            {"role": "user", "content": prompt2},
        ],
        model_id=model_id,
        token=token,
    )

    prompt3 = build_json_output_prompt(emotion_mapping)
    json_output = call_huggingface_llm(
        [
            {"role": "system", "content": "You only output valid JSON."},
            {"role": "user", "content": prompt3},
        ],
        model_id=model_id,
        token=token,
    )

    return _extract_json_from_text(json_output)


def _convert_llm_emotion_predictions(llm_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for section in ("dominant_emotions", "secondary_emotions"):
        for item in llm_output.get(section, []):
            label = item.get("label")
            intensity = item.get("intensity")
            if label is None or intensity is None:
                continue
            try:
                intensity = float(intensity)
            except (TypeError, ValueError):
                continue
            score = max(0.0, min(1.0, intensity / 100.0))
            predictions.append(
                {
                    "label": str(label).strip(),
                    "english_label": str(label).strip(),
                    "score": score,
                }
            )
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions


class HuggingFaceLLMTextEmotion:
    """Online text emotion predictor using Hugging Face Inference API."""

    MAX_EMOTION_CLASSES = len([
        line for line in EMOTION_DEFINITIONS.strip().splitlines() if line.strip()
    ])

    def __init__(
        self,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model_id = model_id or DEFAULT_HF_LLM_MODEL
        self.token = token or os.environ.get(HUGGINGFACE_API_TOKEN_ENV)
        if verbose:
            print(
                f"Initializing Hugging Face LLM text emotion predictor with model: {self.model_id}"
            )
        if not self.model_id:
            raise EnvironmentError(
                f"Set {HUGGINGFACE_LLM_MODEL_ENV} to a Hugging Face chat-capable model ID."
            )
        if not self.token:
            raise EnvironmentError(
                f"Set {HUGGINGFACE_API_TOKEN_ENV} to your Hugging Face API token."
            )

    def predict(self, text: str) -> List[Dict[str, Any]]:
        output = analyze_lyric_emotion(text, model_id=self.model_id, token=self.token)
        return _convert_llm_emotion_predictions(output)

    def predict_top_n(self, text: str, n: Optional[int] = 3) -> List[Dict[str, Any]]:
        preds = self.predict(text)
        if n is None:
            return preds
        if n <= 0:
            return []
        return preds[:n]
