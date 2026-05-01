import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional

EMOTION_MODEL_ID = "Johnson8187/Chinese-Emotion"
LABEL_MAPPING = {
    0: "平淡",
    1: "關切",
    2: "開心",
    3: "憤怒",
    4: "悲傷",
    5: "疑問",
    6: "驚奇",
    7: "厭惡",
}

LABEL_MAPPING_EN = {
    0: "Neutral tone",
    1: "Concerned tone",
    2: "Happy tone",
    3: "Angry tone",
    4: "Sad tone",
    5: "Questioning tone",
    6: "Surprised tone",
    7: "Disgusted tone",
}


class JohnsonChineseEmotion:
    """Chinese emotion predictor wrapper using Johnson8187/Chinese-Emotion."""

    MAX_EMOTION_CLASSES = len(LABEL_MAPPING)

    def __init__(self, model_id: str = EMOTION_MODEL_ID, device: Optional[torch.device] = None, verbose: bool = True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f"Loading emotion model: {model_id} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)

    def _predict_scores(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        return probs.cpu().tolist()

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Return the full emotion scores for the provided text."""
        scores = self._predict_scores(text)
        results = [
            {
                "label": LABEL_MAPPING.get(idx, str(idx)),
                "english_label": LABEL_MAPPING_EN.get(idx, str(idx)),
                "score": float(score),
            }
            for idx, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def predict_top_n(self, text: str, n: Optional[int] = 3) -> List[Dict[str, Any]]:
        """Return the top n emotion scores for the provided text."""
        if n is None:
            return self.predict(text)
        if n <= 0:
            return []
        return self.predict(text)[:n]
