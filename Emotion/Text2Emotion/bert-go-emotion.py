from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Any, Optional

EMOTION_MODEL_ID = "SchuylerH/bert-multilingual-go-emtions"


class BertGoEmotion:
    """BERT Go-Emotion predictor wrapper."""

    MAX_EMOTION_CLASSES = 28

    def __init__(self, model_id: str = EMOTION_MODEL_ID, device: Optional[int] = None, verbose: bool = True):
        if verbose:
            print(f"Loading emotion model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        pipeline_kwargs = {"model": self.model, "tokenizer": self.tokenizer}
        if device is not None:
            pipeline_kwargs["device"] = device
        self.nlp = pipeline("sentiment-analysis", **pipeline_kwargs)

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Return the full emotion scores for the provided text."""
        all_scores = self.nlp(text, top_k=None)
        results = [{"label": item["label"], "score": float(item["score"])} for item in all_scores]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def predict_top_n(self, text: str, n: Optional[int] = 3) -> List[Dict[str, Any]]:
        """Return the top n emotion scores for the provided text."""
        if n is None:
            return self.predict(text)
        if n <= 0:
            return []
        return self.predict(text)[:n]
