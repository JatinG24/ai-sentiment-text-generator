import warnings
warnings.filterwarnings("ignore")
from transformers import pipeline
import torch

class SentimentClassifier:
    def __init__(self, model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"):
        # Detect device: 0 = GPU, -1 = CPU
        self.device = 0 if torch.cuda.is_available() else -1

        # Load pipeline directly on the device; no manual .to()
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name_or_path,
            device=self.device
        )

    def predict(self, text: str) -> str:
        result = self.pipe(text)
        label = result[0]["label"].lower()
        if "pos" in label:
            return "positive"
        if "neg" in label:
            return "negative"
        if "neutral" in label or "mixed" in label:
            return "neutral"
        return "neutral"
