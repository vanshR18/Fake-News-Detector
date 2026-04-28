from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from pathlib import Path

MODEL_PATH = Path("model/saved_model")

class FakeNewsModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_PATH))
        self.model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, title: str, text: str) -> dict:
        input_text = f"{title} [SEP] {text}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits, dim=1).squeeze().tolist()
        label_id = int(probs[1] < probs[0])
        return {
            "label": ["FAKE", "REAL"][label_id],
            "confidence": round(max(probs), 4),
            "fake_probability": round(probs[0], 4),
            "real_probability": round(probs[1], 4),
        }

news_model = FakeNewsModel()
news_model.load()