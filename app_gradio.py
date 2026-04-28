"""
Gradio frontend for HuggingFace Spaces deployment.
"""
import gradio as gr
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

MODEL_PATH = "model/saved_model" # your HF repo
LABELS = {0: "🔴 FAKE", 1: "🟢 REAL"}

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(title, text):
    if not title.strip() or not text.strip():
        return "Please enter both title and text.", None
    
    input_text = f"{title} [SEP] {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1).squeeze().tolist()
    
    label_id = int(probs[1] < probs[0])
    label = LABELS[label_id]
    confidence = max(probs)
    
    return label, {
        "FAKE": round(probs[0], 3),
        "REAL": round(probs[1], 3)
    }

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="News Title", placeholder="Enter the headline..."),
        gr.Textbox(label="News Text", placeholder="Enter the article body...", lines=6),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence Scores"),
    ],
    title="📰 Fake News Detector",
    description="DistilBERT fine-tuned on WELFake dataset (72K articles). Enter a news title and body to classify.",
    examples=[
        ["Scientists develop mRNA vaccine for cancer", "Researchers at MIT have published promising early results..."],
        ["BREAKING: 5G towers cause mind control", "Anonymous sources claim that the government has secretly..."],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()