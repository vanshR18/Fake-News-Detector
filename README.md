# 📰 Fake News Detector

A production-ready fake news detection system powered by **DistilBERT** fine-tuned on the **WELFake** dataset (~72K articles). The system exposes a REST API via FastAPI and a live demo via Gradio on HuggingFace Spaces.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Demo](#demo)
3. [Results](#results)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Project Structure](#project-structure)
7. [Installation & Quickstart](#installation--quickstart)
8. [API Reference](#api-reference)
9. [Training](#training)
10. [Deployment](#deployment)
11. [Tech Stack](#tech-stack)
12. [Author](#author)

---

## Project Overview

Fake news detection is a critical NLP problem. This project builds an end-to-end pipeline:

- Fine-tunes **DistilBERT** (distilbert-base-uncased) on WELFake, a merged dataset of ~72K real and fake news articles
- Combines both **title** and **body text** as a single input (`title [SEP] text`)
- Exposes predictions via a **FastAPI** REST endpoint (`/api/v1/predict`)
- Provides a user-friendly **Gradio** frontend deployed on HuggingFace Spaces
- Establishes a **TF-IDF + Logistic Regression** baseline for comparison

---

## Demo

Live demo: [HuggingFace Spaces](https://huggingface.co/spaces/vanshR18/fake-news-detector)

Training notebook: [Kaggle](https://kaggle.com/vanshR18)

---

## Results

| Model | Accuracy | F1 Score (Weighted) | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression | ~92% | ~0.92 | Baseline |
| DistilBERT (fine-tuned, 3 epochs) | ~96% | ~0.96 | Final model |

Training was done on Kaggle T4 GPU. ~40 minutes for 3 epochs on 64K training samples.

---

## Dataset

**WELFake** — a merged dataset combining four popular news datasets (Kaggle, McIntire, Reuters, BuzzFeed Political), cleaned and deduplicated.

| Split | Samples |
|---|---|
| Total | 72,134 |
| Real News | 35,028 |
| Fake News | 37,106 |
| Train (90%) | ~64,920 |
| Validation (10%) | ~7,214 |

Download: [Kaggle – WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

---

## Model Architecture

```
Input: "title [SEP] article text" (truncated to 512 tokens)
          |
   DistilBERT Encoder
   (6 transformer layers, 66M parameters)
          |
   CLS token representation
          |
   Linear classifier (768 → 2)
          |
   Softmax → [P(FAKE), P(REAL)]
```

**Why DistilBERT?**
- 40% smaller than BERT, 60% faster, retains 97% of performance
- Ideal for deployment on CPU/free-tier GPU
- Strong contextual understanding of news language

**Input Strategy:** Concatenating title and body with `[SEP]` gives the model both the headline signal (often highly indicative) and article-level context.

---

## Project Structure

```
fake-news-detector/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # /predict and /model-info endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   └── model.py             # Model loading & inference logic
│   └── schemas/
│       ├── __init__.py
│       └── prediction.py        # Pydantic I/O schemas
├── model/
│   ├── __init__.py
│   └── train.py                 # Training script (run on Kaggle T4 GPU)
├── notebooks/                   # EDA + training Jupyter notebooks
├── tests/
│   ├── __init__.py
│   └── test_api.py              # API unit tests
├── app_gradio.py                # HuggingFace Spaces frontend
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation & Quickstart

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/vanshR18/fake-news-detector
cd fake-news-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download and place the trained model

After training (see Training section), place the saved model at:

```
model/saved_model/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
└── vocab.txt
```

Or pull directly from HuggingFace Hub:

```python
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model = DistilBertForSequenceClassification.from_pretrained("vanshR18/fake-news-detector")
tokenizer = DistilBertTokenizerFast.from_pretrained("vanshR18/fake-news-detector")
```

### 5. Run the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 6. Run the Gradio demo locally

```bash
python app_gradio.py
```

---

## API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### `POST /predict`

Classify a news article as REAL or FAKE.

**Request body:**
```json
{
  "title": "Scientists develop mRNA vaccine for cancer",
  "text": "Researchers at MIT have published early-stage results showing..."
}
```

**Response:**
```json
{
  "label": "REAL",
  "confidence": 0.9741,
  "fake_probability": 0.0259,
  "real_probability": 0.9741
}
```

| Field | Type | Description |
|---|---|---|
| `label` | string | `"REAL"` or `"FAKE"` |
| `confidence` | float | Probability of the predicted class (0–1) |
| `fake_probability` | float | Model's probability score for FAKE |
| `real_probability` | float | Model's probability score for REAL |

**cURL example:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"title": "5G towers cause mind control", "text": "Anonymous sources claim..."}'
```

#### `GET /model-info`

Returns metadata about the loaded model.

**Response:**
```json
{
  "model": "DistilBERT",
  "dataset": "WELFake (72K samples)",
  "labels": ["FAKE", "REAL"],
  "max_input_length": 512
}
```

#### `GET /health`

Health check endpoint.

```json
{ "status": "healthy" }
```

---

## Training

The training script is `model/train.py`. It is designed to run on **Kaggle with a free T4 GPU**.

### Steps

1. Go to [Kaggle](https://kaggle.com) → Create New Notebook
2. Add the WELFake dataset to the notebook
3. Enable GPU (T4 x2) in Notebook Settings
4. Copy `model/train.py` into a notebook cell and run

### Key hyperparameters

| Parameter | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max sequence length | 512 |
| Train/val split | 90/10 |
| Mixed precision (fp16) | Yes (on GPU) |

### Expected training time

| Hardware | Time |
|---|---|
| Kaggle T4 GPU | ~40 minutes |
| CPU only | ~6–8 hours |

After training, save the model to HuggingFace Hub:

```python
model.push_to_hub("vanshR18/fake-news-detector")
tokenizer.push_to_hub("vanshR18/fake-news-detector")
```

---

## Deployment

### HuggingFace Spaces (Gradio)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set SDK to **Gradio**
3. Upload `app_gradio.py` and `requirements.txt`
4. The Space will auto-build and deploy

### Local Docker (optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | DistilBERT (HuggingFace Transformers) |
| Training | PyTorch + HuggingFace Trainer API |
| Dataset | WELFake via Kaggle |
| API Backend | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Frontend | Gradio |
| Deployment | HuggingFace Spaces |
| Testing | Pytest + FastAPI TestClient |
| GPU Training | Kaggle T4 (free tier) |

---

## Author

**Rohit Pal**
B.Tech CSE (2027) — IET Lucknow

GitHub: [github.com/vanshR18](https://github.com/vanshR18)
Kaggle: [kaggle.com/vanshR18](https://kaggle.com/vanshR18)