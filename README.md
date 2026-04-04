# 📰 Fake News Detector

> DistilBERT fine-tuned on WELFake dataset (~72K articles) for binary fake/real news classification.

[![HuggingFace Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow)]()  
[![Kaggle](https://img.shields.io/badge/Kaggle-Training%20Notebook-blue)](https://www.kaggle.com/code/rohitv18/fake-news-with-bert-acc-99-4)

---

## 🚀 Current Progress

### ✅ Completed

- 📊 **Data Analysis & Preprocessing**
  - Explored WELFake dataset (72K articles)
  - Cleaned text and handled missing values
  - Performed EDA to understand class distribution

- 🧠 **Model Training**
  
  - Fine-tuned **DistilBERT**
    - Achieved ~96% Accuracy / F1 Score
  - Training done using PyTorch + HuggingFace Transformers

- 📓 **Notebooks**
  - EDA notebook
  - Training notebook (Kaggle)

- ⚙️ **Model Integration**
  - Loading trained model for inference
  - Optimizing prediction pipeline

- 🌐 **Backend (FastAPI)**
  - `/predict` API endpoint for real-time inference
  - Input validation using Pydantic schemas
  - Model serving with optimized loading

---

### 🚧 In Progress
- 🎨 **Frontend (Gradio / HuggingFace Spaces)**
  - Simple UI for users to input news text
  - Display prediction (Fake / Real) with confidence score

 ---
 
### 🛠️ To Be Built

- 🚀 **Deployment**
  - Deploy FastAPI backend
  - Integrate with HuggingFace Spaces frontend

- ⚡ **Improvements (Future Scope)**
  - Add confidence visualization
  - Handle long articles (chunking)
  - Compare with larger models (BERT / RoBERTa)
  - Add explainability (attention visualization / SHAP)

---

## 📊 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| DistilBERT (fine-tuned) | ~96% | ~0.96 |

---
## 📚 Dataset

WELFake — 72,134 news articles  
(35,028 real + 37,106 fake)

---

## 🧰 Tech Stack

- 🤖 DistilBERT  
- 🤗 HuggingFace Transformers  
- 🔥 PyTorch  
- ⚡ FastAPI  
- 🎨 Gradio  
- 🌐 HuggingFace Spaces  

---
