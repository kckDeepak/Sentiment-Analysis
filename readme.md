
# IMDB Sentiment Analysis with DistilBERT

A fine-tuned DistilBERT model for binary sentiment classification (Positive/Negative) on the IMDB movie reviews dataset. This project demonstrates the classic "Hello World" of NLP fine-tuning using Hugging Face Transformers.

## Overview

This project fine-tunes **DistilBERT-base-uncased** on the IMDB dataset to classify movie reviews as **positive** or **negative**.

- **Model**: DistilBERT (faster and lighter version of BERT)
- **Dataset**: Stanford IMDB Movie Reviews
- **Task**: Binary Sentiment Classification
- **Framework**: Hugging Face Transformers + Trainer API

## Final Results (on small subset)

- **Accuracy**: 89.0%
- **Training Time**: ~6 minutes on T4 GPU
- **Epochs**: 3

*(Results will be significantly better when trained on the full 25k dataset)*

## How to Use

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Inference

You can test the model with new reviews using the pipeline:

```python
from transformers import pipeline

# Load the fine-tuned model
classifier = pipeline(
    "sentiment-analysis",
    model="my_imdb_sentiment_model",
    tokenizer="my_imdb_sentiment_model"
)

# Example usage
reviews = [
    "This movie was absolutely fantastic!",
    "Worst film I have ever seen. Total waste of time."
]

for review in reviews:
    result = classifier(review)[0]
    label = "POSITIVE" if result["label"] == "LABEL_1" else "NEGATIVE"
    print(f"Review: {review}")
    print(f"Sentiment: {label} ({result['score']:.4f})\n")
```

## Training Details

- **Pre-trained Model**: `distilbert-base-uncased`
- **Dataset**: `imdb` (Hugging Face)
- **Training Arguments**:
  - Learning rate: `2e-5`
  - Batch size: 16
  - Epochs: 3
  - Weight decay: 0.01
- **Hardware**: Google Colab (T4 GPU)

## How to Reproduce

1. Open the notebook in Google Colab
2. Run all cells in order
3. The model will be saved in the `my_imdb_sentiment_model/` folder

## Requirements

```txt
transformers
datasets
evaluate
accelerate
torch
```

*(You can generate `requirements.txt` using `pip freeze > requirements.txt`)*

## Next Steps / Improvements

- Train on the full 25,000 examples for higher accuracy (~92-94%)
- Add precision, recall, and F1-score metrics
- Deploy as a Gradio / Streamlit web app
- Push model to Hugging Face Hub
- Experiment with BERT-base or RoBERTa

## Acknowledgments

- Hugging Face Transformers & Datasets libraries
- Stanford IMDB Dataset

---

**Made with ❤️ for learning NLP fine-tuning**

