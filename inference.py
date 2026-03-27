from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="my_imdb_sentiment_model",
    tokenizer="my_imdb_sentiment_model"
)

reviews = [
    "This movie was absolutely fantastic! I loved it.",
    "Worst film ever. Complete waste of time."
]

for review in reviews:
    result = classifier(review)[0]
    label = "POSITIVE" if result["label"] == "LABEL_1" else "NEGATIVE"
    print(f"Review: {review}")
    print(f"Sentiment: {label} ({result['score']:.4f})\n")