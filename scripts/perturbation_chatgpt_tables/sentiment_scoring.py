# Copyright : Somantra Pty Ltd.
# Licensed under the GNU General Public License v3.0 (see LICENSE / COPYRIGHT).
"""
Local sentiment scoring for --score-sentiment.

Runs a local HuggingFace transformer model (no API key, no network calls beyond the one-time
model download from the HuggingFace Hub) to score positive/negative/neutral sentiment for a
batch of phrases. Used by perturbation_chatgpt_brand_position.py to measure sentiment shift
between a base and perturbed brand-positioning phrase.

Requires the optional dependencies in requirements-optional.txt (torch, transformers).
"""
import torch
from scipy.special import softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _preprocess(text: str) -> str:
    """Normalize the way the model's training data (Twitter) was normalized: usernames and
    URLs carry no sentiment signal of their own, so they're replaced with generic placeholders
    rather than left to confuse the model."""
    words = []
    for word in text.split(" "):
        if word.startswith("@") and len(word) > 1:
            word = "@user"
        elif word.startswith("http"):
            word = "http"
        words.append(word)
    return " ".join(words)


def analyze_sentiment_batch(texts: list[str]) -> list[dict]:
    """Return, for each input text, {"text": ..., "sentiments": [{"label": ..., "score": ...}, ...]}
    with labels ranked highest score first."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    processed = [_preprocess(t) for t in texts]
    encoded = tokenizer(processed, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output = model(**encoded)

    import numpy as np
    scores = softmax(output.logits.detach().cpu().numpy(), axis=1)

    results = []
    for i, text in enumerate(texts):
        ranking = np.argsort(scores[i])[::-1]
        results.append({
            "text": text,
            "sentiments": [
                {"label": config.id2label[rank], "score": float(scores[i][rank])}
                for rank in ranking
            ],
        })
    return results
