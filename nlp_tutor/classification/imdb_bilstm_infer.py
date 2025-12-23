from __future__ import annotations

from typing import Dict
from pathlib import Path

import joblib
import torch

from ..config import PATHS
from .sequence_data import basic_tokenize, Vocab, pad_batch
from .bilstm_model import BiLSTMSentiment


MODEL_PATH = PATHS.models_dir / "sentiment_bilstm.pt"
VOCAB_PATH = PATHS.models_dir / "sentiment_bilstm_vocab.joblib"


def load_bilstm(device: str = "cpu"):
    vocab: Vocab = joblib.load(VOCAB_PATH)
    model = BiLSTMSentiment(vocab_size=len(vocab.itos), pad_idx=vocab.pad_idx)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, vocab


def predict_bilstm(text: str, model, vocab: Vocab, device: str = "cpu") -> Dict[str, float | str]:
    toks = basic_tokenize(text)
    ids = vocab.encode(toks[:400])
    x, lengths = pad_batch([ids], pad_idx=vocab.pad_idx)
    x, lengths = x.to(device), lengths.to(device)

    with torch.no_grad():
        logits = model(x, lengths)
        prob = float(torch.sigmoid(logits)[0].cpu().item())

    label = "positive" if prob >= 0.5 else "negative"
    return {"label": label, "prob_positive": prob}
