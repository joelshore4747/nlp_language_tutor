from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import joblib

from nlp_tutor.corpora import load_imdb_kaggle
from nlp_tutor.config import PATHS
from nlp_tutor.classification.sequence_data import (
    basic_tokenize, Vocab, IMDBSequenceDataset, pad_batch
)
from nlp_tutor.classification.bilstm_model import BiLSTMSentiment


MODEL_PATH = PATHS.models_dir / "sentiment_bilstm.pt"
VOCAB_PATH = PATHS.models_dir / "sentiment_bilstm_vocab.joblib"


def collate_fn(batch):
    seqs, labels = zip(*batch)
    x, lengths = pad_batch(list(seqs), pad_idx=vocab.pad_idx)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, lengths, y


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_imdb_kaggle()

    X_train, X_test, y_train, y_test = train_test_split(
        ds.texts, ds.labels, test_size=0.2, random_state=42, stratify=ds.labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    token_lists = [basic_tokenize(t) for t in X_train]
    vocab = Vocab.build(token_lists, max_size=30000, min_freq=2)
    joblib.dump(vocab, VOCAB_PATH)

    train_ds = IMDBSequenceDataset(X_train, y_train, vocab=vocab)
    val_ds = IMDBSequenceDataset(X_val, y_val, vocab=vocab)
    test_ds = IMDBSequenceDataset(X_test, y_test, vocab=vocab)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMSentiment(vocab_size=len(vocab.itos), pad_idx=vocab.pad_idx).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = 0.0
    for epoch in range(1, 6):  # 5 epochs to start
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        for x, lengths, y in pbar:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=float(loss.item()))

        # validate
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                logits = model(x, lengths)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int).tolist()
                val_preds.extend(preds)
                val_true.extend(y.numpy().astype(int).tolist())

        val_acc = accuracy_score(val_true, val_preds)
        if val_acc > best_val:
            best_val = val_acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"epoch {epoch} val_acc={val_acc:.4f} best={best_val:.4f}")

    # test
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    test_preds = []
    test_true = []
    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int).tolist()
            test_preds.extend(preds)
            test_true.extend(y.numpy().astype(int).tolist())

    acc = accuracy_score(test_true, test_preds)
    f1 = f1_score(test_true, test_preds, average="macro")
    print("\nTEST RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-macro: {f1:.4f}")
    print("\nReport:\n")
    print(classification_report(test_true, test_preds, target_names=["negative", "positive"]))
