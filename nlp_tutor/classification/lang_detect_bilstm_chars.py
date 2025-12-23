from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import json
import random
import unicodedata

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from nlp_tutor import preprocessing as prep
from nlp_tutor.config import PATHS
from nlp_tutor.corpora import TextLabelDataset


MODEL_NAME = "lang_detect_char_bilstm.pt"
META_NAME = "lang_detect_char_bilstm_meta.json"


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_diacritics(s: str) -> str:
    # Useful to test robustness when users omit accents (e.g., "dzien dobry")
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def add_typo_noise(s: str, p: float = 0.08) -> str:
    # Very light noise: randomly drop a char with small probability
    if len(s) < 5:
        return s
    out = []
    for ch in s:
        if random.random() < p:
            continue
        out.append(ch)
    return "".join(out) if out else s


# ----------------------------
# Vocab + Encoding
# ----------------------------

def build_char_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for t in texts:
        for ch in t:
            freq[ch] = freq.get(ch, 0) + 1

    # reserve:
    # 0 = PAD, 1 = UNK
    char2idx = {"<PAD>": 0, "<UNK>": 1}

    for ch, f in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if f >= min_freq and ch not in char2idx:
            char2idx[ch] = len(char2idx)

    return char2idx


def encode_text(text: str, char2idx: Dict[str, int], max_chars: int) -> List[int]:
    ids = []
    for ch in text[:max_chars]:
        ids.append(char2idx.get(ch, 1))  # UNK
    if not ids:
        ids = [1]
    return ids


def pad_batch(seqs: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    x = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x, lengths


# ----------------------------
# Dataset
# ----------------------------

class CharLangDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        char2idx: Dict[str, int],
        max_chars: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.char2idx = char2idx
        self.max_chars = max_chars

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> Tuple[List[int], int]:
        t = self.texts[i]
        x = encode_text(t, self.char2idx, self.max_chars)
        y = self.labels[i]
        return x, y


def collate_fn(batch: List[Tuple[List[int], int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, ys = zip(*batch)
    x, lengths = pad_batch(list(seqs), pad_id=0)
    y = torch.tensor(list(ys), dtype=torch.long)
    return x, lengths, y


# ----------------------------
# Model
# ----------------------------

class CharBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        e = self.emb(x)  # (B, T, E)

        # pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)

        # h_n: (num_layers*2, B, H)
        # take last layer forward + backward
        h_fwd = h_n[-2]  # (B, H)
        h_bwd = h_n[-1]  # (B, H)
        h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2H)

        h = self.dropout(h)
        logits = self.fc(h)  # (B, C)
        return logits


# ----------------------------
# Train/Eval
# ----------------------------

@dataclass(frozen=True)
class TrainResult:
    accuracy: float
    f1_macro: float
    confusion: List[List[int]]
    labels: List[str]
    report: str


def _filter_ds(ds: TextLabelDataset, target_langs: Optional[Set[str]]) -> TextLabelDataset:
    if not target_langs:
        return ds
    texts: List[str] = []
    labels: List[str] = []
    for t, y in zip(ds.texts, ds.labels):
        y = str(y)
        if y in target_langs:
            texts.append(t)
            labels.append(y)
    if len(set(labels)) < 2:
        raise ValueError(f"Not enough classes after filtering: {sorted(set(labels))}")
    return TextLabelDataset(texts=texts, labels=labels)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx2label: List[str],
) -> Tuple[float, float, List[List[int]], str]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for x, lengths, y in loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            logits = model(x, lengths)
            pred = torch.argmax(logits, dim=1)

            all_true.extend(y.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())

    acc = float(accuracy_score(all_true, all_pred))
    f1m = float(f1_score(all_true, all_pred, average="macro"))

    labels_sorted = idx2label
    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(labels_sorted)))).tolist()

    rep = classification_report(
        all_true,
        all_pred,
        target_names=labels_sorted,
        digits=4,
    )

    return acc, f1m, cm, rep


def train_eval_save(
    ds: TextLabelDataset,
    target_langs: Optional[Set[str]] = None,
    max_chars: int = 120,
    test_size: float = 0.2,
    seed: int = 42,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 2e-3,
    emb_dim: int = 64,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    save_dir: Path | None = None,
) -> TrainResult:
    set_seed(seed)

    ds = _filter_ds(ds, target_langs)

    # Preprocess (lowercase/space normalisation)
    texts = [prep.normalise(t) for t in ds.texts]
    labels_str = [str(y) for y in ds.labels]

    # label mapping
    label_set = sorted(set(labels_str))
    label2idx = {l: i for i, l in enumerate(label_set)}
    idx2label = label_set

    y_all = [label2idx[y] for y in labels_str]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y_all,
        test_size=test_size,
        random_state=seed,
        stratify=y_all,
    )

    # char vocab from training set only (good practice)
    char2idx = build_char_vocab(X_train, min_freq=1)

    train_ds = CharLangDataset(X_train, y_train, char2idx, max_chars=max_chars)
    test_ds = CharLangDataset(X_test, y_test, char2idx, max_chars=max_chars)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CharBiLSTM(
        vocab_size=len(char2idx),
        n_classes=len(idx2label),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x, lengths)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        acc, f1m, cm, rep = evaluate_model(model, test_loader, device, idx2label)
        avg_loss = total_loss / max(1, n_batches)

        print(f"Epoch {ep}/{epochs} | loss={avg_loss:.4f} | acc={acc:.4f} | f1_macro={f1m:.4f}")

        if f1m > best_f1:
            best_f1 = f1m
            best_state = {
                "state_dict": model.state_dict(),
                "char2idx": char2idx,
                "idx2label": idx2label,
                "config": {
                    "max_chars": max_chars,
                    "emb_dim": emb_dim,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                }
            }

    if best_state is None:
        raise RuntimeError("Training failed to produce a best state.")

    # Save best
    if save_dir is None:
        save_dir = PATHS.models_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, save_dir / MODEL_NAME)
    (save_dir / META_NAME).write_text(json.dumps({
        "model": MODEL_NAME,
        "labels": best_state["idx2label"],
        "vocab_size": len(best_state["char2idx"]),
        "config": best_state["config"],
    }, indent=2), encoding="utf-8")

    # Reload best state for final report metrics
    model.load_state_dict(best_state["state_dict"])
    acc, f1m, cm, rep = evaluate_model(model, test_loader, device, idx2label)

    return TrainResult(
        accuracy=acc,
        f1_macro=f1m,
        confusion=cm,
        labels=idx2label,
        report=rep,
    )


def load_model(save_dir: Path | None = None) -> Dict:
    if save_dir is None:
        save_dir = PATHS.models_dir
    path = save_dir / MODEL_NAME
    if not path.exists():
        raise FileNotFoundError(f"BiLSTM model not found at {path}. Train it first.")
    return torch.load(path, map_location="cpu")


def predict_language(
    text: str,
    top_k: int = 3,
    save_dir: Path | None = None,
) -> Dict:
    pack = load_model(save_dir=save_dir)

    char2idx = pack["char2idx"]
    idx2label = pack["idx2label"]
    cfg = pack["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CharBiLSTM(
        vocab_size=len(char2idx),
        n_classes=len(idx2label),
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()

    x_ids = encode_text(prep.normalise(text), char2idx, max_chars=cfg["max_chars"])
    x, lengths = pad_batch([x_ids], pad_id=0)
    x = x.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        logits = model(x, lengths)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()

    scored = sorted(
        [(idx2label[i], float(probs[i])) for i in range(len(idx2label))],
        key=lambda t: t[1],
        reverse=True
    )[:top_k]

    return {"label": scored[0][0], "top_k": scored}
