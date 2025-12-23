from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import List, Tuple, Dict

import re
import numpy as np
import torch
from torch.utils.data import Dataset

from .. import preprocessing as prep


_word_re = re.compile(r"[A-Za-z']+")


def basic_tokenize(text: str) -> List[str]:
    # simple and stable for IMDB
    text = prep.normalise(text)
    return _word_re.findall(text)


@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_idx: int
    unk_idx: int

    @classmethod
    def build(cls, token_lists: List[List[str]], max_size: int = 30000, min_freq: int = 2) -> "Vocab":
        counter = Counter(t for toks in token_lists for t in toks)
        itos = ["<PAD>", "<UNK>"]
        for tok, freq in counter.most_common():
            if freq < min_freq:
                continue
            if tok in itos:
                continue
            itos.append(tok)
            if len(itos) >= max_size:
                break
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, itos=itos, pad_idx=0, unk_idx=1)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]


def pad_batch(seqs: List[List[int]], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      x: [B, T] padded
      lengths: [B] original lengths
    """
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) else 0
    x = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x, lengths


class IMDBSequenceDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], vocab: Vocab, max_len: int = 400) -> None:
        self.vocab = vocab
        self.max_len = max_len

        self.x: List[List[int]] = []
        self.y: List[int] = []

        for t, lbl in zip(texts, labels):
            toks = basic_tokenize(t)[:max_len]
            self.x.append(vocab.encode(toks))
            self.y.append(1 if lbl in ("positive", "pos", "1") else 0)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.x[idx], self.y[idx]
