from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over tokens with attention mask.
    last_hidden_state: (B, T, H)
    attention_mask:    (B, T)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # (B, 1)
    return summed / counts


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # a, b: shape (H,)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_length: int = 128
    batch_size: int = 16


class TransformerEmbedder:
    """
    Lightweight sentence embedding using a multilingual Transformer.

    NOTE:
    - We use mean pooling over token embeddings.
    - We L2-normalize embeddings so cosine similarity behaves well.
    """

    def __init__(self, cfg: EmbedderConfig = EmbedderConfig()) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        embs: List[np.ndarray] = []
        bs = self.cfg.batch_size

        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = _l2_normalize(pooled)

            embs.append(pooled.detach().cpu().numpy())

        return np.vstack(embs)

    def similarity(self, a: str, b: str) -> float:
        vecs = self.encode([a, b])
        return cosine_sim(vecs[0], vecs[1])
