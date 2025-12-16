# nlp_tutor/ngram_model.py
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List
import math


@dataclass
class NgramLMConfig:
    order: int = 2          # 2 = bigram, 3 = trigram
    add_k: float = 1.0      # add-k smoothing (1.0 = add-one)


class NgramLanguageModel:
    def __init__(self, config: NgramLMConfig | None = None) -> None:
        self.config = config or NgramLMConfig()
        self.order = self.config.order
        self.add_k = self.config.add_k

        self.ngram_counts: Dict[tuple, Counter] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.vocab: set[str] = set()
        self.trained: bool = False

    def fit(self, sentences: Iterable[List[str]]) -> None:
        order = self.order

        for sent in sentences:
            # add boundary tokens
            tokens = ["<s>"] * (order - 1) + sent + ["</s>"]

            # update counts
            for i in range(order - 1, len(tokens)):
                context = tuple(tokens[i - order + 1 : i])
                token = tokens[i]
                self.ngram_counts[context][token] += 1
                self.context_counts[context] += 1
                self.vocab.add(token)

        self.trained = True


    def _cond_prob(self, context: tuple, token: str) -> float:
        """
        P(token | context) with add-k smoothing.
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        k = self.add_k
        vocab_size = len(self.vocab) or 1

        num = self.ngram_counts[context][token] + k
        den = self.context_counts[context] + k * vocab_size
        return num / den

    def sentence_log_prob(self, tokens: List[str]) -> float:
        """
        Log-probability (natural log) of a sentence, including end token.
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call fit() first.")

        order = self.order
        # apply same boundary handling as in training
        seq = ["<s>"] * (order - 1) + tokens + ["</s>"]

        log_p = 0.0
        for i in range(order - 1, len(seq)):
            context = tuple(seq[i - order + 1 : i])
            tok = seq[i]
            p = self._cond_prob(context, tok)
            log_p += math.log(p)
        return log_p

    def perplexity(self, tokens: List[str]) -> float:
        """
        Perplexity for a single sentence (lower = more fluent under this LM).
        """
        if not tokens:
            return float("inf")
        log_p = self.sentence_log_prob(tokens)
        N = len(tokens) + 1  # +1 for </s>
        return math.exp(-log_p / N)
