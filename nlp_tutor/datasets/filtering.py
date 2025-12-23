from __future__ import annotations
from typing import Set
from nlp_tutor.corpora import TextLabelDataset

def filter_dataset(ds: TextLabelDataset, keep: Set[str]) -> TextLabelDataset:
    texts = []
    labels = []
    for t, y in zip(ds.texts, ds.labels):
        if y in keep:
            texts.append(t)
            labels.append(y)
    return TextLabelDataset(texts=texts, labels=labels)
