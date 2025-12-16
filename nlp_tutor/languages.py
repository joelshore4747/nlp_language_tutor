# nlp_tutor/languages.py
from __future__ import annotations

from enum import Enum


class Lang(str, Enum):
    EN = "en"
    ES = "es"
    PL = "pl"
    RU = "ru"

    @property
    def display_name(self) -> str:
        return {
            Lang.EN: "English",
            Lang.ES: "Spanish",
            Lang.PL: "Polish",
            Lang.RU: "Russian",
        }[self]
