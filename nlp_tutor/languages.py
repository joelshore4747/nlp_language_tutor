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

    @staticmethod
    def parse(value: str) -> "Lang":

        if value is None:
            raise ValueError("Language is required")

        s = value.strip()
        if not s:
            raise ValueError("Language is required")


        u = s.upper()
        if u in Lang.__members__:
            return Lang[u]

        l = s.lower()
        for lang in Lang:
            if l == lang.value:
                return lang

        l2 = s.lower()
        for lang in Lang:
            if l2 == lang.display_name.lower():
                return lang

        raise ValueError(f"Unsupported language: {value!r}")
