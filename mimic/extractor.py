from __future__ import annotations

import re
from typing import Any

from mimic.types import Feature

_NEGATIONS = {"no", "not", "never", "none", "cannot", "n't"}
_UNCERTAIN = ["i don't know", "i do not know", "not sure", "unsure", "maybe", "i'm not sure"]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


class Extractor:
    _NAMES = ["word_count", "negation_count", "has_uncertainty",
              "novel_word_ratio", "entity_overlap_ratio"]

    def feature_names(self) -> list[str]:
        return list(self._NAMES)

    def extract(self, inputs: dict[str, Any], only: list[str] | None = None) -> list[Feature]:
        response = str(inputs.get("response", ""))
        context = str(inputs.get("context", ""))
        rwords = _words(response)
        cwords = set(_words(context))

        values: dict[str, float | bool | int] = {
            "word_count": len(rwords),
            "negation_count": sum(1 for w in rwords if w in _NEGATIONS),
            "has_uncertainty": any(p in response.lower() for p in _UNCERTAIN),
            "novel_word_ratio": (
                sum(1 for w in rwords if w not in cwords) / len(rwords) if rwords else 0.0
            ),
            "entity_overlap_ratio": (
                sum(1 for w in cwords if w in set(rwords)) / len(cwords) if cwords else 0.0
            ),
        }

        wanted = only if only is not None else self._NAMES
        cat = {"word_count": "structural", "negation_count": "linguistic",
               "has_uncertainty": "linguistic", "novel_word_ratio": "semantic",
               "entity_overlap_ratio": "semantic"}
        return [Feature(name, values[name], cat[name], "cheap")
                for name in wanted if name in values]
