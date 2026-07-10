from __future__ import annotations

from typing import Protocol

import tiktoken


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class TiktokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))
