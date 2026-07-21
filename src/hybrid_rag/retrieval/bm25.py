"""Deterministic BM25 lexical scoring for chunk retrieval.

The scorer deliberately has no database, embedding-provider, or service
dependency.  Callers pass the already-loaded chunk :class:`IndexItem` rows so
the same immutable index snapshot can be used for dense and lexical recall.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from hybrid_rag.storage.retrieval_repository import IndexItem

DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.75
BM25_SCORER_VERSION = "bm25-v1"
LEXICAL_TOKENIZER_VERSION = "cjk-word-v1"

_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]+|[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Stable BM25 parameters for one lexical-recall run."""

    k1: float = DEFAULT_BM25_K1
    b: float = DEFAULT_BM25_B

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0.0:
            raise ValueError("k1 must be a finite value greater than zero")
        if not math.isfinite(self.b) or not 0.0 <= self.b <= 1.0:
            raise ValueError("b must be a finite value between zero and one")


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """One positive lexical match, ranked deterministically by score then ID."""

    item: IndexItem
    score: float


class BM25Scorer:
    """Precompute corpus statistics and score lexical queries against chunks.

    ``IndexItem.embedding_text`` is the text contract shared with the dense
    index.  Only ``kind == \"chunk\"`` rows are accepted: entities and relations
    have their own graph-led routes and must not silently enter hybrid chunk
    recall.
    """

    def __init__(
        self,
        items: Sequence[IndexItem],
        *,
        config: BM25Config | None = None,
    ) -> None:
        self.config = config or BM25Config()
        self.items = tuple(items)
        _validate_chunk_items(self.items)
        self._term_frequencies = tuple(
            Counter(tokenize_lexical(item.embedding_text)) for item in self.items
        )
        self._document_lengths = tuple(
            sum(frequencies.values()) for frequencies in self._term_frequencies
        )
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequencies = Counter(
            term for frequencies in self._term_frequencies for term in frequencies
        )

    @property
    def document_count(self) -> int:
        """Return the immutable number of chunk documents in this scorer."""

        return len(self.items)

    @property
    def average_document_length(self) -> float:
        """Return the number of lexical tokens per chunk, averaged over the corpus."""

        return self._average_document_length

    def score(self, query: str, *, limit: int | None = None) -> tuple[BM25Hit, ...]:
        """Return positive BM25 matches for ``query`` in deterministic order.

        Query term repetition does not multiply a score.  This is the standard
        BM25 query-term treatment and prevents accidental prompt repetition from
        distorting lexical recall.  A blank or out-of-vocabulary query returns no
        candidates.  ``limit=None`` keeps every positive match.
        """

        if limit is not None and limit < 1:
            return ()
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not self.items:
            return ()

        query_terms = tuple(dict.fromkeys(tokenize_lexical(query)))
        if not query_terms:
            return ()

        hits = tuple(
            BM25Hit(item=item, score=self._score_item(frequencies, query_terms))
            for item, frequencies in zip(self.items, self._term_frequencies, strict=True)
        )
        ranked = [hit for hit in hits if hit.score > 0.0]
        ranked.sort(key=lambda hit: (-hit.score, hit.item.object_id))
        return tuple(ranked if limit is None else ranked[:limit])

    def _score_item(self, frequencies: Counter[str], query_terms: Sequence[str]) -> float:
        document_length = sum(frequencies.values())
        length_ratio = (
            document_length / self._average_document_length
            if self._average_document_length > 0.0
            else 0.0
        )
        length_normalizer = self.config.k1 * (1.0 - self.config.b + self.config.b * length_ratio)
        score = 0.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            score += inverse_document_frequency * (
                term_frequency * (self.config.k1 + 1.0) / (term_frequency + length_normalizer)
            )
        return score


def tokenize_lexical(text: str) -> tuple[str, ...]:
    """Tokenize English-like words and CJK runs without external dependencies.

    Words are case-folded and emitted as ``word:`` tokens.  A contiguous CJK run
    yields both character unigrams and adjacent character bigrams, which keeps
    queries such as ``图谱检索`` useful without a language-model tokenizer.  Corpus
    and query text pass through exactly this same deterministic function.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.casefold()):
        value = match.group()
        if _is_cjk_run(value):
            tokens.extend(f"cjk:{character}" for character in value)
            tokens.extend(f"cjk2:{value[index : index + 2]}" for index in range(len(value) - 1))
        else:
            tokens.append(f"word:{value}")
    return tuple(tokens)


def _is_cjk_run(value: str) -> bool:
    return all("\u3400" <= character <= "\u9fff" for character in value)


def _validate_chunk_items(items: Sequence[IndexItem]) -> None:
    object_ids: set[str] = set()
    for item in items:
        if item.kind != "chunk":
            raise ValueError("BM25Scorer accepts only chunk IndexItem rows")
        if item.object_id in object_ids:
            raise ValueError(f"BM25Scorer received duplicate chunk ID: {item.object_id}")
        object_ids.add(item.object_id)


__all__ = [
    "BM25_SCORER_VERSION",
    "DEFAULT_BM25_B",
    "DEFAULT_BM25_K1",
    "LEXICAL_TOKENIZER_VERSION",
    "BM25Config",
    "BM25Hit",
    "BM25Scorer",
    "tokenize_lexical",
]
