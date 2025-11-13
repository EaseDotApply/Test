"""Hallucination detection utilities."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from langchain_core.documents import Document

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger

try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # type: ignore[assignment, misc]


@dataclass
class HallucinationVerdict:
    supported: bool
    score: float
    evaluated_documents: int
    error: str | None = None


class HallucinationValidator:
    """Uses natural language inference to detect potential hallucinations."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._pipeline = None

    async def evaluate(self, answer: str, documents: Sequence[Document]) -> HallucinationVerdict:
        if not documents:
            return HallucinationVerdict(supported=False, score=0.0, evaluated_documents=0)

        try:
            scores = await asyncio.to_thread(self._score_documents, answer, documents)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("hallucination.failed", error=str(exc))
            return HallucinationVerdict(supported=True, score=0.5, evaluated_documents=0, error=str(exc))

        best_score = max(scores) if scores else 0.0
        supported = best_score >= self._settings.hallucination_threshold
        return HallucinationVerdict(
            supported=supported,
            score=best_score,
            evaluated_documents=len(scores),
        )

    def _score_documents(self, answer: str, documents: Sequence[Document]) -> list[float]:
        classifier = self._load_pipeline()
        scores: list[float] = []
        for doc in documents:
            result = classifier(
                {
                    "text": doc.page_content,
                    "text_pair": answer,
                },
                truncation=True,
                return_all_scores=True,
            )
            entailment_score = self._extract_entailment(result)
            scores.append(entailment_score)
        return scores

    def _load_pipeline(self):
        if pipeline is None:
            raise ImportError("transformers is not installed. Install it to use hallucination validation.")
        if self._pipeline is None:
            self._logger.info(
                "hallucination.load_model",
                model=self._settings.hallucination_model,
            )
            self._pipeline = pipeline(
                task="text-classification",
                model=self._settings.hallucination_model,
                tokenizer=self._settings.hallucination_model,
                return_all_scores=True,
                device=-1,
            )
        return self._pipeline

    @staticmethod
    def _extract_entailment(result) -> float:
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                candidates = result[0]
            else:
                candidates = result
        else:
            candidates = [result]
        for item in candidates:
            label = str(item.get("label", "")).lower()
            if "entail" in label:
                return float(item.get("score", 0.0))
        return 0.0


__all__ = ["HallucinationValidator", "HallucinationVerdict"]
