"""Evaluation harness for automated QA quality checks."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.services.models import AnswerResult
from app.services.qa import QAService

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError as exc:  # pragma: no cover - dependency guaranteed elsewhere
    raise RuntimeError("HuggingFaceEmbeddings is required for evaluation.") from exc


@dataclass
class EvaluationExample:
    question: str
    expected_answer: str


@dataclass
class EvaluationOutcome:
    example: EvaluationExample
    answer: AnswerResult
    semantic_similarity: float
    lexical_overlap: bool
    supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.example.question,
            "expected_answer": self.example.expected_answer,
            "answer": self.answer.answer,
            "confidence": self.answer.confidence,
            "semantic_similarity": self.semantic_similarity,
            "lexical_overlap": self.lexical_overlap,
            "supported": self.supported,
        }


@dataclass
class EvaluationSummary:
    outcomes: list[EvaluationOutcome]

    @property
    def average_similarity(self) -> float:
        scores = [outcome.semantic_similarity for outcome in self.outcomes]
        return float(np.mean(scores)) if scores else 0.0

    @property
    def lexical_accuracy(self) -> float:
        matches = [outcome.lexical_overlap for outcome in self.outcomes]
        return float(np.mean(matches)) if matches else 0.0

    @property
    def support_rate(self) -> float:
        supports = [outcome.supported for outcome in self.outcomes]
        return float(np.mean(supports)) if supports else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "average_similarity": round(self.average_similarity, 4),
            "lexical_accuracy": round(self.lexical_accuracy, 4),
            "support_rate": round(self.support_rate, 4),
            "examples": [outcome.to_dict() for outcome in self.outcomes],
        }


class EvaluationHarness:
    def __init__(
        self,
        qa_service: QAService | None = None,
        *,
        settings: AppSettings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._qa_service = qa_service or QAService(self._settings)
        self._logger = get_logger(self.__class__.__name__)
        self._embeddings = HuggingFaceEmbeddings(model_name=self._settings.embedding_model)

    async def run(self, dataset_path: Path, output_path: Path) -> EvaluationSummary:
        examples = self._load_dataset(dataset_path)
        outcomes = []
        for example in examples:
            self._logger.info("evaluation.run_example", question=example.question)
            answer = await self._qa_service.ask(example.question)
            similarity = self._semantic_similarity(example.expected_answer, answer.answer)
            lexical = self._lexical_overlap(example.expected_answer, answer.answer)
            supported = (answer.guardrails or {}).get("hallucination_supported", False)
            outcomes.append(
                EvaluationOutcome(
                    example=example,
                    answer=answer,
                    semantic_similarity=similarity,
                    lexical_overlap=lexical,
                    supported=supported,
                )
            )

        summary = EvaluationSummary(outcomes=outcomes)
        self._write_report(summary, output_path)
        return summary

    def _load_dataset(self, path: Path) -> list[EvaluationExample]:
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {path}")
        examples: list[EvaluationExample] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(
                EvaluationExample(
                    question=payload["question"],
                    expected_answer=payload["expected_answer"],
                )
            )
        return examples

    def _semantic_similarity(self, expected: str, predicted: str) -> float:
        vectors = self._embeddings.embed_documents([expected, predicted])
        expected_vec, predicted_vec = vectors
        return float(self._cosine_similarity(expected_vec, predicted_vec))

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        a_vec = np.array(a)
        b_vec = np.array(b)
        denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
        if denom == 0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / denom)

    def _lexical_overlap(self, expected: str, predicted: str) -> bool:
        expected_norm = expected.lower().strip()
        predicted_norm = predicted.lower().strip()
        return expected_norm in predicted_norm or predicted_norm in expected_norm

    def _write_report(self, summary: EvaluationSummary, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary.to_dict(), indent=2))
        self._logger.info("evaluation.report_written", path=str(output_path))


async def run_evaluation(dataset: Path | None = None, output: Path | None = None) -> EvaluationSummary:
    settings = get_settings()
    dataset_path = dataset or settings.data_dir / "eval" / "gold.jsonl"
    output_path = output or settings.reports_dir / "evaluation.json"
    harness = EvaluationHarness(settings=settings)
    return await harness.run(dataset_path, output_path)


__all__ = ["EvaluationHarness", "run_evaluation", "EvaluationExample", "EvaluationOutcome", "EvaluationSummary"]
