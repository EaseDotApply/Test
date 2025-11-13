"""Custom ensemble retriever combining dense and sparse retrieval."""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class EnsembleRetriever(BaseRetriever):
    """Combines multiple retrievers using weighted reciprocal rank fusion."""

    retrievers: Sequence[BaseRetriever]
    weights: Sequence[float]

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        weights: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        if len(retrievers) == 0:
            raise ValueError("At least one retriever must be provided.")
        if weights and len(weights) != len(retrievers):
            raise ValueError("Number of weights must match number of retrievers.")
        super().__init__(
            retrievers=retrievers,
            weights=weights or [1.0 / len(retrievers)] * len(retrievers),
            **kwargs,
        )

    def _get_relevant_documents(self, query: str) -> list[Document]:
        all_docs: list[list[Document]] = []
        for retriever in self.retrievers:
            docs = retriever.invoke(query)
            all_docs.append(docs)

        return self._merge_results(all_docs)

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        all_docs: list[list[Document]] = []
        for retriever in self.retrievers:
            docs = await retriever.ainvoke(query)
            all_docs.append(docs)

        return self._merge_results(all_docs)

    def _merge_results(self, all_docs: list[list[Document]]) -> list[Document]:
        """Merge results using reciprocal rank fusion with weights."""
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for retriever_docs, weight in zip(all_docs, self.weights):
            for rank, doc in enumerate(retriever_docs, start=1):
                doc_id = self._doc_id(doc)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                score = weight / (60 + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Generate a unique ID for a document."""
        metadata = doc.metadata
        if "id" in metadata:
            return str(metadata["id"])
        return f"{doc.page_content[:50]}"


__all__ = ["EnsembleRetriever"]

