"""Vectorstore construction and retrieval utilities."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
except ImportError:
    HuggingFaceEmbeddings = None  # type: ignore[assignment, misc]
    BM25Retriever = None  # type: ignore[assignment, misc]
    FAISS = None  # type: ignore[assignment, misc]

from app.services.ensemble_retriever import EnsembleRetriever

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.domain.processed import ProcessedBundle, ProcessedMessage
from app.pipelines.preprocess import MessagePreprocessor
from app.pipelines.processed_repository import ProcessedRepository


class VectorStoreService:
    """Handles creation and loading of hybrid retrievers."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        preprocessor: MessagePreprocessor | None = None,
        processed_repository: ProcessedRepository | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._preprocessor = preprocessor or MessagePreprocessor(self._settings)
        self._processed_repository = processed_repository or ProcessedRepository(self._settings)
        self._retriever: EnsembleRetriever | None = None
        self._document_count: int = 0

    @property
    def vectorstore_path(self) -> Path:
        return self._settings.vectorstore_dir / "faiss_index"

    @property
    def manifest_path(self) -> Path:
        return self.vectorstore_path.with_suffix(".manifest.json")

    async def ensure_retriever(self, *, force_refresh: bool = False) -> EnsembleRetriever:
        if self._retriever is not None and not force_refresh:
            return self._retriever

        if FAISS is None or HuggingFaceEmbeddings is None or BM25Retriever is None:
            raise ImportError(
                "FAISS, HuggingFaceEmbeddings, and BM25Retriever are required. "
                "Install with: pip install faiss-cpu sentence-transformers langchain-community"
            )

        processed_bundle = await self._ensure_processed(force_refresh=force_refresh)
        embeddings = self._load_embeddings()
        documents = self._to_documents(processed_bundle.messages)
        self._document_count = len(documents)

        if force_refresh or not self.vectorstore_path.exists():
            self._logger.info("vectorstore.build", count=len(processed_bundle.messages))
            faiss_store = FAISS.from_documents(documents, embeddings)
            faiss_store.save_local(str(self.vectorstore_path))
            self._write_manifest(processed_bundle)
        else:
            self._logger.info("vectorstore.load", path=str(self.vectorstore_path))
            faiss_store = FAISS.load_local(
                str(self.vectorstore_path),
                embeddings,
                allow_dangerous_deserialization=True,
            )

        dense_retriever = faiss_store.as_retriever(search_kwargs={"k": 6})
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 8

        self._retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )
        return self._retriever

    async def _ensure_processed(self, *, force_refresh: bool) -> ProcessedBundle:
        if force_refresh:
            return await self._preprocessor.run(force_refresh=True)

        cached = self._processed_repository.load()
        if cached:
            return cached
        return await self._preprocessor.run(force_refresh=False)

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        if HuggingFaceEmbeddings is None:
            raise ImportError("HuggingFaceEmbeddings is required. Install with: pip install sentence-transformers")
        self._logger.info("embeddings.load", model=self._settings.embedding_model)
        return HuggingFaceEmbeddings(model_name=self._settings.embedding_model)

    def _to_documents(self, messages: Sequence[ProcessedMessage]) -> list[Document]:
        docs: list[Document] = []
        for message in messages:
            metadata = {
                "id": message.id,
                "user_id": message.user_id,
                "user_name": message.user_name,
                "timestamp": message.timestamp_utc.isoformat(),
                "temporal_key": message.temporal_key,
                "token_count": message.token_count,
            }
            docs.append(Document(page_content=message.message_clean, metadata=metadata))
        return docs

    def _write_manifest(self, bundle: ProcessedBundle) -> None:
        manifest = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "source_total": bundle.source_total,
            "processed_total": len(bundle.messages),
            "embedding_model": self._settings.embedding_model,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2))
        self._logger.info("vectorstore.manifest", path=str(self.manifest_path))

    @property
    def document_count(self) -> int:
        return self._document_count


__all__ = ["VectorStoreService"]
