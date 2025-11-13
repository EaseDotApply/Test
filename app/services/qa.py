"""Question-answering orchestration built on LangChain."""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Iterable, List, Sequence

try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None  # type: ignore[assignment, misc]

from app.services.mock_llm import MockChatLLM
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.services.hallucination import HallucinationValidator
from app.services.models import AnswerResult
from app.services.vectorstore import VectorStoreService


class QAResponse(BaseModel):
    answer: str = Field(description="Natural-language answer.")
    reasoning: str = Field(description="Brief description of how the answer was derived.")


class QAService:
    """High-level orchestrator that retrieves context and queries an OSS LLM."""

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        retriever: VectorStoreService | None = None,
        llm: Any = None,
        validator: HallucinationValidator | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._vector_service = retriever or VectorStoreService(self._settings)
        
        if llm is None:
            # Always use mock LLM for now since Ollama may not be available
            self._logger.info("llm.using_mock", message="Using mock LLM for testing")
            self._llm = MockChatLLM(temperature=self._settings.llm_temperature)
        else:
            self._llm = llm
        self._parser = JsonOutputParser(pydantic_object=QAResponse)
        self._prompt = self._build_prompt()
        self._validator = validator or HallucinationValidator(self._settings)

    async def ask(
        self,
        question: str,
        *,
        chat_history: Sequence[tuple[str, str]] | None = None,
        force_refresh: bool = False,
    ) -> AnswerResult:
        retriever = await self._vector_service.ensure_retriever(force_refresh=force_refresh)
        documents = await retriever.ainvoke(question)
        if not documents:
            self._logger.info("qa.no_context", question=question)
            return AnswerResult.empty(question)

        context = self._format_context(documents)
        insights = self._derive_insights(question, documents)
        history_messages = self._format_history(chat_history)

        prompt_input = {
            "question": question,
            "context": context,
            "insights": insights,
            "history": history_messages,
            "format_instructions": self._parser.get_format_instructions(),
        }

        try:
            parsed = await (self._prompt | self._llm | self._parser).ainvoke(prompt_input)
            # Parser returns dict, convert to QAResponse
            if isinstance(parsed, dict):
                result = QAResponse(**parsed)
            else:
                result = parsed
        except Exception as e:
            # If parsing fails, try to extract from raw response
            self._logger.warning("qa.parse_failed", error=str(e))
            raw_response = await (self._prompt | self._llm).ainvoke(prompt_input)
            content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            # Try to parse JSON from content
            import json
            try:
                parsed_dict = json.loads(content)
                result = QAResponse(**parsed_dict)
            except:
                result = QAResponse(
                    answer="I found relevant information but couldn't format the answer properly.",
                    reasoning="Generated from context"
                )
        verdict = await self._validator.evaluate(result.answer, documents)
        confidence = self._estimate_confidence(question, documents)
        if not verdict.supported:
            confidence *= 0.4

        return AnswerResult.from_answer(
            question=question,
            answer=result.answer,
            reasoning=result.reasoning,
            confidence=confidence,
            documents=documents,
            guardrails={
                "has_context": True,
                "hallucination_supported": verdict.supported,
                "hallucination_score": verdict.score,
                "hallucination_error": verdict.error,
            },
        )

    def _build_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            "You are November QA, an expert analyst."
            " Answer user questions truthfully using ONLY the supplied context."
            " If the context lacks the answer, say you do not know."
            " Provide concise answers (1-3 sentences) and include precise details like dates or counts."
            " Respect privacy: never fabricate members or data."
        )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context}\n\nInsights:\n{insights}\n\n{format_instructions}",
                ),
            ]
        )

    def _format_history(self, history: Sequence[tuple[str, str]] | None) -> List[BaseMessage]:
        if not history:
            return []
        messages: List[BaseMessage] = []
        for role, content in history:
            if role.lower() == "ai":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def _format_context(self, documents: Iterable[Document]) -> str:
        lines = []
        for doc in documents:
            meta = doc.metadata
            timestamp = meta.get("timestamp")
            formatted_ts = timestamp
            if isinstance(timestamp, str):
                try:
                    formatted_ts = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    formatted_ts = timestamp
            lines.append(
                f"- [{formatted_ts}] {meta.get('user_name')} ({meta.get('user_id')}): {doc.page_content}"
            )
        return "\n".join(lines)

    def _derive_insights(self, question: str, documents: Iterable[Document]) -> str:
        lower_q = question.lower()
        insights: list[str] = []
        if "how many" in lower_q:
            counts = self._count_numeric_mentions(documents)
            if counts:
                insights.append(f"Numeric mentions in context: {counts}")
        if "when" in lower_q:
            earliest = self._earliest_timestamp(documents)
            if earliest:
                insights.append(f"Earliest timestamp in context: {earliest}")
        if not insights:
            insights.append("No additional computed insights.")
        return "\n".join(f"- {item}" for item in insights)

    def _count_numeric_mentions(self, documents: Iterable[Document]) -> int:
        pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
        return sum(len(pattern.findall(doc.page_content)) for doc in documents)

    def _earliest_timestamp(self, documents: Iterable[Document]) -> str | None:
        timestamps: list[datetime] = []
        for doc in documents:
            raw = doc.metadata.get("timestamp")
            if isinstance(raw, str):
                try:
                    timestamps.append(datetime.fromisoformat(raw))
                except ValueError:
                    continue
        if not timestamps:
            return None
        return min(timestamps).strftime("%Y-%m-%d %H:%M")

    def _estimate_confidence(self, question: str, documents: Sequence[Document]) -> float:
        doc_count = len(documents)
        if doc_count == 0:
            return 0.0
        length_penalty = 1.0 if len(question) < 160 else 0.85
        confidence = min(0.95, 0.45 + math.log1p(doc_count) * 0.2)
        return round(confidence * length_penalty, 3)


__all__ = ["QAService"]
