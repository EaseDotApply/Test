"""Mock LLM for testing when Ollama is not available."""
from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class MockChatLLM(BaseChatModel):
    """Simple mock LLM that returns structured answers based on context."""

    model_name: str = Field(default="mock-llm")
    temperature: float = Field(default=0.2)

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        answer = self._generate_answer(messages)
        message = AIMessage(content=answer)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        answer = self._generate_answer(messages)
        message = AIMessage(content=answer)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _generate_answer(self, messages: List[BaseMessage]) -> str:
        """Generate a simple answer based on the last user message."""
        if not messages:
            return '{"answer": "I need more context to answer.", "reasoning": "No input provided."}'

        last_message = messages[-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)

        if "when" in content.lower() and "layla" in content.lower() and "london" in content.lower():
            return '{"answer": "Based on the messages, Layla is planning her trip to London in June.", "reasoning": "Found mention of London trip in June in Layla\'s messages."}'
        elif "how many" in content.lower() and "car" in content.lower():
            return '{"answer": "Based on the messages, the person mentioned having multiple cars.", "reasoning": "Found references to cars in the messages."}'
        elif "favorite" in content.lower() or "restaurant" in content.lower():
            return '{"answer": "Based on the messages, several restaurants were mentioned as favorites.", "reasoning": "Found restaurant mentions in the context."}'
        else:
            return '{"answer": "I found relevant information in the messages, but need more specific details to provide a precise answer.", "reasoning": "Context available but question needs refinement."}'


__all__ = ["MockChatLLM"]

