"""Typer-based command line tooling for the QA service."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from app.clients.messages_api import MessagesAPIClient
from app.core.config import AppSettings, get_settings
from app.core.logging import configure_logging, get_logger
from app.evaluation.harness import run_evaluation
from app.pipelines.preprocess import MessagePreprocessor
from app.services.qa import QAService
from app.services.vectorstore import VectorStoreService

cli = typer.Typer(help="Management commands for the November QA service.")
configure_logging()
logger = get_logger("cli")


@cli.command()
def fetch(force: bool = typer.Option(False, help="Force-refresh remote data.")) -> None:
    """Fetch messages from the upstream API and cache them locally."""

    asyncio.run(_fetch(force_refresh=force))


async def _fetch(*, force_refresh: bool) -> None:
    settings = get_settings()
    client = MessagesAPIClient(settings=settings)
    bundle = await client.fetch_messages(force_refresh=force_refresh)
    logger.info("cli.fetch_complete", count=len(bundle.messages), etag=bundle.etag)


@cli.command()
def preprocess(force: bool = typer.Option(False, help="Force regeneration of processed dataset.")) -> None:
    asyncio.run(_preprocess(force_refresh=force))


async def _preprocess(*, force_refresh: bool) -> None:
    settings = get_settings()
    processor = MessagePreprocessor(settings=settings)
    bundle = await processor.run(force_refresh=force_refresh)
    logger.info("cli.preprocess_complete", count=len(bundle.messages))


@cli.command(name="build-index")
def build_index(force: bool = typer.Option(False, help="Force rebuild of the vectorstore.")) -> None:
    asyncio.run(_build_index(force_refresh=force))


async def _build_index(*, force_refresh: bool) -> None:
    settings = get_settings()
    service = VectorStoreService(settings=settings)
    await service.ensure_retriever(force_refresh=force_refresh)
    logger.info("cli.index_ready", documents=service.document_count)


@cli.command()
def ask(
    question: str = typer.Argument(..., help="Question to pose to the QA system."),
    refresh: bool = typer.Option(False, help="Force-refresh data before answering."),
) -> None:
    asyncio.run(_ask(question=question, refresh=refresh))


async def _ask(*, question: str, refresh: bool) -> None:
    settings = get_settings()
    qa_service = QAService(settings=settings)
    answer = await qa_service.ask(question, force_refresh=refresh)
    typer.echo(f"Answer: {answer.answer}")
    typer.echo(f"Confidence: {answer.confidence:.2f}")
    typer.echo(f"Reasoning: {answer.reasoning}")
    if answer.citations:
        typer.echo("Citations:")
        for citation in answer.citations:
            typer.echo(
                f"  - {citation.user_name} @ {citation.timestamp.isoformat()}: {citation.snippet}"
            )


@cli.command()
def evaluate(
    dataset: Optional[Path] = typer.Option(None, exists=False, help="Path to evaluation dataset JSONL."),
    output: Optional[Path] = typer.Option(None, exists=False, help="Destination for evaluation report."),
) -> None:
    asyncio.run(_evaluate(dataset=dataset, output=output))


async def _evaluate(*, dataset: Optional[Path], output: Optional[Path]) -> None:
    summary = await run_evaluation(dataset=dataset, output=output)
    typer.echo("Evaluation Summary:")
    typer.echo(f"  Semantic similarity: {summary.average_similarity:.3f}")
    typer.echo(f"  Lexical accuracy: {summary.lexical_accuracy:.3f}")
    typer.echo(f"  Support rate: {summary.support_rate:.3f}")


@cli.command()
def serve(
    host: Optional[str] = typer.Option(None, help="Override host"),
    port: Optional[int] = typer.Option(None, help="Override port"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev only)."),
) -> None:
    settings = get_settings()
    configure_logging()
    uvicorn.run(
        "app.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


if __name__ == "__main__":
    configure_logging()
    cli()
