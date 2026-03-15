"""Pipeline types and MCTSPipeline export."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    """Shared dependencies injected into every pipeline strategy."""

    http_client: Any
    grammar: str
    db_path: str
    text2sql_url: str
    nlg_url: str
    model_timeout: float
    max_attempts: int
    text2sql_retriever: Any = None


@dataclass
class PipelineResult:
    """Outcome of a single pipeline run — success or structured error."""

    sql: str = ""
    results: dict | None = None
    raw_answer: str = ""
    attempts: int = 0
    error: str | None = None
    error_detail: str | None = None
    metadata: dict | None = field(default_factory=lambda: None)


class PipelineStrategy(ABC):
    """Abstract base for pluggable text-to-SQL pipeline strategies."""

    def __init__(self, ctx: PipelineContext) -> None:
        self.ctx = ctx

    @abstractmethod
    async def run(self, question: str) -> PipelineResult:
        """Execute the pipeline for *question* and return a result."""


from pipelines.mcts import MCTSPipeline  # noqa: E402

__all__ = ["PipelineContext", "PipelineResult", "PipelineStrategy", "MCTSPipeline"]
