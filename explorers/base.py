from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable


@dataclass
class ContextRegion:
    """单个文件中的一段上下文区间。"""

    path: str
    start: int
    end: int
    snippet: str | None = None


@dataclass
class ExplorerResult:
    """Explore 子模块统一返回格式。"""

    instance_id: str
    score: float
    regions: list[ContextRegion]


#: How one case ended. Every result row records one; see README "Case outcomes".
SUCCESS = "success"
TIMEOUT = "timeout"
PROVIDER_ERROR = "provider_error"
INVALID_OUTPUT = "invalid_output"
BINARY_NOT_FOUND = "binary_not_found"
#: An exception no explorer classified (a crash, a missing config file, ...).
ERROR = "error"
OUTCOMES = (SUCCESS, TIMEOUT, PROVIDER_ERROR, INVALID_OUTPUT, BINARY_NOT_FOUND, ERROR)
#: No checkout, so never run and never scored; only the summary counts it.
NOT_ATTEMPTED = "not_attempted"


class ExplorerFailure(RuntimeError):
    """A case that produced no usable answer, and why."""

    def __init__(self, outcome: str, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome


def classify_failure(exc: BaseException) -> str:
    """The outcome recorded for an exception raised by ``explore``."""
    if isinstance(exc, ExplorerFailure):
        return exc.outcome
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
        return TIMEOUT
    return ERROR


@runtime_checkable
class Explorer(Protocol):
    """统一的探索接口，便于后续 SFT / RL 等调优。

    输入：Issue / Query 等自然语言描述；
    输出：与之相关的代码上下文行区间（可以跨多个文件）。
    """

    def explore(self, *, instance_id: str, query: str, top_k: int = 5) -> list[ExplorerResult]:
        ...

