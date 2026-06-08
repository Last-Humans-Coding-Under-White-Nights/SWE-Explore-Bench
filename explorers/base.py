from __future__ import annotations

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


@runtime_checkable
class Explorer(Protocol):
    """统一的探索接口，便于后续 SFT / RL 等调优。

    输入：Issue / Query 等自然语言描述；
    输出：与之相关的代码上下文行区间（可以跨多个文件）。
    """

    def explore(self, *, instance_id: str, query: str, top_k: int = 5) -> list[ExplorerResult]:
        ...

