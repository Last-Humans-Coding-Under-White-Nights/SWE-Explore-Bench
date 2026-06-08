"""Datasets module for converting various trajectory formats to unified format."""

from typing import Any

from .models import (
    ToolCall,
    ToolCallFunction,
    TrajectoryInfo,
    TrajectoryMessage,
    UnifiedTrajectory,
)

__all__ = [
    "TrajectoryInfo",
    "TrajectoryMessage",
    "ToolCall",
    "ToolCallFunction",
    "UnifiedTrajectory",
    "load_nebius_dataset",
    "convert_existing_trajectory",
    "load_coderforge_dataset",
    "convert_coderforge_instance",
]


def __getattr__(name: str) -> Any:
    """Lazy import heavy dependencies only when accessed."""
    if name == "load_nebius_dataset":
        from .nebius_loader import load_nebius_dataset

        return load_nebius_dataset
    if name == "convert_existing_trajectory":
        from .existing_adapter import convert_existing_trajectory

        return convert_existing_trajectory
    if name in {"load_coderforge_dataset", "convert_coderforge_instance"}:
        from .coderforge_loader import convert_coderforge_instance, load_coderforge_dataset

        return {
            "load_coderforge_dataset": load_coderforge_dataset,
            "convert_coderforge_instance": convert_coderforge_instance,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
