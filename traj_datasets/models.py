"""Pydantic models for unified trajectory format."""

from typing import Any

from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    """Function call within a tool call."""

    name: str
    arguments: dict[str, Any] | str  # Can be dict or JSON string


class ToolCall(BaseModel):
    """Tool call in assistant message."""

    id: str | None = None
    type: str = "function"
    function: ToolCallFunction


class TrajectoryMessage(BaseModel):
    """Single message in trajectory."""

    role: str  # "system", "assistant", "user", "tool"
    content: str | list[Any] | None = None
    # Optional fields based on role
    tool_calls: list[ToolCall] | None = None  # For assistant messages
    name: str | None = None  # For tool messages
    tool_call_id: str | None = None  # For tool messages
    meta: dict[str, Any] | None = None  # For additional metadata

    class Config:
        extra = "allow"  # Allow additional fields


class TrajectoryInfo(BaseModel):
    """Metadata information for a trajectory."""

    repo: str
    model: str
    issue: str
    answer: str
    instance_id: str
    rounds: int
    # Allow additional metadata fields
    exit_status: str | None = None
    submission: str | None = None
    model_stats: dict[str, Any] | None = None
    config: dict[str, Any] | None = None

    class Config:
        extra = "allow"  # Allow additional fields


class UnifiedTrajectory(BaseModel):
    """Unified trajectory format for swe-explore."""

    info: TrajectoryInfo
    traj: list[TrajectoryMessage] = Field(default_factory=list)

    class Config:
        extra = "allow"  # Allow additional fields
