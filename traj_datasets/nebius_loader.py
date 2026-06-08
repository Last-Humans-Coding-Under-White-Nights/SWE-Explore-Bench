"""Loader for nebius/SWE-rebench-openhands-trajectories dataset."""

import json
import traceback
from pathlib import Path
from typing import Any, Callable

import typer
from datasets import load_dataset
from rich.console import Console

from .models import TrajectoryInfo, TrajectoryMessage, UnifiedTrajectory

console = Console()

# Role to field names mapping
ROLE_TO_FIELDS: dict[str, list[str]] = {
    "system": ["role", "content"],
    "assistant": ["role", "content", "tool_calls"],
    "user": ["role", "content"],
    "tool": ["role", "content", "name", "tool_call_id"],
}


def _deserialize_tool_call_arguments(tool_calls: list[dict[str, Any]]) -> None:
    """Deserialize tool call arguments from JSON strings to dicts (in-place).

    Args:
        tool_calls: List of tool call dictionaries to modify
    """
    for tool_call in tool_calls:
        if function := tool_call.get("function"):
            if args := function.get("arguments"):
                if isinstance(args, str):
                    # Parse JSON string to dict
                    try:
                        function["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        # Keep as string if not valid JSON
                        pass


NEBius_MODEL = "qwen3-coder-480b-instruct"


def _parse_repo_from_instance_id(instance_id: str) -> str:
    """Parse repo as org/reponame from instance_id.

    Instance_id format: org__reponame-issueid (e.g. aio-libs__aiosmtpd-115).
    Returns e.g. aio-libs/aiosmtpd.
    """
    if not instance_id or "__" not in instance_id:
        return ""
    parts = instance_id.split("__", 1)
    if len(parts) < 2:
        return ""
    org, rest = parts[0], parts[1]
    reponame = rest.rsplit("-", 1)[0] if "-" in rest else rest
    return f"{org}/{reponame}"


def _extract_issue_from_trajectory(trajectory: list[dict[str, Any]]) -> str:
    """Extract issue from first user message's <issue_description> XML tag."""
    for msg in trajectory:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    content = item.get("text", "")
                    break
        if not isinstance(content, str):
            continue
        start_tag, end_tag = "<issue_description>", "</issue_description>"
        if start_tag in content and end_tag in content:
            start_idx = content.find(start_tag) + len(start_tag)
            end_idx = content.find(end_tag, start_idx)
            if end_idx > start_idx:
                return content[start_idx:end_idx].strip()
        break
    return ""


def filter_and_deserialize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Filter message fields based on role and deserialize tool calls.

    Args:
        msg: Raw message dictionary from dataset

    Returns:
        Filtered message dictionary with deserialized tool calls
    """
    role = msg["role"]

    # Filter fields based on role
    filtered_msg = {
        field_name: msg[field_name]
        for field_name in ROLE_TO_FIELDS[role]
        if field_name in msg
    }

    # Deserialize tool_calls arguments if present
    if role == "assistant" and (tool_calls := filtered_msg.get("tool_calls")):
        _deserialize_tool_call_arguments(tool_calls)

    return filtered_msg


def convert_nebius_row_to_unified(row: dict[str, Any]) -> UnifiedTrajectory:
    """Convert a single nebius dataset row to unified format.

    Args:
        row: Dataset row containing trajectory and metadata

    Returns:
        UnifiedTrajectory object

    Raises:
        KeyError: If required fields are missing
        ValueError: If data format is invalid
    """
    # Process trajectory messages (only if resolved)
    if not row.get("resolved", False):
        return UnifiedTrajectory(info=TrajectoryInfo(
            repo="", model=NEBius_MODEL, issue="", answer="",
            instance_id=row.get("instance_id", ""), rounds=0,
        ), traj=[])

    trajectory_messages = [
        TrajectoryMessage(**filter_and_deserialize_message(msg))
        for msg in row["trajectory"]
    ]

    instance_id = row.get("instance_id", "")
    repo = _parse_repo_from_instance_id(instance_id)
    issue = _extract_issue_from_trajectory(row.get("trajectory", []))
    git_patch = row.get("git_patch") or ""

    info = TrajectoryInfo(
        repo=repo,
        model=NEBius_MODEL,
        issue=issue,
        answer=git_patch,
        instance_id=instance_id,
        rounds=len(trajectory_messages),
        exit_status=row.get("status"),
        submission=git_patch,
    )

    return UnifiedTrajectory(info=info, traj=trajectory_messages)


def load_nebius_dataset(
    split: str = "train",
    output_dir: Path | str | None = None,
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> list[UnifiedTrajectory]:
    """Load nebius/SWE-rebench-openhands-trajectories and convert to unified format.

    Args:
        split: Dataset split to load (default: "train")
        output_dir: Optional directory to save converted trajectories as JSON files
        filter_fn: Optional filter function to apply to rows before conversion

    Returns:
        List of successfully converted UnifiedTrajectory objects
    """
    console.log(f"Loading nebius dataset (split: {split})...")

    # Load dataset
    ds = load_dataset("nebius/SWE-rebench-openhands-trajectories", split=split)

    # Apply filter if provided
    if filter_fn is not None:
        original_count = len(ds)
        ds = ds.filter(filter_fn)
        console.log(f"Filtered {original_count} -> {len(ds)} rows")

    # Prepare output directory if specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Convert to unified format — keep ALL resolved trajectories (no dedup)
    trajectories: list[UnifiedTrajectory] = []
    failed_rows: list[tuple[int, Exception]] = []

    # Track run index per instance_id for unique file names
    run_counter: dict[str, int] = {}

    for idx, row in enumerate(ds):
        try:
            unified_traj = convert_nebius_row_to_unified(row)
            if not unified_traj.traj:
                # HINT: failed to convert messages or not resolved
                continue
            trajectories.append(unified_traj)

            # Save to file: output_dir/<instance_id>__<run>.json
            if output_dir is not None:
                instance_id = unified_traj.info.instance_id
                run_idx = run_counter.get(instance_id, 0)
                run_counter[instance_id] = run_idx + 1
                output_file = output_path / f"{instance_id}__{run_idx}.json"
                with open(output_file, "w") as f:
                    json.dump(
                        unified_traj.model_dump(), f, indent=2, ensure_ascii=False
                    )

            if (idx + 1) % 5000 == 0:
                console.log(f"Processed {idx + 1}/{len(ds)} rows, {len(trajectories)} resolved trajectories...")

        except Exception as e:
            failed_rows.append((idx, e))
            console.log(f"[red]✗ Failed at row {idx}[/red]")
            traceback.print_exc()

    # Summary
    unique_instances = len({t.info.instance_id for t in trajectories})
    console.print(
        f"\n[green]Successfully converted: {len(trajectories)} resolved trajectories "
        f"({unique_instances} instances, from {len(ds)} rows)[/green]"
    )
    if failed_rows:
        console.print(f"[red]Failed rows: {len(failed_rows)}[/red]")
        for idx, error in failed_rows[:5]:  # Show first 5 failures
            console.print(f"  - Row {idx}: {error}")
        if len(failed_rows) > 5:
            console.print(f"  ... and {len(failed_rows) - 5} more")

    return trajectories


def main(
    split: str = typer.Option(
        "train[:5]",
        "--split",
        "-s",
        help="Dataset split to load (e.g., 'train', 'train[:10]')",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Optional directory to save converted trajectories as JSON files",
    ),
    show_first: bool = typer.Option(
        True,
        "--show-first/--no-show-first",
        help="Show details of the first trajectory",
    ),
) -> None:
    """Load nebius/SWE-rebench-openhands-trajectories and convert to unified format.

    Examples:
        Load first 5 trajectories:
            python -m traj_datasets.nebius_loader

        Load entire train split and save to files:
            python -m traj_datasets.nebius_loader --split train --output-dir ./nebius_converted

        Load first 100 without showing details:
            python -m traj_datasets.nebius_loader --split "train[:100]" --no-show-first
    """
    console.print(f"[cyan]Loading nebius dataset (split: {split})...[/cyan]")
    trajectories = load_nebius_dataset(split=split, output_dir=output_dir)

    console.print(f"\n[green]Loaded {len(trajectories)} trajectories[/green]")

    if trajectories and show_first:
        first_traj = trajectories[0]
        console.print("\n[cyan]First trajectory:[/cyan]")
        console.print(f"  Instance ID: {first_traj.info.instance_id}")
        console.print(f"  Model: {first_traj.info.model}")
        console.print(f"  Rounds: {first_traj.info.rounds}")
        console.print(f"  Messages: {len(first_traj.traj)}")


if __name__ == "__main__":
    typer.run(main)
