"""Loader for nebius/SWE-agent-trajectories dataset.

SWE-agent trajectories use a different message format than OpenHands:
- Roles: "system", "user", "ai" (not "assistant")
- Content: in "text" field (user/ai) or "system_prompt" (system), not "content"
- Issue: embedded in first user message between "ISSUE:\\n" and "INSTRUCTIONS:"
- No tool_calls: bash commands are inline in the assistant text
- Commands like ``open file.py`` are normalized to ``cat file.py`` in ```bash blocks
  so that bench_build.py can detect file reads without modification.
"""

import json
import re
import traceback
from pathlib import Path
from typing import Any, Callable

import typer
from datasets import load_dataset
from rich.console import Console

from .models import TrajectoryInfo, TrajectoryMessage, UnifiedTrajectory

console = Console()

# Match SWE-agent code blocks: ``` (with optional language tag) ... ```
_CODE_BLOCK_RE = re.compile(r"```(\w*)\s*\n(.*?)\n```", re.DOTALL)


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
    """Extract issue from first user message.

    SWE-agent format: first user message contains the issue between
    "ISSUE:\\n" and "INSTRUCTIONS:" markers.
    """
    for msg in trajectory:
        if msg.get("role") != "user":
            continue
        text = msg.get("text") or ""
        if not text:
            continue

        # Look for "ISSUE:\n" marker
        issue_marker = "ISSUE:\n"
        if issue_marker not in text:
            # Fallback: try without newline
            issue_marker = "ISSUE:"
            if issue_marker not in text:
                break

        issue_start = text.find(issue_marker) + len(issue_marker)

        # Find end of issue section
        end_markers = ["INSTRUCTIONS:", "\nINSTRUCTIONS"]
        issue_end = len(text)
        for marker in end_markers:
            pos = text.find(marker, issue_start)
            if pos > 0:
                issue_end = pos
                break

        return text[issue_start:issue_end].strip()

    return ""


def _normalize_swe_agent_command(cmd: str) -> str:
    """Normalize a SWE-agent command to a format bench_build.py can parse.

    Mappings:
        open file.py [line]  ->  cat file.py
        find_file "pat" dir  ->  grep -rn "pat" dir
        search_dir "pat" dir ->  grep -rn "pat" dir
        grep ...             ->  grep ... (unchanged)
        others               ->  unchanged
    """
    stripped = cmd.strip()
    parts = stripped.split()
    if not parts:
        return cmd

    verb = parts[0]

    if verb == "open" and len(parts) >= 2:
        # open file.py [line_number] -> cat file.py
        filepath = parts[1]
        return f"cat {filepath}"

    if verb == "find_file" and len(parts) >= 2:
        # find_file "pattern" [dir] -> grep -rn "pattern" [dir]
        pattern = parts[1]
        directory = parts[2] if len(parts) >= 3 else "."
        return f"grep -rn {pattern} {directory}"

    if verb == "search_dir" and len(parts) >= 2:
        # search_dir "pattern" [dir] -> grep -rn "pattern" [dir]
        pattern = parts[1]
        directory = parts[2] if len(parts) >= 3 else "."
        return f"grep -rn {pattern} {directory}"

    return cmd


def _normalize_code_blocks(content: str) -> str:
    """Normalize code blocks in assistant content for bench_build compatibility.

    1. Add ``bash`` language tag to untagged code blocks.
    2. Normalize read commands (open -> cat, etc.) inside the blocks.
    """

    def _replace_block(m: re.Match) -> str:
        lang = m.group(1)
        body = m.group(2).strip()
        normalized_body = _normalize_swe_agent_command(body)
        # Always emit ```bash so bench_build's _BASH_BLOCK regex matches
        return f"```bash\n{normalized_body}\n```"

    return _CODE_BLOCK_RE.sub(_replace_block, content)


def _convert_swe_agent_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a SWE-agent message to unified format.

    SWE-agent format:
        - role: "system" | "user" | "ai"
        - text: message content (for user and ai roles)
        - system_prompt: content for system role
        - mask: whether the message is from the model (True for ai)
        - cutoff_date: date string

    Unified format:
        - role: "system" | "user" | "assistant"
        - content: message text (with normalized code blocks for assistant)
    """
    raw_role = msg.get("role", "user")

    # Map SWE-agent roles to unified roles
    role = "assistant" if raw_role == "ai" else raw_role

    # Extract content from the appropriate field
    if raw_role == "system":
        content = msg.get("system_prompt") or msg.get("text") or ""
    else:
        content = msg.get("text") or ""

    # Normalize code blocks in assistant messages for bench_build compatibility
    if role == "assistant" and content:
        content = _normalize_code_blocks(content)

    return {"role": role, "content": content}


def convert_swe_agent_row_to_unified(row: dict[str, Any]) -> UnifiedTrajectory:
    """Convert a single SWE-agent row to unified trajectory format.

    Args:
        row: A row from nebius/SWE-agent-trajectories dataset

    Returns:
        UnifiedTrajectory object
    """
    instance_id = row.get("instance_id", "")
    model_name = row.get("model_name", "unknown")
    repo = _parse_repo_from_instance_id(instance_id)

    # Extract trajectory messages
    trajectory = row.get("trajectory", [])
    messages: list[TrajectoryMessage] = []

    if isinstance(trajectory, list):
        for msg in trajectory:
            if isinstance(msg, dict):
                converted = _convert_swe_agent_message(msg)
                if converted.get("content"):
                    messages.append(TrajectoryMessage(**converted))

    # Extract issue from trajectory
    issue = _extract_issue_from_trajectory(trajectory)

    # Create info
    answer = row.get("generated_patch") or ""
    if not isinstance(answer, str):
        answer = ""

    info = TrajectoryInfo(
        repo=repo,
        model=model_name,
        issue=issue,
        instance_id=instance_id,
        answer=answer,
        rounds=sum(1 for m in messages if m.role == "assistant"),
        exit_status=row.get("exit_status") or "",
        submission=answer,
        model_stats=None,
        config=None,
    )

    return UnifiedTrajectory(info=info, traj=messages)


def load_swe_agent_dataset(
    split: str = "train[:5]",
    output_dir: Path | str | None = None,
    filter_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> list[UnifiedTrajectory]:
    """Load nebius/SWE-agent-trajectories and convert to unified format.

    The dataset contains ~80k rows: 3 models x multiple runs per instance.
    All trajectories are kept (no deduplication) because bench_build.py uses
    multiple trajectories per instance to compute core regions (intersection).

    When output_dir is given, files are organized as:
        output_dir/<model_name>/<instance_id>__<run_index>.json

    Args:
        split: Dataset split to load (e.g., 'train', 'train[:10]')
        output_dir: Optional directory to save converted trajectories as JSON files
        filter_fn: Optional filter function to apply to dataset rows

    Returns:
        List of UnifiedTrajectory objects
    """
    console.print(
        f"[cyan]Loading nebius/SWE-agent-trajectories (split: {split})...[/cyan]"
    )

    try:
        ds = load_dataset("nebius/SWE-agent-trajectories", split=split)
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        raise

    console.print(f"[green]Loaded {len(ds)} rows[/green]")

    # Apply filter if provided
    if filter_fn is not None:
        original_count = len(ds)
        ds = ds.filter(filter_fn)
        console.print(f"Filtered {original_count} -> {len(ds)} rows")

    # Prepare output directory if specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Convert to unified format — keep ALL trajectories (no dedup)
    trajectories: list[UnifiedTrajectory] = []
    failed_rows: list[tuple[int, Exception]] = []

    # Track run index per (model, instance_id) for unique file names
    run_counter: dict[tuple[str, str], int] = {}

    for idx, row in enumerate(ds):
        try:
            unified_traj = convert_swe_agent_row_to_unified(row)
            if not unified_traj.traj:
                continue
            trajectories.append(unified_traj)

            # Save to file: output_dir/<model>/<instance_id>__<run>.json
            if output_dir is not None:
                instance_id = unified_traj.info.instance_id
                model_name = unified_traj.info.model
                key = (model_name, instance_id)
                run_idx = run_counter.get(key, 0)
                run_counter[key] = run_idx + 1

                model_dir = output_path / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                output_file = model_dir / f"{instance_id}__{run_idx}.json"
                with open(output_file, "w") as f:
                    json.dump(
                        unified_traj.model_dump(), f, indent=2, ensure_ascii=False
                    )

            if (idx + 1) % 5000 == 0:
                console.log(
                    f"Processed {idx + 1}/{len(ds)} rows, "
                    f"{len(trajectories)} trajectories saved..."
                )

        except Exception as e:
            failed_rows.append((idx, e))
            if len(failed_rows) <= 5:
                console.log(f"[red]Failed at row {idx}[/red]")
                traceback.print_exc()

    # Summary
    unique_instances = len({t.info.instance_id for t in trajectories})
    unique_models = len({t.info.model for t in trajectories})
    console.print(
        f"\n[green]Successfully converted: {len(trajectories)} trajectories "
        f"({unique_instances} instances, {unique_models} models, "
        f"from {len(ds)} rows)[/green]"
    )
    if failed_rows:
        console.print(f"[red]Failed rows: {len(failed_rows)}[/red]")
        for idx, error in failed_rows[:5]:
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
    """Load nebius/SWE-agent-trajectories and convert to unified format.

    Examples:
        Load first 5 trajectories:
            python -m traj_datasets.swe_agent_loader

        Load entire train split and save to files:
            python -m traj_datasets.swe_agent_loader --split train --output-dir ./swe_agent_converted

        Load first 100 without showing details:
            python -m traj_datasets.swe_agent_loader --split "train[:100]" --no-show-first
    """
    console.print(
        f"[cyan]Loading nebius/SWE-agent-trajectories (split: {split})...[/cyan]"
    )
    trajectories = load_swe_agent_dataset(split=split, output_dir=output_dir)

    console.print(f"\n[green]Loaded {len(trajectories)} trajectories[/green]")

    if trajectories and show_first:
        first_traj = trajectories[0]
        console.print("\n[cyan]First trajectory:[/cyan]")
        console.print(f"  Instance ID: {first_traj.info.instance_id}")
        console.print(f"  Model: {first_traj.info.model}")
        console.print(f"  Repo: {first_traj.info.repo}")
        console.print(f"  Issue: {first_traj.info.issue[:100]}...")
        console.print(f"  Rounds: {first_traj.info.rounds}")
        console.print(f"  Messages: {len(first_traj.traj)}")


if __name__ == "__main__":
    typer.run(main)
