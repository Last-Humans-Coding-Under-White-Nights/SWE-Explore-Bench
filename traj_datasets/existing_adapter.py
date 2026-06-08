"""Adapter for converting existing trajectory format to unified format."""

import json
import traceback
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from .models import TrajectoryInfo, TrajectoryMessage, UnifiedTrajectory

console = Console()


def _repo_from_instance_id(instance_id: str) -> str:
    """Parse repo as org/reponame from instance_id (e.g. astropy__astropy-7166 -> astropy/astropy)."""
    if not instance_id or "__" not in instance_id:
        return ""
    parts = instance_id.split("__", 1)
    if len(parts) < 2:
        return ""
    rest = parts[1]
    reponame = rest.rsplit("-", 1)[0] if "-" in rest else rest
    return f"{parts[0]}/{reponame}"


def _extract_instance_info(
    config: dict[str, Any], file_path: Path | None = None
) -> tuple[str, str]:
    """Extract instance_id and repo from config or file path.

    Args:
        config: Configuration dictionary from trajectory
        file_path: Optional path to trajectory file (parent dir name is used as instance_id)

    Returns:
        Tuple of (instance_id, repo). repo is org/reponame (e.g. astropy/astropy).
    """
    instance_id = ""
    repo = ""

    # Priority 1: Extract from file path (parent directory name)
    # e.g., /path/to/astropy__astropy-7166/file.json -> instance_id=astropy__astropy-7166, repo=astropy/astropy
    if file_path:
        parent_name = file_path.parent.name
        if "__" in parent_name:
            instance_id = parent_name
            repo = _repo_from_instance_id(instance_id)
            return instance_id, repo

    # Priority 2: Try to extract from environment.image field
    # e.g., "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-7166:latest"
    if env := config.get("environment"):
        if image := env.get("image"):
            # Look for pattern like "repo_number_repo-issue" in image name
            # Convert underscores to double underscores for instance_id format
            parts = image.split("/")[-1].split(
                ":"
            )  # Get image name without registry and tag
            image_name = parts[0] if parts else ""

            # Try to find instance pattern in image name (e.g., astropy_1776_astropy-7166)
            if "_" in image_name:
                # Split by dots first
                for segment in image_name.split("."):
                    # Look for segments with underscores that might contain instance info
                    if segment.count("_") >= 2:
                        # Convert to instance_id format: repo__repo-issue
                        parts = segment.split("_")
                        if len(parts) >= 3:
                            # Expected format: something_number_repo-issue
                            # or repo_number_repo-issue
                            potential_repo = parts[-3] if len(parts) > 3 else parts[0]
                            issue_part = parts[-1]
                            instance_id = f"{potential_repo}__{issue_part}"
                            repo = _repo_from_instance_id(instance_id)
                            return instance_id, repo

    return instance_id, repo


def _extract_issue_from_messages(
    messages: list[dict[str, Any]], template: str | None = None
) -> str:
    """Extract issue description from messages using template.

    Args:
        messages: List of message dictionaries
        template: Optional template string with {{task}} placeholder

    Returns:
        Issue description string
    """
    if not messages:
        return ""

    # Get first user message content
    first_user_msg = None
    for msg in messages:
        if msg.get("role") == "user":
            first_user_msg = msg
            break

    if not first_user_msg:
        return ""

    content = first_user_msg.get("content", "")

    # Handle list content (e.g., [{"type": "text", "text": "..."}])
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                content = item.get("text", "")
                break

    if not isinstance(content, str):
        return ""

    # If template is provided, try to extract issue using the template
    if template and "{{task}}" in template:
        # Find the portion before and after {{task}} in the template
        template_parts = template.split("{{task}}")
        if len(template_parts) == 2:
            before, after = template_parts
            # Remove common template wrappers to find the actual markers
            # Look for markers in the filled content
            if before in content:
                start_idx = content.find(before) + len(before)
                # Find where the after part starts
                if after and after.strip():
                    # Find first substantial part of 'after' in content
                    after_clean = after.strip()[:50]  # Use first 50 chars as marker
                    end_idx = content.find(after_clean, start_idx)
                    if end_idx > start_idx:
                        return content[start_idx:end_idx].strip()
                else:
                    # No after part, take everything after before marker
                    # But try to find where instructions section starts
                    if "</pr_description>" in content:
                        end_idx = content.find("</pr_description>", start_idx)
                        if end_idx > start_idx:
                            return content[start_idx:end_idx].strip()
                    return content[start_idx:].strip()

    # Fallback: Extract from <pr_description> tags if present
    if "<pr_description>" in content and "</pr_description>" in content:
        start_marker = "<pr_description>"
        # Skip the "Consider the following PR description:" line if present
        start_idx = content.find(start_marker) + len(start_marker)
        end_idx = content.find("</pr_description>", start_idx)
        if end_idx > start_idx:
            issue_text = content[start_idx:end_idx].strip()
            # Remove "Consider the following PR description:" prefix if present
            if issue_text.startswith("Consider the following PR description:"):
                issue_text = issue_text[
                    len("Consider the following PR description:") :
                ].strip()
            return issue_text

    return ""


def _convert_message(msg: dict[str, Any]) -> TrajectoryMessage:
    """Convert a single message to TrajectoryMessage format.

    Args:
        msg: Message dictionary

    Returns:
        TrajectoryMessage object
    """
    msg_dict: dict[str, Any] = {"role": msg["role"]}

    # Add standard fields if present
    if "content" in msg:
        msg_dict["content"] = msg["content"]
    if "tool_calls" in msg and msg["tool_calls"] is not None:
        msg_dict["tool_calls"] = msg["tool_calls"]
    if "name" in msg:
        msg_dict["name"] = msg["name"]
    if "tool_call_id" in msg:
        msg_dict["tool_call_id"] = msg["tool_call_id"]

    # Collect extra metadata
    extra_fields = {
        k: v
        for k, v in msg.items()
        if k not in {"role", "content", "tool_calls", "name", "tool_call_id"}
    }
    if extra_fields:
        msg_dict["meta"] = extra_fields

    return TrajectoryMessage(**msg_dict)


def convert_existing_trajectory(
    traj_data: dict[str, Any] | Path,
) -> UnifiedTrajectory:
    """Convert existing trajectory format to unified format.

    Args:
        traj_data: Either a dict containing trajectory data or a Path to JSON file

    Returns:
        UnifiedTrajectory object

    Raises:
        ValueError: If trajectory data is invalid or missing required fields
        FileNotFoundError: If provided path doesn't exist
    """
    # Track file path for instance_id extraction
    file_path = None

    # Load from file if Path is provided
    if isinstance(traj_data, Path):
        file_path = traj_data
        if not traj_data.exists():
            raise FileNotFoundError(f"Trajectory file not found: {traj_data}")
        with open(traj_data) as f:
            traj_data = json.load(f)

    # Validate basic structure
    if not isinstance(traj_data, dict):
        raise ValueError("Trajectory data must be a dictionary")

    info_dict = traj_data.get("info", {})
    config = info_dict.get("config", {})
    messages = traj_data.get("messages", [])

    # Handle both "messages" and "traj" field names
    if not messages and "traj" in traj_data:
        messages = traj_data["traj"]

    # Extract instance and repo information (with file path hint)
    instance_id, repo = _extract_instance_info(config, file_path)

    # Fallback: try to get from info dict
    if not instance_id:
        instance_id = info_dict.get("instance_id", "")
    if instance_id and not repo:
        repo = _repo_from_instance_id(instance_id)

    # Extract model name
    model_name = "unknown"
    if model_config := config.get("model"):
        model_name = model_config.get("model_name", "unknown")

    # Extract issue using template if available
    template = None
    if agent_config := config.get("agent"):
        template = agent_config.get("instance_template")

    issue = _extract_issue_from_messages(messages, template)
    answer = info_dict.get("submission", "")

    # Count rounds (number of assistant messages)
    rounds = sum(1 for msg in messages if msg.get("role") == "assistant")

    # Create info object
    info = TrajectoryInfo(
        repo=repo,
        model=model_name,
        issue=issue,
        answer=answer,
        instance_id=instance_id,
        rounds=rounds,
        exit_status=info_dict.get("exit_status"),
        submission=info_dict.get("submission"),
        model_stats=info_dict.get("model_stats"),
        config=config,
    )

    # Convert messages
    trajectory_messages = [_convert_message(msg) for msg in messages]

    return UnifiedTrajectory(info=info, traj=trajectory_messages)


def convert_existing_trajectory_dir(
    input_dir: Path | str,
    output_dir: Path | str,
    pattern: str = "**/*.traj.json",
) -> list[UnifiedTrajectory]:
    """Convert all trajectory files in a directory to unified format.

    Args:
        input_dir: Directory containing existing trajectory files
        output_dir: Directory to save converted trajectories
        pattern: Glob pattern for finding trajectory files

    Returns:
        List of successfully converted UnifiedTrajectory objects
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    trajectories: list[UnifiedTrajectory] = []
    failed_files: list[tuple[Path, Exception]] = []

    for traj_file in input_dir.glob(pattern):
        try:
            # Convert trajectory
            unified_traj = convert_existing_trajectory(traj_file)

            # Determine output path (preserve directory structure)
            rel_path = traj_file.relative_to(input_dir)
            output_file = output_dir / rel_path.with_suffix(".json")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save converted trajectory
            with open(output_file, "w") as f:
                json.dump(unified_traj.model_dump(), f, indent=2, ensure_ascii=False)

            trajectories.append(unified_traj)
            console.log(f"✓ Converted: {traj_file.name}")

        except Exception as e:
            failed_files.append((traj_file, e))
            console.log(f"[red]✗ Failed: {traj_file.name}[/red]")
            traceback.print_exc()

    # Summary
    console.print(f"\n[green]Successfully converted: {len(trajectories)}[/green]")
    if failed_files:
        console.print(f"[red]Failed: {len(failed_files)}[/red]")
        for file, error in failed_files:
            console.print(f"  - {file.name}: {error}")

    return trajectories


def main(
    path: Path = typer.Argument(
        ...,
        help="Path to trajectory file or directory to convert",
        exists=True,
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path (file or directory). If not specified, auto-generated based on input.",
    ),
) -> None:
    """Convert existing trajectory format to unified format.

    For single files: outputs to <name>.unified.json by default.
    For directories: outputs to <name>_unified/ by default.
    """
    if path.is_file():
        # Convert single file
        unified_traj = convert_existing_trajectory(path)
        console.print("[green]Converted trajectory:[/green]")
        console.print(f"  Instance ID: {unified_traj.info.instance_id}")
        console.print(f"  Repo: {unified_traj.info.repo}")
        console.print(f"  Model: {unified_traj.info.model}")
        console.print(f"  Rounds: {unified_traj.info.rounds}")
        console.print(f"  Messages: {len(unified_traj.traj)}")

        # Save to output file
        output_file = output or path.with_suffix(".unified.json")
        with open(output_file, "w") as f:
            json.dump(unified_traj.model_dump(), f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Saved to: {output_file}[/green]")

    elif path.is_dir():
        # Convert directory
        output_dir = output or (path.parent / f"{path.name}_unified")
        trajectories = convert_existing_trajectory_dir(path, output_dir)
        console.print(
            f"\n[green]Converted {len(trajectories)} trajectories to {output_dir}[/green]"
        )

    else:
        console.print(f"[red]Error: {path} is not a file or directory[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
