"""Loader for CoderForge trajectory data.

Supports both raw CoderForge JSONL and the cleaned/enriched JSONL derived from it.
The loader can emit swe-explore unified trajectories and an optional issue map.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from .models import TrajectoryInfo, TrajectoryMessage, UnifiedTrajectory

console = Console()


def _parse_repo_from_image(image: str) -> str:
    """Extract repo from Docker image name like 'qingyangwu/..._spectree-64'."""
    if not image or "_" not in image:
        return ""
    parts = image.split("_")
    if len(parts) < 2:
        return ""
    repo_part = parts[-1]
    return repo_part.rsplit("-", 1)[0] if "-" in repo_part else repo_part


def _extract_issue_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract issue from first user message."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content[:20000]
        if isinstance(content, list):
            return " ".join(str(x) for x in content)[:20000]
        if content is not None:
            return str(content)[:20000]
    return ""


def _parse_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    messages = item.get("messages", "[]")
    if isinstance(messages, str):
        return json.loads(messages)
    return messages if isinstance(messages, list) else []


def _get_instance_id(item: dict[str, Any], use_base_instance_id: bool) -> str:
    trajectory_id = item.get("trajectory_id", "")
    clean_meta = item.get("clean_meta") or {}
    if use_base_instance_id:
        base_id = clean_meta.get("trajectory_base_id")
        if isinstance(base_id, str) and base_id:
            return base_id
    return trajectory_id


def _get_repo(item: dict[str, Any]) -> str:
    clean_meta = item.get("clean_meta") or {}
    derived_repo = clean_meta.get("derived_repo")
    if isinstance(derived_repo, str) and derived_repo:
        return derived_repo
    image = item.get("image", "")
    return _parse_repo_from_image(image)


def convert_coderforge_instance(
    item: dict[str, Any],
    reward_threshold: float = 0.0,
    use_base_instance_id: bool = True,
) -> UnifiedTrajectory | None:
    """Convert CoderForge item to UnifiedTrajectory.

    Returns None if reward < reward_threshold.
    """
    reward = item.get("reward", 0.0)
    if reward < reward_threshold:
        return None

    trajectory_id = item.get("trajectory_id", "")
    instance_id = _get_instance_id(item, use_base_instance_id=use_base_instance_id)
    image = item.get("image", "")
    clean_meta = item.get("clean_meta") or {}

    messages = _parse_messages(item)
    issue = _extract_issue_from_messages(messages)
    repo = _get_repo(item)

    traj_messages = []
    for msg in messages:
        if not isinstance(msg, dict) or not msg.get("role"):
            continue
        traj_messages.append(
            TrajectoryMessage(
                role=msg.get("role"),
                content=msg.get("content"),
                tool_calls=msg.get("tool_calls"),
                name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id"),
            )
        )

    info = TrajectoryInfo(
        repo=repo,
        model="coderforge-openhands",
        issue=issue,
        answer="",
        instance_id=instance_id,
        rounds=sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant"),
        exit_status=item.get("finish_reason"),
        reward=reward,
        raw_image=image,
        source_trajectory_id=trajectory_id,
        clean_meta=clean_meta,
    )

    return UnifiedTrajectory(info=info, traj=traj_messages)


def load_coderforge_dataset(
    jsonl_file: Path | str,
    output_dir: Path | str | None = None,
    reward_threshold: float = 0.0,
    use_base_instance_id: bool = True,
    issue_map_output: Path | str | None = None,
) -> list[UnifiedTrajectory]:
    """Load CoderForge JSONL file and convert to unified format.

    Args:
        jsonl_file: Path to CoderForge JSONL file.
        output_dir: If given, save each trajectory as <trajectory_id>.json.
        reward_threshold: Only keep instances with reward >= this value.
        use_base_instance_id: Use clean_meta.trajectory_base_id as info.instance_id.
        issue_map_output: If given, save {instance_id: issue_text} JSON.
    """
    jsonl_file = Path(jsonl_file)
    if not jsonl_file.exists():
        console.print(f"[red]File not found: {jsonl_file}[/red]")
        return []

    out_dir = None
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    issue_map: dict[str, str] = {}
    trajectories: list[UnifiedTrajectory] = []
    total = skipped = failed = 0

    console.log(f"Loading {jsonl_file.name}...")
    with jsonl_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
                traj = convert_coderforge_instance(
                    item,
                    reward_threshold=reward_threshold,
                    use_base_instance_id=use_base_instance_id,
                )
                if traj is None:
                    skipped += 1
                    continue
                trajectories.append(traj)
                if traj.info.issue and traj.info.instance_id not in issue_map:
                    issue_map[traj.info.instance_id] = traj.info.issue
                if out_dir is not None:
                    source_id = getattr(traj.info, "source_trajectory_id", None) or traj.info.instance_id
                    out_file = out_dir / f"{source_id}.json"
                    with out_file.open("w") as fp:
                        json.dump(traj.model_dump(), fp, ensure_ascii=False)
            except Exception as e:
                failed += 1
                console.log(f"[red]Failed: {e}[/red]")

    if issue_map_output is not None:
        issue_map_path = Path(issue_map_output)
        issue_map_path.parent.mkdir(parents=True, exist_ok=True)
        with issue_map_path.open("w") as f:
            json.dump(issue_map, f, ensure_ascii=False, indent=2)

    console.print(
        f"[green]CoderForge: {len(trajectories)} loaded "
        f"(skipped reward<{reward_threshold}: {skipped}, failed: {failed}, total: {total})[/green]"
    )
    return trajectories


def main(
    jsonl_file: Path = typer.Argument(..., help="CoderForge JSONL file"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    reward_threshold: float = typer.Option(0.0, "--reward-threshold"),
    use_base_instance_id: bool = typer.Option(
        True,
        "--use-base-instance-id/--use-run-instance-id",
        help="Use clean_meta.trajectory_base_id instead of trajectory_id as info.instance_id",
    ),
    issue_map_output: Path | None = typer.Option(
        None,
        "--issue-map-output",
        help="Optional JSON file to save {instance_id: issue_text}",
    ),
) -> None:
    """Load CoderForge trajectories and optionally save as unified JSON files."""
    trajs = load_coderforge_dataset(
        jsonl_file,
        output_dir,
        reward_threshold=reward_threshold,
        use_base_instance_id=use_base_instance_id,
        issue_map_output=issue_map_output,
    )
    console.print(f"Loaded {len(trajs)} trajectories")
    if trajs:
        t = trajs[0]
        console.print(
            f"  First: {t.info.instance_id}, source={getattr(t.info, 'source_trajectory_id', '')}, "
            f"rounds={t.info.rounds}, msgs={len(t.traj)}"
        )


if __name__ == "__main__":
    typer.run(main)
