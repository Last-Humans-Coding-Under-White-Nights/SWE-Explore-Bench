"""Loader for ScaleSWE trajectory JSONL files.

ScaleSWE format: each line is {instance_id: {score, agent_run_result, patch, ...}}
agent_run_result uses OpenHands inline-XML style (no tool_calls field).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from .models import TrajectoryInfo, TrajectoryMessage, UnifiedTrajectory

console = Console()

SCALESWE_MODEL = "scaleswe-deepseek-v3"

_DEFAULT_PARTS = [
    "Scale-SWE-Agent_SWE-Bench-Verified_result_part1.jsonl",
    "Scale-SWE-Agent_SWE-Bench-Verified_result_part2.jsonl",
    "Scale-SWE-Agent_SWE-Bench-Verified_result_part3.jsonl",
    "Scale-SWE-Agent_SWE-Bench-Verified_result_part4.jsonl",
]


def _parse_repo_from_instance_id(instance_id: str) -> str:
    if not instance_id or "__" not in instance_id:
        return ""
    org, rest = instance_id.split("__", 1)
    repo = rest.rsplit("-", 1)[0] if "-" in rest else rest
    return f"{org}/{repo}"


def _extract_issue_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract issue text from the first user message (after <uploaded_files> block)."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        # Strip <uploaded_files>...</uploaded_files> header if present
        if "<uploaded_files>" in content:
            end = content.find("</uploaded_files>")
            if end != -1:
                content = content[end + len("</uploaded_files>"):].strip()
        return content[:4000]  # cap length
    return ""


def convert_scaleswe_instance(
    instance_id: str,
    val: dict[str, Any],
    score_threshold: float = 1.0,
) -> UnifiedTrajectory | None:
    """Convert a single ScaleSWE instance dict to UnifiedTrajectory.

    Returns None if score < score_threshold.
    """
    score = val.get("score", 0.0)
    if score < score_threshold:
        return None

    messages: list[dict[str, Any]] = val.get("agent_run_result", [])
    issue = _extract_issue_from_messages(messages)
    patch = val.get("patch", "")

    traj_messages = [TrajectoryMessage(**{k: v for k, v in msg.items()
                                          if k in ("role", "content")})
                     for msg in messages if isinstance(msg, dict) and msg.get("role")]

    info = TrajectoryInfo(
        repo=_parse_repo_from_instance_id(instance_id),
        model=SCALESWE_MODEL,
        issue=issue,
        answer=patch,
        instance_id=instance_id,
        rounds=sum(1 for m in messages if m.get("role") == "assistant"),
        submission=patch,
    )
    return UnifiedTrajectory(info=info, traj=traj_messages)


def load_scaleswe_dataset(
    trajs_dir: Path | str,
    output_dir: Path | str | None = None,
    score_threshold: float = 1.0,
) -> list[UnifiedTrajectory]:
    """Load ScaleSWE JSONL files and convert to unified format.

    Args:
        trajs_dir: Directory containing the 4 part JSONL files.
        output_dir: If given, save each trajectory as <instance_id>.json.
        score_threshold: Only keep instances with score >= this value.
    """
    trajs_dir = Path(trajs_dir)
    part_files = sorted(trajs_dir.glob("*.jsonl"))
    if not part_files:
        console.print(f"[red]No JSONL files found in {trajs_dir}[/red]")
        return []

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    trajectories: list[UnifiedTrajectory] = []
    total = skipped = failed = 0

    for part_file in part_files:
        console.log(f"Loading {part_file.name}...")
        with part_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    data = json.loads(line)
                    # Each line: {instance_id: {...}}
                    for instance_id, val in data.items():
                        traj = convert_scaleswe_instance(instance_id, val, score_threshold)
                        if traj is None:
                            skipped += 1
                            continue
                        trajectories.append(traj)
                        if output_dir is not None:
                            out_file = out / f"{instance_id}.json"
                            with out_file.open("w") as fp:
                                json.dump(traj.model_dump(), fp, ensure_ascii=False)
                except Exception as e:
                    failed += 1
                    console.log(f"[red]Failed: {e}[/red]")

    console.print(
        f"[green]ScaleSWE: {len(trajectories)} loaded "
        f"(skipped score<{score_threshold}: {skipped}, failed: {failed}, total: {total})[/green]"
    )
    return trajectories


def main(
    trajs_dir: Path = typer.Argument(..., help="Directory with ScaleSWE JSONL files"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    score_threshold: float = typer.Option(1.0, "--score-threshold"),
) -> None:
    """Load ScaleSWE trajectories and optionally save as unified JSON files."""
    trajs = load_scaleswe_dataset(trajs_dir, output_dir, score_threshold)
    console.print(f"Loaded {len(trajs)} trajectories")
    if trajs:
        t = trajs[0]
        console.print(f"  First: {t.info.instance_id}, rounds={t.info.rounds}, msgs={len(t.traj)}")


if __name__ == "__main__":
    typer.run(main)
