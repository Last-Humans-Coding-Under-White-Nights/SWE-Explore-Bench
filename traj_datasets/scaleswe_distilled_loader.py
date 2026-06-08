"""Loader for Scale-SWE-Distilled parquet files.

Format: each row has {messages: ndarray[{content, loss_mask, role}], data_source, system}
- data_source: "{user}_{repo}_pr{id}" (instance identifier)
- messages: numpy array of message dicts with content/loss_mask/role
- Tool calls are inline XML: <function=str_replace_editor> / <function=execute_bash>
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from .models import TrajectoryInfo, TrajectoryMessage, UnifiedTrajectory

console = Console()

DISTILLED_MODEL = "scaleswe-distilled-deepseek-v3"

_PR_SUFFIX = re.compile(r"_pr\d+$", re.IGNORECASE)


def _parse_repo_from_data_source(data_source: str) -> str:
    """Parse GitHub repo from data_source like 'streamlink_streamlink_pr6220' -> 'streamlink/streamlink'."""
    base = _PR_SUFFIX.sub("", data_source)  # remove _pr{N}
    # split at first underscore: user_repo_with_possible_underscores
    parts = base.split("_", 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}"
    return base


def _extract_issue_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract issue text from the first user message (after <uploaded_files> block)."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = str(content) if content else ""
        if "<uploaded_files>" in content:
            end = content.find("</uploaded_files>")
            if end != -1:
                content = content[end + len("</uploaded_files>"):].strip()
        return content[:4000]
    return ""


def convert_distilled_row(
    data_source: str,
    messages_raw: Any,  # numpy ndarray or list of dicts
    traj_index: int = 0,
) -> UnifiedTrajectory | None:
    """Convert a single Scale-SWE-Distilled row to UnifiedTrajectory."""
    if not data_source or messages_raw is None:
        return None

    msgs: list[dict[str, Any]] = list(messages_raw)
    if not msgs:
        return None

    issue = _extract_issue_from_messages(msgs)

    traj_messages = [
        TrajectoryMessage(
            role=str(m.get("role", "")),
            content=str(m.get("content", "")) if m.get("content") is not None else None,
        )
        for m in msgs
        if m.get("role")
    ]

    # instance_id: append traj index for multi-trajectory instances
    instance_id = data_source if traj_index == 0 else f"{data_source}_t{traj_index}"

    info = TrajectoryInfo(
        repo=_parse_repo_from_data_source(data_source),
        model=DISTILLED_MODEL,
        issue=issue,
        answer="",
        instance_id=instance_id,
        rounds=sum(1 for m in msgs if m.get("role") == "assistant"),
        submission="",
    )
    return UnifiedTrajectory(info=info, traj=traj_messages)


def load_distilled_dataset(
    parquet_dir: Path | str,
    output_dir: Path | str | None = None,
    max_per_instance: int = 1,
) -> list[UnifiedTrajectory]:
    """Load Scale-SWE-Distilled parquet files and convert to unified format.

    Args:
        parquet_dir: Directory containing train-*.parquet files.
        output_dir: If given, save each trajectory as <instance_id>.json.
        max_per_instance: Max trajectories to keep per data_source (default 1 = first only).
    """
    try:
        import pandas as pd
    except ImportError:
        console.print("[red]pandas is required: uv add pandas[/red]")
        return []

    import json

    parquet_dir = Path(parquet_dir)
    part_files = sorted(parquet_dir.glob("*.parquet"))
    if not part_files:
        console.print(f"[red]No parquet files found in {parquet_dir}[/red]")
        return []

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    trajectories: list[UnifiedTrajectory] = []
    per_instance_count: dict[str, int] = {}
    total = skipped = failed = 0

    for part_file in part_files:
        console.log(f"Loading {part_file.name}...")
        try:
            df = pd.read_parquet(part_file)
        except Exception as e:
            console.log(f"[red]Failed to read {part_file.name}: {e}[/red]")
            continue

        for _, row in df.iterrows():
            total += 1
            try:
                ds = str(row["data_source"])
                count = per_instance_count.get(ds, 0)
                if count >= max_per_instance:
                    skipped += 1
                    continue
                traj = convert_distilled_row(ds, row["messages"], traj_index=count)
                if traj is None:
                    skipped += 1
                    continue
                per_instance_count[ds] = count + 1
                trajectories.append(traj)
                if output_dir is not None:
                    out_file = out / f"{traj.info.instance_id}.json"
                    with out_file.open("w") as fp:
                        json.dump(traj.model_dump(), fp, ensure_ascii=False)
            except Exception as e:
                failed += 1
                console.log(f"[red]Failed row {total}: {e}[/red]")

    console.print(
        f"[green]Scale-SWE-Distilled: {len(trajectories)} loaded "
        f"(skipped: {skipped}, failed: {failed}, total rows: {total})[/green]"
    )
    return trajectories


def main(
    parquet_dir: Path = typer.Argument(..., help="Directory with Scale-SWE-Distilled parquet files"),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o"),
    max_per_instance: int = typer.Option(1, "--max-per-instance", help="Max trajectories per instance (default 1)"),
) -> None:
    """Load Scale-SWE-Distilled trajectories and save as unified JSON files."""
    trajs = load_distilled_dataset(parquet_dir, output_dir, max_per_instance)
    console.print(f"Loaded {len(trajs)} trajectories")
    if trajs:
        t = trajs[0]
        console.print(f"  First: {t.info.instance_id}, rounds={t.info.rounds}, msgs={len(t.traj)}")


if __name__ == "__main__":
    typer.run(main)
