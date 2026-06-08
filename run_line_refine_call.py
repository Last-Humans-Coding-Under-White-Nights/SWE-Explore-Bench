from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from line_refine import (
    _apply_refine_reply_to_record,
    _invoke_refine_call,
    _load_bench_records,
)
from models import get_default_openrouter_client


console = Console()


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["instance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _default_checkpoint_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".checkpoint.json")


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tmp.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _remaining_calls(
    bundles: list[dict[str, Any]],
    done_ids: set[str],
    checkpoint: dict[str, Any] | None,
) -> int:
    total = 0
    pending_instance = checkpoint.get("instance_id") if checkpoint else None
    next_call_index = int(checkpoint.get("next_call_index", 0)) if checkpoint else 0
    for bundle in bundles:
        iid = bundle.get("instance_id")
        if iid in done_ids:
            continue
        calls = bundle.get("calls", [])
        if iid == pending_instance:
            total += max(0, len(calls) - next_call_index)
        else:
            total += len(calls)
    return total


def _run_single_call(
    *,
    record: dict[str, Any],
    target: str,
    call: dict[str, Any],
    client: Any,
) -> tuple[dict[str, Any], int, int]:
    input_str = str(call.get("agent_input") or "")
    reply = _invoke_refine_call(client, call)
    updated = _apply_refine_reply_to_record(record, target, call, reply)
    return updated, len(input_str), len(reply)


def main(
    calls_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="由 line_refine.py refine --dry-run -o 生成的 call bundle JSONL",
    ),
    output: Path = typer.Option(
        Path("bench.refined.from_calls.jsonl"),
        "--output",
        "-o",
        help="执行 LLM call 后输出的 bench-like JSONL 路径",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="断点续跑检查点路径，默认使用 output 同目录的 sidecar 文件",
    ),
    nums: int | None = typer.Option(
        None,
        "--nums",
        "-n",
        help="本次最多执行多少个 LLM call（按 call 数，不是按 record 数）",
    ),
    input_cost: float = typer.Option(
        3.0,
        "--input-cost",
        help="输入成本，单位$/M tokens",
    ),
    output_cost: float = typer.Option(
        15.0,
        "--output-cost",
        help="输出成本，单位$/M tokens",
    ),
) -> None:
    """读取 line_refine call bundle，执行 LLM 调用并输出新的 bench-like JSONL。"""
    bundles = list(_load_bench_records(calls_path))
    if not bundles:
        console.print(f"[red]call bundle 为空: {calls_path}[/red]")
        raise typer.Exit(1)

    invalid = [
        bundle.get("instance_id", "<unknown>")
        for bundle in bundles
        if bundle.get("kind") != "line_refine_calls"
    ]
    if invalid:
        console.print(
            "[red]输入文件不是 line_refine call bundle JSONL，以下 instance 非法：[/red]"
        )
        for instance_id in invalid[:20]:
            console.print(f"  {instance_id}")
        raise typer.Exit(1)

    if nums is not None and nums <= 0:
        console.print(f"[red]--nums 必须大于 0，当前为: {nums}[/red]")
        raise typer.Exit(1)

    checkpoint = checkpoint or _default_checkpoint_path(output)
    done_ids = _load_done_ids(output)
    checkpoint_data = _load_checkpoint(checkpoint)

    if checkpoint_data is not None:
        if checkpoint_data.get("calls_path") != str(calls_path.resolve()):
            console.print("[red]checkpoint 与当前 calls_path 不匹配[/red]")
            raise typer.Exit(1)
        if checkpoint_data.get("output_path") != str(output.resolve()):
            console.print("[red]checkpoint 与当前 output 不匹配[/red]")
            raise typer.Exit(1)
        if checkpoint_data.get("instance_id") in done_ids:
            checkpoint.unlink(missing_ok=True)
            checkpoint_data = None

    client = get_default_openrouter_client()
    total_input_chars = 0
    total_output_chars = 0
    total_calls = 0

    output.parent.mkdir(parents=True, exist_ok=True)
    total_calls_budget = _remaining_calls(bundles, done_ids, checkpoint_data)
    if nums is not None:
        total_calls_budget = min(total_calls_budget, nums)

    if total_calls_budget == 0:
        console.print("[yellow]No pending LLM calls to run.[/yellow]")
        return

    with (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress,
        output.open("a") as fout,
    ):
        task = progress.add_task("Running line refine calls...", total=total_calls_budget)
        stop_early = False
        for bundle in bundles:
            instance_id = bundle.get("instance_id", "")
            if instance_id in done_ids:
                continue

            if checkpoint_data is not None and checkpoint_data.get("instance_id") == instance_id:
                current_record = checkpoint_data["record"]
                call_start = int(checkpoint_data.get("next_call_index", 0))
            else:
                current_record = json.loads(
                    json.dumps(bundle.get("record", {}), ensure_ascii=False)
                )
                call_start = 0

            calls = bundle.get("calls", [])
            target = str(bundle.get("target") or "core")
            if not calls:
                fout.write(json.dumps(current_record, ensure_ascii=False) + "\n")
                fout.flush()
                done_ids.add(instance_id)
                checkpoint.unlink(missing_ok=True)
                checkpoint_data = None
                continue

            for call_idx in range(call_start, len(calls)):
                if nums is not None and total_calls >= nums:
                    stop_early = True
                    break

                current_record, in_chars, out_chars = _run_single_call(
                    record=current_record,
                    target=target,
                    call=calls[call_idx],
                    client=client,
                )
                total_input_chars += in_chars
                total_output_chars += out_chars
                total_calls += 1
                progress.advance(task, 1)

                next_call_index = call_idx + 1
                if next_call_index < len(calls):
                    _write_checkpoint(
                        checkpoint,
                        {
                            "schema_version": 1,
                            "calls_path": str(calls_path.resolve()),
                            "output_path": str(output.resolve()),
                            "instance_id": instance_id,
                            "target": target,
                            "next_call_index": next_call_index,
                            "record": current_record,
                        },
                    )
                else:
                    fout.write(json.dumps(current_record, ensure_ascii=False) + "\n")
                    fout.flush()
                    done_ids.add(instance_id)
                    checkpoint.unlink(missing_ok=True)
                    checkpoint_data = None

            if stop_early:
                break

            if checkpoint_data is not None and checkpoint_data.get("instance_id") == instance_id:
                checkpoint_data = None

    estimated_input_tokens = total_input_chars / 2.5
    estimated_output_tokens = total_output_chars / 2.5
    input_cost_total = (estimated_input_tokens / 1_000_000) * input_cost
    output_cost_total = (estimated_output_tokens / 1_000_000) * output_cost
    total_cost = input_cost_total + output_cost_total

    if checkpoint.exists():
        console.print(
            f"[yellow]Checkpoint saved to [bold]{checkpoint}[/bold]. "
            "下次运行会从中断位置继续。[/yellow]"
        )
    else:
        console.print(
            f"[green]Wrote refined benchmark to [bold]{output}[/bold][/green]"
        )
    console.print("\n[bold cyan]Token & Cost Statistics:[/bold cyan]")
    console.print(f"  Completed records in output: [yellow]{len(done_ids)}[/yellow]")
    console.print(f"  LLM calls executed this run: [yellow]{total_calls}[/yellow]")
    console.print(f"  Input chars: [yellow]{total_input_chars:,}[/yellow]")
    console.print(f"  Output chars: [yellow]{total_output_chars:,}[/yellow]")
    console.print(
        f"  Estimated input tokens: [yellow]{estimated_input_tokens:,.0f}[/yellow]"
    )
    console.print(
        f"  Estimated output tokens: [yellow]{estimated_output_tokens:,.0f}[/yellow]"
    )
    console.print(
        f"  Input cost (${input_cost}/M): [green]${input_cost_total:.4f}[/green]"
    )
    console.print(
        f"  Output cost (${output_cost}/M): [green]${output_cost_total:.4f}[/green]"
    )
    console.print(
        f"  [bold]Total estimated cost: [green]${total_cost:.4f}[/green][/bold]"
    )


if __name__ == "__main__":
    typer.run(main)
