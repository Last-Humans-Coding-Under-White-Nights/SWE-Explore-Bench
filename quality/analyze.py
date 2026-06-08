"""分析各实验组的 SWE-bench 评估结果，输出对比报告。

用法:
    uv run python -m quality.analyze \
        --report-dir quality/outputs/reports \
        --output quality/outputs/analysis.json

功能:
    1. 加载各实验组的 SWE-bench report JSON
    2. 计算 pass rate、sufficiency score
    3. 逐实例对比，输出失败归因列表
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _find_reports(report_dir: Path) -> dict[str, dict[str, Any]]:
    """扫描 report_dir 下所有 .json 报告，按文件名分组。"""
    reports: dict[str, dict[str, Any]] = {}
    for f in sorted(report_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            reports[f.stem] = data
        except (json.JSONDecodeError, OSError):
            continue
    return reports


def _extract_mode_from_name(name: str) -> str:
    """从 report 文件名中提取实验模式。"""
    for mode in ("core_only", "core_optional", "no_context"):
        if mode in name:
            return mode
    return name


def compute_analysis(
    reports: dict[str, dict[str, Any]],
    contexts_path: Path | None = None,
) -> dict[str, Any]:
    """计算分析结果。"""
    mode_stats: dict[str, dict[str, Any]] = {}

    for name, report in reports.items():
        mode = _extract_mode_from_name(name)

        total = report.get("submitted_instances", 0)
        resolved = report.get("resolved_instances", 0)
        completed = report.get("completed_instances", 0)
        errors = report.get("error_instances", 0)
        empty = report.get("empty_patch_instances", 0)

        resolved_ids = set(report.get("resolved_ids", []))
        unresolved_ids = set(report.get("unresolved_ids", []))
        error_ids = set(report.get("error_ids", []))

        if mode not in mode_stats:
            mode_stats[mode] = {
                "total": 0,
                "resolved": 0,
                "completed": 0,
                "errors": 0,
                "empty_patch": 0,
                "resolved_ids": set(),
                "unresolved_ids": set(),
                "error_ids": set(),
                "reports": [],
            }

        s = mode_stats[mode]
        s["total"] += total
        s["resolved"] += resolved
        s["completed"] += completed
        s["errors"] += errors
        s["empty_patch"] += empty
        s["resolved_ids"] |= resolved_ids
        s["unresolved_ids"] |= unresolved_ids
        s["error_ids"] |= error_ids
        s["reports"].append(name)

    # pass rate
    pass_rates: dict[str, float] = {}
    for mode, s in mode_stats.items():
        pass_rates[mode] = s["resolved"] / s["total"] if s["total"] > 0 else 0.0

    baseline_rate = pass_rates.get(
        "core_optional", max(pass_rates.values()) if pass_rates else 0.0
    )

    result: dict[str, Any] = {"modes": {}}
    for mode, s in mode_stats.items():
        pr = pass_rates[mode]
        result["modes"][mode] = {
            "total": s["total"],
            "resolved": s["resolved"],
            "completed": s["completed"],
            "errors": s["errors"],
            "empty_patch": s["empty_patch"],
            "pass_rate": round(pr, 4),
            "sufficiency_score": round(pr / baseline_rate, 4) if baseline_rate > 0 else None,
            "reports": s["reports"],
        }

    # core_only vs core_optional 逐实例对比
    core_only_s = mode_stats.get("core_only")
    core_opt_s = mode_stats.get("core_optional")
    if core_only_s and core_opt_s:
        degraded = sorted(core_opt_s["resolved_ids"] - core_only_s["resolved_ids"])
        improved = sorted(core_only_s["resolved_ids"] - core_opt_s["resolved_ids"])
        result["diff_core_vs_optional"] = {
            "degraded_count": len(degraded),
            "degraded_ids": degraded[:100],
            "improved_count": len(improved),
            "improved_ids": improved[:100],
        }

    # 失败归因: core_only 失败 + no_context 也失败 → coding 能力不足
    no_ctx_s = mode_stats.get("no_context")
    if core_only_s and no_ctx_s:
        core_fail = core_only_s["unresolved_ids"] | core_only_s["error_ids"]
        no_ctx_fail = no_ctx_s["unresolved_ids"] | no_ctx_s["error_ids"]
        both_fail = sorted(core_fail & no_ctx_fail)
        only_core_fail = sorted(core_fail - no_ctx_fail - core_only_s["resolved_ids"])
        result["failure_attribution"] = {
            "both_fail_count": len(both_fail),
            "core_only_fail_count": len(only_core_fail),
            "note": "both_fail = likely coding ability issue; core_only_fail = possibly missing context",
        }

    # contexts 统计
    if contexts_path and contexts_path.exists():
        ctx_meta: dict[str, dict] = {}
        with open(contexts_path) as f:
            for line in f:
                r = json.loads(line)
                ctx_meta[r["instance_id"]] = {
                    "num_core_regions": len(r.get("core_context", [])),
                    "num_optional_regions": len(r.get("optional_context", [])),
                    "num_modified_files": len(r.get("modified_files", [])),
                }
        result["context_stats"] = {
            "total_instances": len(ctx_meta),
            "avg_core_regions": round(
                sum(v["num_core_regions"] for v in ctx_meta.values()) / len(ctx_meta), 2
            ) if ctx_meta else 0,
            "avg_optional_regions": round(
                sum(v["num_optional_regions"] for v in ctx_meta.values()) / len(ctx_meta), 2
            ) if ctx_meta else 0,
        }

    return result


def print_summary(analysis: dict[str, Any]) -> None:
    """打印结果摘要表格。"""
    table = Table(title="Bench Quality Evaluation Results")
    table.add_column("Mode", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Resolved", justify="right")
    table.add_column("Pass Rate", justify="right", style="green")
    table.add_column("Sufficiency", justify="right", style="yellow")
    table.add_column("Errors", justify="right", style="red")

    for mode, stats in sorted(analysis["modes"].items()):
        suf = f"{stats['sufficiency_score']:.2%}" if stats["sufficiency_score"] is not None else "N/A"
        table.add_row(
            mode,
            str(stats["total"]),
            str(stats["resolved"]),
            f"{stats['pass_rate']:.2%}",
            suf,
            str(stats["errors"]),
        )

    console.print(table)

    if "diff_core_vs_optional" in analysis:
        d = analysis["diff_core_vs_optional"]
        console.print(f"\n[cyan]Core vs Optional diff:[/cyan]")
        console.print(f"  Degraded (optional OK, core fail): {d['degraded_count']}")
        console.print(f"  Improved (core OK, optional fail): {d['improved_count']}")

    if "failure_attribution" in analysis:
        fa = analysis["failure_attribution"]
        console.print(f"\n[cyan]Failure attribution:[/cyan]")
        console.print(f"  Both fail (likely coding issue): {fa['both_fail_count']}")
        console.print(f"  Core-only fail (possibly missing context): {fa['core_only_fail_count']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(
    report_dir: Path = typer.Option(
        "quality/outputs/reports", "--report-dir", "-r",
        help="SWE-bench 评估报告目录",
    ),
    contexts: Path = typer.Option(
        None, "--contexts", "-c",
        help="contexts.jsonl 路径 (可选，用于统计上下文信息)",
    ),
    output: Path = typer.Option(
        "quality/outputs/analysis.json", "--output", "-o",
        help="分析结果输出路径",
    ),
):
    """分析各实验组的 SWE-bench 评估结果。"""
    reports = _find_reports(report_dir)
    if not reports:
        console.print(f"[red]No reports found in {report_dir}[/red]")
        raise typer.Exit(1)

    console.log(f"Found {len(reports)} report(s)")
    for name in reports:
        console.log(f"  {name}")

    analysis = compute_analysis(reports, contexts)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    console.log(f"Analysis written to {output}")

    print_summary(analysis)


if __name__ == "__main__":
    typer.run(main)
