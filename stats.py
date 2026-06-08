from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import typer
from rich.console import Console
from rich.table import Table


console = Console()
app = typer.Typer(rich_markup_mode="rich")


@dataclass
class Region:
    path: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start + 1)


def _load_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_regions(record: dict[str, Any]) -> Iterable[Region]:
    gt = record.get("ground_truth") or {}
    for r in gt.get("read_core_regions", []) or []:
        try:
            yield Region(
                path=str(r["path"]),
                start=int(r["start"]),
                end=int(r["end"]),
            )
        except Exception:
            continue


def _bucket_length(n: int) -> str:
    # 对行区间长度做粗粒度分桶
    if n <= 0:
        return "0"
    bounds = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    for b in bounds:
        if n <= b:
            return f"≤{b}"
    return ">1000"


@app.command()
def summary(
    bench_path: Path = typer.Argument(
        Path("bench.jsonl"),
        exists=True,
        readable=True,
        help="要分析的基准文件（可为原始 / 粗修 / 精修版本）",
    ),
    dump_json: Path | None = typer.Option(
        None,
        "--dump-json",
        "-d",
        help="可选：将统计结果以 JSON 格式写入文件，便于后续画图",
    ),
) -> None:
    """对基准文件做整体统计：实例数、文件数、行区间分布等。"""
    num_instances = 0
    num_regions = 0
    num_files = 0

    regions_per_instance: Counter[int] = Counter()
    files_per_instance: Counter[int] = Counter()
    length_buckets: Counter[str] = Counter()

    per_repo_instances: Counter[str] = Counter()

    # repo -> 文件 -> 总行数（这里按区间长度粗略估计）
    repo_file_line_approx: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rec in _load_records(bench_path):
        num_instances += 1
        instance_id = str(rec.get("instance_id", ""))
        repo_name = instance_id.split("__")[0] if instance_id else "unknown"
        per_repo_instances[repo_name] += 1

        regions = list(_iter_regions(rec))
        num_regions += len(regions)
        regions_per_instance[len(regions)] += 1

        files = {r.path for r in regions}
        num_files += len(files)
        files_per_instance[len(files)] += 1

        for r in regions:
            length_buckets[_bucket_length(r.length)] += 1
            repo_file_line_approx[repo_name][r.path] += r.length

    # 控制台输出
    console.rule("[bold cyan]Benchmark Summary[/bold cyan]")
    console.print(f"[bold]基准文件[/bold]: {bench_path}")
    console.print(f"[bold]实例数[/bold]: {num_instances}")
    console.print(f"[bold]总核心区间数[/bold]: {num_regions}")
    console.print(f"[bold]累计涉及文件数（去重按实例内）[/bold]: {num_files}")

    # 行区间长度分布
    table_len = Table(title="行区间长度分布", show_lines=False)
    table_len.add_column("长度桶", justify="right")
    table_len.add_column("区间数", justify="right")
    for bucket, cnt in sorted(
        length_buckets.items(),
        key=lambda kv: (
            float("inf") if kv[0].startswith(">") else int(kv[0].lstrip("≤")),
        ),
    ):
        table_len.add_row(bucket, str(cnt))
    console.print(table_len)

    # 每实例文件数分布
    table_files = Table(title="每实例涉及文件数分布", show_lines=False)
    table_files.add_column("文件数", justify="right")
    table_files.add_column("实例数", justify="right")
    for k in sorted(files_per_instance):
        table_files.add_row(str(k), str(files_per_instance[k]))
    console.print(table_files)

    # 每实例行区间数分布
    table_regions = Table(title="每实例核心区间数分布", show_lines=False)
    table_regions.add_column("区间数", justify="right")
    table_regions.add_column("实例数", justify="right")
    for k in sorted(regions_per_instance):
        table_regions.add_row(str(k), str(regions_per_instance[k]))
    console.print(table_regions)

    # 按 repo 视角的大致复杂度
    table_repo = Table(title="按仓库的行上下文粗略规模（近似）", show_lines=False)
    table_repo.add_column("Repo", justify="left")
    table_repo.add_column("实例数", justify="right")
    table_repo.add_column("近似总核心行数", justify="right")
    for repo, inst_cnt in per_repo_instances.most_common():
        approx_lines = sum(repo_file_line_approx[repo].values())
        table_repo.add_row(repo, str(inst_cnt), str(approx_lines))
    console.print(table_repo)

    # dump 为 JSON
    if dump_json is not None:
        stats_obj = {
            "bench_path": str(bench_path),
            "num_instances": num_instances,
            "num_regions": num_regions,
            "num_files": num_files,
            "regions_per_instance": dict(regions_per_instance),
            "files_per_instance": dict(files_per_instance),
            "length_buckets": dict(length_buckets),
            "per_repo_instances": dict(per_repo_instances),
            "repo_file_line_approx": {
                repo: dict(files) for repo, files in repo_file_line_approx.items()
            },
        }
        with dump_json.open("w") as f:
            json.dump(stats_obj, f, ensure_ascii=False, indent=2)
        console.print(f"[green]统计结果已写入[/green] [bold]{dump_json}[/bold]")


if __name__ == "__main__":
    app()

