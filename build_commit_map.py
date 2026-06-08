"""
从 HuggingFace 数据集构建 instance_id -> base_commit 的映射。

数据来源：
- SWE-bench Verified (princeton-nlp/SWE-bench_Verified): mini-* 轨迹
- SWE-rebench (nebius/SWE-rebench): openhands-* 轨迹
"""

import json
from pathlib import Path

import typer
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
app = typer.Typer()


def load_swebench_verified() -> dict[str, str]:
    """加载 SWE-bench Verified 数据集，返回 instance_id -> base_commit 映射"""
    console.print("[cyan]Loading SWE-bench Verified dataset...[/cyan]")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    mapping: dict[str, str] = {}
    for item in ds:
        instance_id = item["instance_id"]
        base_commit = item["base_commit"]
        mapping[instance_id] = base_commit
    console.print(f"[green]Loaded {len(mapping)} instances from SWE-bench Verified[/green]")
    return mapping


def load_swe_rebench() -> dict[str, str]:
    """加载 SWE-rebench 数据集，返回 instance_id -> base_commit 映射"""
    console.print("[cyan]Loading SWE-rebench dataset...[/cyan]")
    ds = load_dataset("nebius/SWE-rebench", split="test")
    mapping: dict[str, str] = {}
    for item in ds:
        instance_id = item["instance_id"]
        base_commit = item["base_commit"]
        mapping[instance_id] = base_commit
    console.print(f"[green]Loaded {len(mapping)} instances from SWE-rebench[/green]")
    return mapping


@app.command()
def build(
    output: Path = typer.Option(Path("commit_map.json"), "-o", "--output", help="输出 JSON 文件路径"),
    include_verified: bool = typer.Option(True, "--verified/--no-verified", help="包含 SWE-bench Verified"),
    include_rebench: bool = typer.Option(True, "--rebench/--no-rebench", help="包含 SWE-rebench"),
):
    """
    构建 instance_id -> base_commit 的映射并保存为 JSON 文件。
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        all_mappings: dict[str, str] = {}

        if include_verified:
            progress.add_task("Loading SWE-bench Verified...", total=None)
            verified_map = load_swebench_verified()
            all_mappings.update(verified_map)

        if include_rebench:
            progress.add_task("Loading SWE-rebench...", total=None)
            rebench_map = load_swe_rebench()
            all_mappings.update(rebench_map)

    # 保存到 JSON
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_mappings, f, indent=2)

    console.print(f"\n[bold green]Done![/bold green] Saved {len(all_mappings)} mappings to {output}")


@app.command()
def stats(
    commit_map: Path = typer.Argument(..., help="commit_map.json 文件路径"),
):
    """
    显示 commit_map 的统计信息。
    """
    with open(commit_map) as f:
        mapping = json.load(f)

    console.print(f"Total instances: {len(mapping)}")

    # 按 repo 统计
    repos: dict[str, int] = {}
    for instance_id in mapping:
        # instance_id 格式: owner__repo-PR-number
        parts = instance_id.rsplit("-", 1)
        repo_part = parts[0] if len(parts) > 1 else instance_id
        repo = repo_part.replace("__", "/")
        repos[repo] = repos.get(repo, 0) + 1

    console.print(f"Unique repos: {len(repos)}")


if __name__ == "__main__":
    app()
