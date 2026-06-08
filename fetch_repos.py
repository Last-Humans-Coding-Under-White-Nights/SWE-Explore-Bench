"""
通过 GitHub Archive API 下载仓库到指定 commit。

用法：
    # 先构建 commit_map
    uv run python build_commit_map.py build -o commit_map.json

    # 下载仓库
    uv run python fetch_repos.py clone --trajs-dir unify_trajs --commit-map commit_map.json --repos-dir repos
"""

import json
import re
import shutil
import tarfile
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()
app = typer.Typer()


def extract_repos_from_trajs(trajs_dir: Path) -> dict[str, set[str]]:
    """
    从轨迹目录提取所有 instance_id，并按 repo 分组。
    返回 {repo: {instance_id1, instance_id2, ...}}
    """
    repo_instances: dict[str, set[str]] = defaultdict(set)

    for traj_file in trajs_dir.rglob("*.json"):
        try:
            with open(traj_file) as f:
                data = json.load(f)
            # 兼容两种格式：统一轨迹格式 (info.instance_id/repo) 和 mini-swe-agent 格式 (顶层 instance_id)
            instance_id = data.get("info", {}).get("instance_id") or data.get(
                "instance_id"
            )
            repo = data.get("info", {}).get("repo")
            # 若无 repo 字段，从 instance_id 解析：org__repo-issue -> org/repo
            if not repo and instance_id and "__" in instance_id:
                org, rest = instance_id.split("__", 1)
                repo_name = rest.rsplit("-", 1)[0] if "-" in rest else rest
                repo = f"{org}/{repo_name}"
            if instance_id and repo:
                repo_instances[repo].add(instance_id)
        except Exception:
            traceback.print_exc()
            console.print(f"[yellow]Warning: Failed to parse {traj_file}[/yellow]")

    return repo_instances


def download_archive(
    repo: str,
    repos_dir: Path,
    commit: str | None = None,
    timeout: float = 300.0,
    mirror: str | None = None,
    target_name: str | None = None,
) -> bool:
    """
    通过 GitHub Archive API 下载仓库压缩包并解压。

    Args:
        repo: 仓库路径 (owner/name)
        repos_dir: 仓库保存目录
        commit: 要下载的 commit hash，None 则下载默认分支
        target_name: 解压后的目录名，默认为 repo 名（即 repo.split("/")[-1]）

    Returns:
        是否成功
    """
    name = target_name or repo.split("/")[-1]
    target_dir = repos_dir / name

    if target_dir.exists():
        return True

    # GitHub archive URL: https://github.com/{owner}/{repo}/archive/{ref}.tar.gz
    if not commit:
        console.print(f"[red]FATAL: No commit specified for {repo} (target: {target_name}), refusing to download HEAD[/red]")
        return False
    ref = commit
    base = mirror.rstrip("/") if mirror else "https://github.com"
    url = f"{base}/{repo}/archive/{ref}.tar.gz"

    tmp_file = repos_dir / f".tmp_{name}.tar.gz"
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(tmp_file, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

        # 解压 tarball
        with tarfile.open(tmp_file, mode="r:gz") as tar:
            # tarball 内部有一层目录，名为 {repo_name}-{commit}
            members = tar.getmembers()
            if not members:
                console.print(f"[red]Empty archive for {repo}[/red]")
                return False

            # 获取顶层目录名
            top_dir = members[0].name.split("/")[0]

            # 解压到临时位置
            tar.extractall(repos_dir)

            # 重命名为 target_name（instance_id 或 repo_name）
            extracted_dir = repos_dir / top_dir
            if extracted_dir.exists():
                shutil.move(str(extracted_dir), str(target_dir))

        return True

    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP {e.response.status_code} for {repo}: {url}[/red]")
        return False
    except Exception:
        traceback.print_exc()
        console.print(f"[red]Error downloading {repo}[/red]")
        return False
    finally:
        if tmp_file.exists():
            tmp_file.unlink(missing_ok=True)


@app.command()
def clone(
    trajs_dir: Path = typer.Option(..., "--trajs-dir", "-t", help="轨迹目录"),
    commit_map_file: Path = typer.Option(
        ..., "--commit-map", "-c", help="commit_map.json 文件路径"
    ),
    repos_dir: Path = typer.Option(
        Path("repos"), "--repos-dir", "-r", help="仓库保存目录"
    ),
    timeout: float = typer.Option(300.0, "--timeout", help="下载超时时间（秒）"),
    workers: int = typer.Option(8, "--workers", "-w", help="并发下载数"),
    mirror: str | None = typer.Option(
        None, "--mirror", "-m", help="GitHub 镜像地址，如 https://ghfast.top"
    ),
):
    """
    从轨迹目录提取仓库列表，通过 GitHub Archive API 下载到指定 commit。
    """
    # 加载 commit map
    console.print(f"[cyan]Loading commit map from {commit_map_file}...[/cyan]")
    with open(commit_map_file) as f:
        commit_map: dict[str, str] = json.load(f)
    console.print(f"[green]Loaded {len(commit_map)} commit mappings[/green]")

    # 从轨迹提取 repo -> instances 映射
    console.print(f"[cyan]Scanning trajectories in {trajs_dir}...[/cyan]")
    repo_instances = extract_repos_from_trajs(trajs_dir)
    console.print(f"[green]Found {len(repo_instances)} repos from trajectories[/green]")

    # 准备下载任务：按 instance_id 逐个下载，每个 instance 对应正确的 commit
    repos_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str, str | None]] = []  # (repo, instance_id, commit)

    no_commit_ids: list[str] = []
    for repo, instances in repo_instances.items():
        for instance_id in instances:
            commit = commit_map.get(instance_id)
            if not commit:
                console.print(
                    f"[yellow]Warning: No commit found for {instance_id} (repo: {repo})[/yellow]"
                )
                continue
            tasks.append((repo, instance_id, commit))

    if no_commit_ids:
        console.print(
            f"[yellow]Warning: {len(no_commit_ids)} instances skipped (no commit in commit_map). "
            f"First 5: {no_commit_ids[:5]}[/yellow]"
        )

    # 过滤已存在的（按 instance_id 命名的目录）
    pending_tasks: list[tuple[str, str, str]] = []
    skip_count = 0
    for repo, instance_id, commit in tasks:
        if (repos_dir / instance_id).exists():
            skip_count += 1
        else:
            pending_tasks.append((repo, instance_id, commit))

    # 统计
    with_commit = sum(1 for _, _, c in pending_tasks if c is not None)
    without_commit = len(pending_tasks) - with_commit
    console.print(
        f"[cyan]To download: {len(pending_tasks)} (with commit: {with_commit}, without: {without_commit}), skipped: {skip_count}[/cyan]"
    )

    if not pending_tasks:
        console.print("[green]All repos already downloaded.[/green]")
        return

    # 并发下载
    success_count = 0
    fail_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            f"Downloading with {workers} workers...", total=len(pending_tasks)
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_archive,
                    repo,
                    repos_dir,
                    commit,
                    timeout,
                    mirror,
                    instance_id,
                ): instance_id
                for repo, instance_id, commit in pending_tasks
            }

            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception:
                    traceback.print_exc()
                    fail_count += 1
                progress.advance(task_id)

    console.print(
        f"\n[bold green]Done![/bold green] Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}"
    )


@app.command()
def clone_bench(
    bench_file: Path = typer.Option(..., "--bench", "-b", help="bench JSONL 文件路径"),
    commit_map_file: Path = typer.Option(
        ..., "--commit-map", "-c", help="commit_map.json 文件路径"
    ),
    repos_dir: Path = typer.Option(
        Path("repos"), "--repos-dir", "-r", help="仓库保存目录"
    ),
    timeout: float = typer.Option(300.0, "--timeout", help="下载超时时间（秒）"),
    workers: int = typer.Option(8, "--workers", "-w", help="并发下载数"),
    mirror: str | None = typer.Option(
        None, "--mirror", "-m", help="GitHub 镜像地址，如 https://ghfast.top"
    ),
):
    """从 bench JSONL 文件中读取 instance_id，通过 GitHub Archive API 下载到指定 commit。"""
    console.print(f"[cyan]Loading commit map from {commit_map_file}...[/cyan]")
    with open(commit_map_file) as f:
        commit_map: dict[str, str] = json.load(f)
    console.print(f"[green]Loaded {len(commit_map)} commit mappings[/green]")

    # 从 bench JSONL 读取所有 instance_id
    instance_ids: list[str] = []
    with open(bench_file) as f:
        for line in f:
            line = line.strip()
            if line:
                instance_ids.append(json.loads(line)["instance_id"])
    console.print(f"[green]Found {len(instance_ids)} instances in bench[/green]")

    repos_dir.mkdir(parents=True, exist_ok=True)

    # 构建下载任务：跳过已存在的
    pending_tasks: list[tuple[str, str, str | None]] = []
    skip_count = 0
    no_commit = 0
    for instance_id in instance_ids:
        if (repos_dir / instance_id).exists():
            skip_count += 1
            continue
        commit = commit_map.get(instance_id)
        if not commit:
            base_id = re.sub(r"_run\d+$", "", instance_id)
            commit = commit_map.get(base_id)
        if not commit:
            no_commit += 1
            continue  # commit 必须对齐，跳过没有 commit 的 instance
        # 从 instance_id 解析 repo: org__repo-issue -> org/repo
        if "__" not in instance_id:
            continue
        org, rest = instance_id.split("__", 1)
        repo_name = rest.rsplit("-", 1)[0] if "-" in rest else rest
        repo = f"{org}/{repo_name}"
        pending_tasks.append((repo, instance_id, commit))

    with_commit = sum(1 for _, _, c in pending_tasks if c is not None)
    console.print(
        f"[cyan]To download: {len(pending_tasks)} (with commit: {with_commit}), skipped: {skip_count}, no commit: {no_commit}[/cyan]"
    )

    if not pending_tasks:
        console.print("[green]All repos already downloaded.[/green]")
        return

    success_count = 0
    fail_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            f"Downloading with {workers} workers...", total=len(pending_tasks)
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_archive,
                    repo,
                    repos_dir,
                    commit,
                    timeout,
                    mirror,
                    instance_id,
                ): instance_id
                for repo, instance_id, commit in pending_tasks
            }
            for future in as_completed(futures):
                instance_id = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception:
                    traceback.print_exc()
                    fail_count += 1
                progress.advance(task_id)

    console.print(
        f"\n[bold green]Done![/bold green] Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}"
    )


@app.command()
def list_repos(
    trajs_dir: Path = typer.Option(..., "--trajs-dir", "-t", help="轨迹目录"),
    commit_map_file: Path = typer.Option(
        None, "--commit-map", "-c", help="commit_map.json 文件路径（可选）"
    ),
    output: Path = typer.Option(
        None, "--output", "-o", help="输出文件路径（不指定则打印到终端）"
    ),
):
    """
    列出轨迹目录中的所有仓库及其 commit 信息。
    """
    repo_instances = extract_repos_from_trajs(trajs_dir)

    commit_map: dict[str, str] = {}
    if commit_map_file and commit_map_file.exists():
        with open(commit_map_file) as f:
            commit_map = json.load(f)

    lines: list[str] = []
    for repo in sorted(repo_instances.keys()):
        instances = repo_instances[repo]
        for instance_id in sorted(instances):
            commit = commit_map.get(instance_id)
            if commit:
                lines.append(f"{repo}\t{instance_id}\t{commit}")
            else:
                if not commit:
                    console.print(
                        f"[yellow]Warning: No commit found for {instance_id} (repo: {repo})[/yellow]"
                    )
                    continue
                # lines.append(f"{repo}\t{instance_id}")

    if output:
        with open(output, "w") as f:
            f.write("\n".join(lines))
        console.print(f"[green]Written {len(lines)} repos to {output}[/green]")
    else:
        for line in lines:
            console.print(line)


if __name__ == "__main__":
    app()
