"""封装 SWE-bench docker 评估流程 (使用已有镜像)。

用法:
    # 评估 SWE-bench Verified 实例 (使用预拉取镜像)
    sudo uv run python -m quality.run_swebench \
        --predictions quality/outputs/predictions.jsonl \
        --dataset princeton-nlp/SWE-bench_Verified \
        --run-id my_eval \
        --max-workers 32

    # 自动按数据集来源拆分并分别评估
    sudo uv run python -m quality.run_swebench \
        --predictions quality/outputs/predictions.jsonl \
        --auto-split \
        --run-id my_eval \
        --max-workers 32

核心逻辑:
    - 使用 namespace="swebench" 让 harness 查找预拉取的镜像
      (swebench/sweb.eval.x86_64.{org}_1776_{repo}-{issue}:latest)
    - 跳过所有 env/instance image 构建
    - 仅过滤出 harness 支持的 repo
    - 自动跳过已完成的实例 (断点续跑)
"""

from __future__ import annotations

import json
import platform
import resource
from pathlib import Path

import docker
import typer
from rich.console import Console
from swebench.harness.run_evaluation import (
    get_predictions_from_file,
    get_dataset_from_preds,
    run_instances,
    make_run_report,
)
from swebench.harness.docker_utils import list_images, clean_images
from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.test_spec.test_spec import MAP_REPO_VERSION_TO_SPECS
from datasets import load_dataset as hf_load_dataset

console = Console()

# 预拉取镜像的 namespace
NAMESPACE = "swebench"

# ---------------------------------------------------------------------------
# harness 支持性检查
# ---------------------------------------------------------------------------

_SUPPORTED_REPOS: set[str] = set(MAP_REPO_VERSION_TO_SPECS.keys())


def _instance_id_to_repo(instance_id: str) -> str:
    """org__repo-issue -> org/repo"""
    if "__" not in instance_id:
        return ""
    org, rest = instance_id.split("__", 1)
    repo_name = rest.rsplit("-", 1)[0] if "-" in rest else rest
    return f"{org}/{repo_name}"


def _filter_supported_predictions(preds_path: Path, output_path: Path) -> tuple[int, int]:
    """过滤出 harness 支持的 predictions，写入 output_path。返回 (kept, skipped)。"""
    kept = 0
    skipped = 0
    with open(preds_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            repo = _instance_id_to_repo(pred["instance_id"])
            if repo in _SUPPORTED_REPOS:
                f_out.write(json.dumps(pred, ensure_ascii=False) + "\n")
                kept += 1
            else:
                skipped += 1
    return kept, skipped


# ---------------------------------------------------------------------------
# 数据集分组
# ---------------------------------------------------------------------------

def _load_instance_id_sets() -> dict[str, set[str]]:
    """加载各数据集的 instance_id 集合。"""
    sets: dict[str, set[str]] = {}

    ds = hf_load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    sets["swe_bench_verified"] = set(ds["instance_id"])

    ds_test = hf_load_dataset("princeton-nlp/SWE-bench", split="test")
    sets["swe_bench_test"] = set(ds_test["instance_id"]) - sets["swe_bench_verified"]

    ds_train = hf_load_dataset("princeton-nlp/SWE-bench", split="train")
    sets["swe_bench_train"] = set(ds_train["instance_id"])

    try:
        ds_nebius = hf_load_dataset("nebius/SWE-rebench", split="test")
        sets["nebius"] = set(ds_nebius["instance_id"])
    except Exception:
        sets["nebius"] = set()

    return sets


def _split_predictions(predictions_path: Path) -> dict[str, list[dict]]:
    """按数据集来源拆分 predictions。"""
    id_sets = _load_instance_id_sets()

    preds: list[dict] = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    groups: dict[str, list[dict]] = {
        "swe_bench_verified": [],
        "swe_bench": [],
        "nebius": [],
        "unknown": [],
    }

    for p in preds:
        iid = p["instance_id"]
        if iid in id_sets["swe_bench_verified"]:
            groups["swe_bench_verified"].append(p)
        elif iid in id_sets["swe_bench_test"] or iid in id_sets["swe_bench_train"]:
            groups["swe_bench"].append(p)
        elif iid in id_sets["nebius"]:
            groups["nebius"].append(p)
        else:
            groups["unknown"].append(p)

    return groups


# ---------------------------------------------------------------------------
# 核心评估
# ---------------------------------------------------------------------------

def _materialize_eval_summary(report_file, report_dir: Path) -> Path | None:
    """Move/adapt swebench harness report into ``report_dir/eval_summary.json``.

    The upstream harness writes ``<model>__<split>.<run_id>.json`` into cwd.
    We materialize a canonical ``eval_summary.json`` next to the predictions so
    aggregators don't need to know the harness filename convention.
    """
    if report_file is None:
        return None
    src = Path(report_file)
    if not src.exists():
        # The harness sometimes returns a Path that's relative to cwd;
        # try resolving against cwd before giving up.
        candidate = Path.cwd() / src.name
        if candidate.exists():
            src = candidate
        else:
            return None
    try:
        harness = json.loads(src.read_text())
    except Exception:  # noqa: BLE001
        return None
    submitted = int(harness.get("submitted_instances", 0))
    resolved = int(harness.get("resolved_instances", 0))
    summary = {
        "run_id": harness.get("schema_version") and harness.get("run_id") or None,
        "total": submitted,
        "resolved": resolved,
        "resolve_rate": (resolved / submitted) if submitted else 0.0,
        "completed": int(harness.get("completed_instances", 0)),
        "unresolved": int(harness.get("unresolved_instances", 0)),
        "empty_patch": int(harness.get("empty_patch_instances", 0)),
        "errors": int(harness.get("error_instances", 0)),
        "harness_report_name": src.name,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    target_summary = report_dir / "eval_summary.json"
    target_summary.write_text(json.dumps(summary, indent=2))
    # Preserve the full harness report alongside (for ID-level details).
    target_full = report_dir / "harness_report.json"
    target_full.write_text(json.dumps(harness, indent=2))
    try:
        src.unlink()
    except OSError:
        pass
    return target_summary


def run_swebench_eval(
    predictions_path: Path,
    dataset_name: str,
    split: str = "test",
    run_id: str = "quality_eval",
    max_workers: int = 4,
    timeout: int = 1800,
    cache_level: str = "instance",
    open_file_limit: int = 4096,
    report_dir: Path = Path("quality/outputs/reports"),
) -> Path | None:
    """使用已有镜像运行 swebench 评估，返回 report 文件路径。"""
    report_dir.mkdir(parents=True, exist_ok=True)

    # 过滤出 harness 支持的 predictions
    filtered_path = report_dir / f"_filtered_{predictions_path.name}"
    kept, skipped = _filter_supported_predictions(predictions_path, filtered_path)
    console.log(f"Predictions: {kept} supported, {skipped} unsupported (skipped)")

    if kept == 0:
        console.print("[yellow]No supported instances to evaluate[/yellow]")
        return None

    console.log(f"Running swebench evaluation:")
    console.log(f"  dataset: {dataset_name}, split: {split}")
    console.log(f"  predictions: {filtered_path} ({kept} instances)")
    console.log(f"  run_id: {run_id}, workers: {max_workers}")
    console.log(f"  namespace: {NAMESPACE} (using pre-pulled images)")

    # 加载 predictions
    predictions = get_predictions_from_file(str(filtered_path), dataset_name, split)
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    if not predictions:
        console.print("[yellow]No predictions matched dataset[/yellow]")
        return None

    # 加载 dataset (自动跳过已完成的实例)
    dataset = get_dataset_from_preds(dataset_name, split, [], predictions, run_id, False)
    full_dataset = list(hf_load_dataset(dataset_name, split=split))

    if not dataset:
        console.print("[yellow]No instances to run (all completed?)[/yellow]")
        return None

    # 设置 open file limit (Linux only)
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    client = docker.from_env()
    existing_images = list_images(client)

    # 直接运行实例 — 使用 namespace="swebench" 匹配预拉取镜像
    # 不需要 build_env_images (namespace != None 时 harness 自动跳过)
    # build_container 中 is_remote_image=True → 直接 images.get() 而非 build
    run_instances(
        predictions, dataset, cache_level, False, False,
        max_workers, run_id, timeout,
        namespace=NAMESPACE,
        instance_image_tag="latest",
        env_image_tag="latest",
        rewrite_reports=False,
    )

    # 生成报告 (不清理镜像 — cache_level="instance" 保留所有)
    clean_images(client, existing_images, cache_level, False)
    report_file = make_run_report(
        predictions, full_dataset, run_id, client,
        namespace=NAMESPACE,
        instance_image_tag="latest",
        env_image_tag="latest",
    )

    # Move + adapt harness report into report_dir/eval_summary.json so
    # downstream aggregators can locate it consistently. The harness writes the
    # report into cwd by default; copy and normalize the schema here.
    canonical_path = _materialize_eval_summary(report_file, report_dir)
    if canonical_path is not None:
        console.print(f"[green]Evaluation complete. Report: {canonical_path}[/green]")
        return canonical_path

    console.print(f"[green]Evaluation complete. Report: {report_file}[/green]")
    return report_file


def run_auto_split(
    predictions_path: Path,
    run_id: str = "quality_eval",
    max_workers: int = 4,
    timeout: int = 1800,
    cache_level: str = "instance",
    report_dir: Path = Path("quality/outputs/reports"),
):
    """自动拆分 predictions 并分别评估 (仅评估 harness 支持的 instance)。"""
    console.log("Splitting predictions by dataset source...")
    groups = _split_predictions(predictions_path)

    for group_name, preds in groups.items():
        if not preds:
            continue
        console.print(f"\n[cyan]Group: {group_name} ({len(preds)} instances)[/cyan]")

        if group_name == "unknown":
            console.print("[yellow]Skipping unknown instances[/yellow]")
            continue

        # 写临时 predictions 文件
        tmp_path = report_dir / f"predictions_{group_name}.jsonl"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        if group_name == "swe_bench_verified":
            dataset_name = "princeton-nlp/SWE-bench_Verified"
            split = "test"
        elif group_name == "swe_bench":
            dataset_name = "princeton-nlp/SWE-bench"
            split = "test"
        elif group_name == "nebius":
            dataset_name = "nebius/SWE-rebench"
            split = "test"
        else:
            continue

        group_run_id = f"{run_id}_{group_name}"
        run_swebench_eval(
            tmp_path, dataset_name, split,
            run_id=group_run_id,
            max_workers=max_workers,
            timeout=timeout,
            cache_level=cache_level,
            report_dir=report_dir,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(
    predictions: Path = typer.Option(
        ..., "--predictions", "-p",
        help="predictions JSONL 路径",
    ),
    dataset: str = typer.Option(
        None, "--dataset", "-d",
        help="SWE-bench 数据集名称 (e.g. princeton-nlp/SWE-bench_Verified)",
    ),
    auto_split: bool = typer.Option(
        False, "--auto-split",
        help="自动按数据集来源拆分并分别评估 (跳过不支持的 repo)",
    ),
    split: str = typer.Option("test", "--split", "-s"),
    run_id: str = typer.Option("quality_eval", "--run-id", "-r"),
    max_workers: int = typer.Option(4, "--max-workers", "-j"),
    timeout: int = typer.Option(1800, "--timeout"),
    cache_level: str = typer.Option("instance", "--cache-level"),
    report_dir: Path = typer.Option(
        "quality/outputs/reports", "--report-dir",
    ),
):
    """运行 SWE-bench docker 评估 (使用已有镜像，跳过所有 build)。"""
    if auto_split:
        run_auto_split(
            predictions, run_id=run_id,
            max_workers=max_workers, timeout=timeout,
            cache_level=cache_level, report_dir=report_dir,
        )
    elif dataset:
        run_swebench_eval(
            predictions, dataset, split,
            run_id=run_id, max_workers=max_workers,
            timeout=timeout, cache_level=cache_level,
            report_dir=report_dir,
        )
    else:
        console.print("[red]请指定 --dataset 或使用 --auto-split[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
