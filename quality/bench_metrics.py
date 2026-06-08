from __future__ import annotations

import math
from pathlib import Path
from typing import Any

END_OF_FILE = -1

METRIC_FIELDS = [
    "precision",
    "recall",
    "f1_score",
    "hit_file_rate",
    "noise_file_rate",
    "hit_region_rate",
    "noise_region_rate",
    "weighted_core_coverage",
    "context_efficiency",
    "ndcg_at_100",
    "ndcg_at_300",
    "ndcg_at_500",
    "recall_at_100",
    "recall_at_300",
    "recall_at_500",
    "first_useful_hit",
]

CONTEXT_STAT_FIELDS = [
    "kept_regions",
    "unique_files_kept",
    "total_lines_kept",
    "whole_file_region_share",
    "avg_lines_per_region",
    "main_file_share",
    "modified_file_share",
    "core_line_coverage_after_slice",
]

Region = dict[str, Any] | tuple[str, int, int]


def _normalize_region(region: Region) -> tuple[str, int, int]:
    if isinstance(region, dict):
        return str(region["path"]), int(region["start"]), int(region["end"])
    return str(region[0]), int(region[1]), int(region[2])


def _resolve_interval(
    path: str,
    start: int,
    end: int,
    path_to_lines: dict[str, int],
) -> tuple[int, int] | None:
    total = path_to_lines.get(path)
    need_total = end == END_OF_FILE or start < 0
    if need_total and (total is None or total < 1):
        return None
    if need_total and total is not None:
        resolved_end = total if end == END_OF_FILE else end
        resolved_start = (total + start + 1) if start < 0 else start
        resolved_start = max(1, min(resolved_start, total))
        resolved_end = max(1, min(resolved_end, total))
        return (resolved_start, resolved_end)
    if end == END_OF_FILE or start < 0:
        return None
    start = max(1, start)
    if end < 1:
        return None
    return (start, end)


def _regions_to_lines(regions: list[Region], path_to_lines: dict[str, int]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for region in regions:
        path, start, end = _normalize_region(region)
        resolved = _resolve_interval(path, start, end, path_to_lines)
        if resolved is None:
            continue
        s, e = resolved
        for line in range(s, e + 1):
            out.add((path, line))
    return out


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _region_overlap(region_a: Region, region_b: Region, path_to_lines: dict[str, int]) -> bool:
    path_a, start_a, end_a = _normalize_region(region_a)
    path_b, start_b, end_b = _normalize_region(region_b)
    if path_a != path_b:
        return False
    resolved_a = _resolve_interval(path_a, start_a, end_a, path_to_lines)
    resolved_b = _resolve_interval(path_b, start_b, end_b, path_to_lines)
    if resolved_a is None or resolved_b is None:
        return False
    return _intervals_overlap(resolved_a, resolved_b)


def _optional_files(bench_gt: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for files in (bench_gt.get("read_optional_files_map") or {}).values():
        out.update(str(path) for path in files or [])
    return out


def _optional_regions(bench_gt: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for regions in (bench_gt.get("read_optional_regions_map") or {}).values():
        for region in regions or []:
            out.append(region)
    return out


def collect_path_line_counts(repo_dir: Path, regions: list[Region], bench_gt: dict[str, Any] | None = None) -> dict[str, int]:
    paths: set[str] = set()
    for region in regions:
        path, _, _ = _normalize_region(region)
        paths.add(path)
    for region in (bench_gt or {}).get("read_core_regions") or []:
        paths.add(str(region.get("path")))
    for region in _optional_regions(bench_gt or {}):
        paths.add(str(region.get("path")))

    counts: dict[str, int] = {}
    for path in sorted(paths):
        file_path = repo_dir / path
        if not file_path.is_file():
            continue
        try:
            counts[path] = len(file_path.read_text(errors="replace").splitlines())
        except OSError:
            continue
    return counts


def compute_region_metrics(
    pred_regions: list[Region],
    formal_ground_truth: dict[str, Any],
    repo_dir: Path | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    if repo_dir is None or not repo_dir.is_dir():
        return {}, {"missing_repo_dir": str(repo_dir) if repo_dir else None}

    path_to_lines = collect_path_line_counts(repo_dir, pred_regions, formal_ground_truth)
    core_regions = formal_ground_truth.get("read_core_regions") or []
    main_files = set(formal_ground_truth.get("main_files") or [])
    pred_lines = _regions_to_lines(pred_regions, path_to_lines)
    core_lines = _regions_to_lines(core_regions, path_to_lines)
    optional_regions = _optional_regions(formal_ground_truth)
    optional_lines = _regions_to_lines(optional_regions, path_to_lines)
    visited_files = {path for path, _start, _end in (_normalize_region(r) for r in pred_regions)}
    core_files = set(formal_ground_truth.get("read_core_files") or [])
    noise_files = visited_files - core_files - _optional_files(formal_ground_truth)

    precision = len(pred_lines & core_lines) / len(pred_lines) if pred_lines else 0.0
    recall = len(pred_lines & core_lines) / len(core_lines) if core_lines else 0.0
    f1_score = 0.0 if precision + recall == 0 else (2.0 * precision * recall / (precision + recall))
    hit_file_rate = len(visited_files & core_files) / len(core_files) if core_files else 0.0
    noise_file_rate = len(noise_files) / len(visited_files) if visited_files else 0.0

    hit_region_count = 0
    for core_region in core_regions:
        if any(_region_overlap(core_region, pred_region, path_to_lines) for pred_region in pred_regions):
            hit_region_count += 1
    hit_region_rate = hit_region_count / len(core_regions) if core_regions else 0.0

    noise_region_count = 0
    for pred_region in pred_regions:
        overlap_core = any(_region_overlap(pred_region, core_region, path_to_lines) for core_region in core_regions)
        overlap_optional = any(_region_overlap(pred_region, optional_region, path_to_lines) for optional_region in optional_regions)
        if not overlap_core and not overlap_optional:
            noise_region_count += 1
    noise_region_rate = noise_region_count / len(pred_regions) if pred_regions else 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for core_region in core_regions:
        path, start, end = _normalize_region(core_region)
        resolved = _resolve_interval(path, start, end, path_to_lines)
        if resolved is None:
            continue
        s, e = resolved
        gt_lines = {(path, line) for line in range(s, e + 1)}
        if not gt_lines:
            continue
        coverage = len(pred_lines & gt_lines) / len(gt_lines)
        weight = 3.0 if path in main_files else 2.0
        weighted_sum += weight * coverage
        weight_total += weight
    weighted_core_coverage = weighted_sum / weight_total if weight_total else 0.0

    useful_gold_lines = core_lines | optional_lines
    context_efficiency = len(pred_lines & useful_gold_lines) / len(pred_lines) if pred_lines else 0.0

    def dcg_with_budget(gains: list[float], line_counts: list[int], budget: int) -> float:
        dcg = 0.0
        cumulative = 0
        for index, (gain, line_count) in enumerate(zip(gains, line_counts)):
            cumulative += line_count
            if cumulative > budget and index > 0:
                break
            dcg += gain / math.log2(index + 2)
        return dcg

    core_line_weight = {line: (1.5 if line[0] in main_files else 1.0) for line in core_lines}
    region_gains: list[float] = []
    region_line_counts: list[int] = []
    for pred_region in pred_regions:
        path, start, end = _normalize_region(pred_region)
        resolved = _resolve_interval(path, start, end, path_to_lines)
        if resolved is None:
            region_gains.append(0.0)
            region_line_counts.append(0)
            continue
        s, e = resolved
        line_count = e - s + 1
        gain = sum(core_line_weight.get((path, line), 0.0) for line in range(s, e + 1))
        region_gains.append(gain)
        region_line_counts.append(line_count)

    def ndcg_at_budget(budget: int) -> float:
        if not pred_regions or not core_regions:
            return 0.0
        dcg = dcg_with_budget(region_gains, region_line_counts, budget)
        ideal = sorted(
            zip(region_gains, region_line_counts),
            key=lambda item: item[0] / max(item[1], 1),
            reverse=True,
        )
        ideal_gains = [item[0] for item in ideal]
        ideal_counts = [item[1] for item in ideal]
        idcg = dcg_with_budget(ideal_gains, ideal_counts, budget)
        return min(dcg / idcg, 1.0) if idcg > 0 else 0.0

    def recall_at_budget(budget: int) -> float:
        if not core_lines:
            return 0.0
        covered: set[tuple[str, int]] = set()
        cumulative = 0
        for pred_region in pred_regions:
            path, start, end = _normalize_region(pred_region)
            resolved = _resolve_interval(path, start, end, path_to_lines)
            if resolved is None:
                continue
            s, e = resolved
            for line in range(s, e + 1):
                cumulative += 1
                key = (path, line)
                if key in core_lines:
                    covered.add(key)
                if cumulative >= budget:
                    break
            if cumulative >= budget:
                break
        return len(covered) / len(core_lines)

    first_useful_hit = 0.0
    if pred_regions and core_lines:
        for index, pred_region in enumerate(pred_regions):
            path, start, end = _normalize_region(pred_region)
            resolved = _resolve_interval(path, start, end, path_to_lines)
            if resolved is None:
                continue
            s, e = resolved
            if any((path, line) in core_lines for line in range(s, e + 1)):
                first_useful_hit = 1.0 - (index / len(pred_regions))
                break

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "hit_file_rate": hit_file_rate,
        "noise_file_rate": noise_file_rate,
        "hit_region_rate": hit_region_rate,
        "noise_region_rate": noise_region_rate,
        "weighted_core_coverage": weighted_core_coverage,
        "context_efficiency": context_efficiency,
        "ndcg_at_100": ndcg_at_budget(100),
        "ndcg_at_300": ndcg_at_budget(300),
        "ndcg_at_500": ndcg_at_budget(500),
        "recall_at_100": recall_at_budget(100),
        "recall_at_300": recall_at_budget(300),
        "recall_at_500": recall_at_budget(500),
        "first_useful_hit": first_useful_hit,
    }
    return metrics, {}


def compute_context_mass_stats(
    pred_regions: list[Region],
    pseudo_ground_truth: dict[str, Any],
    formal_ground_truth: dict[str, Any],
    repo_dir: Path | None,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    if repo_dir is None or not repo_dir.is_dir():
        return {}, {"missing_repo_dir": str(repo_dir) if repo_dir else None}

    path_to_lines = collect_path_line_counts(repo_dir, pred_regions, formal_ground_truth)
    total_lines_kept = 0
    resolved_region_count = 0
    for region in pred_regions:
        path, start, end = _normalize_region(region)
        resolved = _resolve_interval(path, start, end, path_to_lines)
        if resolved is None:
            continue
        resolved_region_count += 1
        total_lines_kept += resolved[1] - resolved[0] + 1

    unique_files = {path for path, _start, _end in (_normalize_region(r) for r in pred_regions)}
    unique_files_kept = len(unique_files)
    main_files = set(pseudo_ground_truth.get("main_files") or [])
    modified_files = set(
        pseudo_ground_truth.get("modified_core_files")
        or formal_ground_truth.get("modified_core_files")
        or []
    )
    whole_file_count = sum(1 for region in pred_regions if _normalize_region(region)[2] == END_OF_FILE)
    avg_lines_per_region = total_lines_kept / resolved_region_count if resolved_region_count else 0.0
    main_file_share = len(unique_files & main_files) / unique_files_kept if unique_files_kept else 0.0
    modified_file_share = len(unique_files & modified_files) / unique_files_kept if unique_files_kept else 0.0
    core_lines = _regions_to_lines(formal_ground_truth.get("read_core_regions") or [], path_to_lines)
    pred_lines = _regions_to_lines(pred_regions, path_to_lines)
    core_line_coverage_after_slice = len(pred_lines & core_lines) / len(core_lines) if core_lines else 0.0

    stats: dict[str, float | int] = {
        "kept_regions": len(pred_regions),
        "unique_files_kept": unique_files_kept,
        "total_lines_kept": total_lines_kept,
        "whole_file_region_share": (whole_file_count / len(pred_regions)) if pred_regions else 0.0,
        "avg_lines_per_region": avg_lines_per_region,
        "main_file_share": main_file_share,
        "modified_file_share": modified_file_share,
        "core_line_coverage_after_slice": core_line_coverage_after_slice,
    }
    return stats, {}
