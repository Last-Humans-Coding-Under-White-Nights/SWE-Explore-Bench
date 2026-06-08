"""SWE-bench-Pro evaluation runner (custom — `swebench` PyPI does not support Pro).

Per-instance flow inside a fresh container based on `meta.docker_image`:
    1. entrypoint=[""] + sleep keeps container alive while we exec into it.
    2. cd /app, run `before_repo_set_cmd` to reset to base_commit and apply test_patch.
    3. Apply `model_patch` from predictions via `git apply`.
    4. Run pytest on `fail_to_pass + pass_to_pass` with PYTEST_ADDOPTS already
       baked into the image. Parse stdout for PASS/FAIL counts.
    5. Resolved iff every fail_to_pass test now PASSES and every pass_to_pass
       test still PASSES.

Inputs: `--predictions <jsonl>` and `--bench bench.swepro.jsonl`.
Outputs: `--out-dir/<run_id>/eval_summary.json` + per-instance log files.

Usage:
    sg docker -c '.venv/bin/python -m quality.run_swebench_pro \
        --predictions results/.../predictions.jsonl \
        --bench bench.swepro.jsonl \
        --out-dir results/.../pro_eval \
        --run-id swepro_eval --max-workers 4'
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import shlex
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, NotFound


# Per-instance container timeout (seconds). Pro test suites can be slow.
DEFAULT_TIMEOUT_S = 1800

# A heredoc terminator unlikely to appear in a real patch.
PATCH_HEREDOC = "__SWE_PRO_PATCH_EOF__"


@dataclass
class _Result:
    instance_id: str
    resolved: bool
    error: str | None
    fail_to_pass_results: dict[str, str]
    pass_to_pass_results: dict[str, str]
    raw_log: str


def _load_predictions(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = obj.get("instance_id") or obj.get("KEY_INSTANCE_ID")
            if iid:
                out[iid] = obj
    return out


def _load_bench(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            out[obj["instance_id"]] = obj
    return out


def _coerce_test_list(value: Any) -> list[str]:
    """Pro bench rows store fail_to_pass/pass_to_pass as Python list-repr strings.
    Accept either a list or such a string; return list[str].
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            import ast
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [s]
        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed]
        return [str(parsed)]
    return [str(value)]


def _exec_or_raise(container, cmd: list[str], *, workdir: str | None = None) -> tuple[int, str]:
    rc, out = container.exec_run(cmd, workdir=workdir, demux=False)
    return rc, out.decode("utf-8", errors="replace") if isinstance(out, (bytes, bytearray)) else (out or "")


# -----------------------------------------------------------------------------
# pytest output parsing
# -----------------------------------------------------------------------------

# pytest -v line examples:
#   tests/foo.py::TestBar::test_baz PASSED                                     [ 12%]
#   tests/foo.py::TestBar::test_baz FAILED                                     [ 24%]
#   tests/foo.py::TestBar::test_baz SKIPPED (reason)                           [ 50%]
#   tests/foo.py::TestBar::test_baz ERROR                                      [ 75%]
#   tests/foo.py::TestBar::test_baz XFAIL                                      [ 99%]
_PYTEST_LINE_RE = re.compile(
    r"^(?P<name>[^\s]+)\s+(?P<status>PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS|RERUN)\b"
)


def _parse_pytest_log(log: str) -> dict[str, str]:
    """Return {test_id -> last seen status}.

    With --reruns=3, the same test_id can appear with RERUN then PASSED/FAILED.
    We keep the LAST non-RERUN status seen.
    """
    results: dict[str, str] = {}
    for line in log.splitlines():
        m = _PYTEST_LINE_RE.match(line)
        if not m:
            continue
        status = m.group("status")
        if status == "RERUN":
            continue
        results[m.group("name")] = status
    return results


def _eval_one(
    *,
    docker_client,
    instance_id: str,
    bench_row: dict,
    pred: dict,
    out_dir: Path,
    timeout_s: int,
) -> _Result:
    meta = bench_row.get("meta") or {}
    image = meta.get("docker_image")
    workdir = meta.get("workdir") or "/app"
    before_cmd = meta.get("before_repo_set_cmd") or ""
    fail_to_pass = _coerce_test_list(meta.get("fail_to_pass"))
    pass_to_pass = _coerce_test_list(meta.get("pass_to_pass"))

    if not image:
        return _Result(instance_id, False, "no docker_image in bench meta", {}, {}, "")

    model_patch = pred.get("model_patch") or pred.get("patch") or ""

    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(msg)

    container = None
    try:
        container = docker_client.containers.run(
            image,
            command=["sleep", str(timeout_s + 60)],
            entrypoint=[""],
            detach=True,
            remove=False,
            working_dir=workdir,
        )
        _log(f"[start] image={image} workdir={workdir} container={container.short_id}")

        # 1) reset to base_commit + apply test_patch
        if before_cmd:
            rc, out = _exec_or_raise(container, ["bash", "-lc", before_cmd], workdir=workdir)
            _log(f"[before_cmd rc={rc}]\n{out[-2000:]}")
            if rc != 0:
                return _Result(instance_id, False, "before_repo_set_cmd failed", {}, {}, "\n".join(log_lines))

        # 2) apply model_patch (if present)
        if model_patch.strip():
            # We write to /tmp via a small heredoc, then `git apply`.
            heredoc = (
                f"cat > /tmp/model.patch <<'{PATCH_HEREDOC}'\n"
                f"{model_patch}\n"
                f"{PATCH_HEREDOC}\n"
                "git apply --whitespace=nowarn -v /tmp/model.patch"
            )
            rc, out = _exec_or_raise(container, ["bash", "-lc", heredoc], workdir=workdir)
            _log(f"[git apply rc={rc}]\n{out[-2000:]}")
            if rc != 0:
                # Try with -p1 strip + --reject as fallback (3-way isn't always usable).
                heredoc2 = (
                    "git apply --whitespace=nowarn -p1 --reject -v /tmp/model.patch || true"
                )
                rc2, out2 = _exec_or_raise(container, ["bash", "-lc", heredoc2], workdir=workdir)
                _log(f"[git apply fallback rc={rc2}]\n{out2[-2000:]}")
                if rc2 != 0:
                    return _Result(
                        instance_id, False, "model_patch failed to apply",
                        {}, {}, "\n".join(log_lines),
                    )

        # 3) run pytest on the union of test ids. We pass them as positional args.
        all_tests = fail_to_pass + pass_to_pass
        if not all_tests:
            return _Result(instance_id, False, "no fail_to_pass/pass_to_pass tests", {}, {}, "\n".join(log_lines))

        # Quote each test id for safety (some have :: which is fine, but spaces)
        joined = " ".join(shlex.quote(t) for t in all_tests)
        # PYTEST_ADDOPTS already baked into image (--reruns=3 etc.).
        rc, out = _exec_or_raise(
            container,
            ["bash", "-lc", f"timeout {timeout_s} pytest {joined}"],
            workdir=workdir,
        )
        _log(f"[pytest rc={rc}]\n{out[-12000:]}")

        results = _parse_pytest_log(out)

        f2p_results = {t: results.get(t, "MISSING") for t in fail_to_pass}
        p2p_results = {t: results.get(t, "MISSING") for t in pass_to_pass}

        all_f2p_pass = all(s == "PASSED" for s in f2p_results.values()) if fail_to_pass else True
        all_p2p_pass = all(s == "PASSED" for s in p2p_results.values()) if pass_to_pass else True
        resolved = bool(all_f2p_pass and all_p2p_pass)

        return _Result(instance_id, resolved, None, f2p_results, p2p_results, "\n".join(log_lines))

    except Exception as e:
        return _Result(
            instance_id, False, f"exception: {e}\n{traceback.format_exc()}",
            {}, {}, "\n".join(log_lines),
        )
    finally:
        if container is not None:
            try:
                container.kill()
            except Exception:
                pass
            try:
                container.remove(force=True)
            except Exception:
                pass


def _save_result(out_dir: Path, r: _Result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{r.instance_id}.log"
    log_path.write_text(r.raw_log)
    json_path = out_dir / f"{r.instance_id}.json"
    json_path.write_text(json.dumps(
        {
            "instance_id": r.instance_id,
            "resolved": r.resolved,
            "error": r.error,
            "fail_to_pass_results": r.fail_to_pass_results,
            "pass_to_pass_results": r.pass_to_pass_results,
        },
        indent=2,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="swepro_eval")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N instances (0=all).")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated instance_id whitelist.")
    parser.add_argument("--skip-resolved", action="store_true",
                        help="Skip instances that already have a result file marked resolved.")
    args = parser.parse_args()

    bench = _load_bench(args.bench)
    preds = _load_predictions(args.predictions)

    only: set[str] | None = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}

    instances = []
    for iid, pred in preds.items():
        if iid not in bench:
            print(f"[skip] {iid}: not in bench", file=sys.stderr)
            continue
        if only and iid not in only:
            continue
        instances.append((iid, pred))

    if args.limit:
        instances = instances[: args.limit]

    out_root = args.out_dir / args.run_id
    per_inst_dir = out_root / "instances"
    out_root.mkdir(parents=True, exist_ok=True)
    per_inst_dir.mkdir(parents=True, exist_ok=True)

    # Skip already-evaluated.
    pending = []
    for iid, pred in instances:
        result_json = per_inst_dir / f"{iid}.json"
        if result_json.exists():
            try:
                prior = json.loads(result_json.read_text())
                if args.skip_resolved or "resolved" in prior:
                    print(f"[cache] {iid}: already evaluated (resolved={prior.get('resolved')}), skipping")
                    continue
            except Exception:
                pass
        pending.append((iid, pred))

    print(f"Total: {len(instances)}, pending: {len(pending)}")

    docker_client = docker.from_env()
    started = time.time()

    results: list[_Result] = []
    if pending:
        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futs = {
                pool.submit(
                    _eval_one,
                    docker_client=docker_client,
                    instance_id=iid,
                    bench_row=bench[iid],
                    pred=pred,
                    out_dir=per_inst_dir,
                    timeout_s=args.timeout,
                ): iid
                for iid, pred in pending
            }
            for fut in cf.as_completed(futs):
                iid = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = _Result(iid, False, f"runner exception: {e}", {}, {}, "")
                _save_result(per_inst_dir, r)
                results.append(r)
                marker = "✓" if r.resolved else "✗"
                err = f" ({r.error})" if r.error else ""
                print(f"  [{marker}] {iid}{err}")

    # Aggregate (re-read all per-instance files in case of resumed runs).
    all_results: list[dict] = []
    for iid, _pred in instances:
        rj = per_inst_dir / f"{iid}.json"
        if rj.exists():
            all_results.append(json.loads(rj.read_text()))

    resolved_ids = sorted(r["instance_id"] for r in all_results if r["resolved"])
    unresolved_ids = sorted(r["instance_id"] for r in all_results if not r["resolved"] and not r.get("error"))
    error_ids = sorted(r["instance_id"] for r in all_results if r.get("error"))
    total = len(all_results)
    summary = {
        "run_id": args.run_id,
        "total": total,
        "resolved": len(resolved_ids),
        "resolve_rate": len(resolved_ids) / total if total else 0.0,
        "resolved_ids": resolved_ids,
        "unresolved_ids": unresolved_ids,
        "error_ids": error_ids,
        "predictions_path": str(args.predictions),
        "bench_path": str(args.bench),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out_root / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(
        {k: summary[k] for k in ("total", "resolved", "resolve_rate", "elapsed_seconds")},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
