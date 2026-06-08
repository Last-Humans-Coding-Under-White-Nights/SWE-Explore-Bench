"""使用 mini-swe-agent (Docker 模式) 生成 patch，将 core context 作为初始上下文。

环境采用 sanitize 模式：清理 Docker 容器只保留 context 相关代码，
非 context 文件被删除，非 context 行被替换为空行，模型可自由使用任何 shell 命令。

每个 instance 在独立的 SWE-bench Docker 容器中运行，支持并发。

用法:
    uv run python -m quality.run_mini_swe_agent \
        --bench bench.jsonl \
        --mode core_only \
        --dataset swe_bench_verified \
        --model gpt-4o-mini \
        --api-key sk-xxxx \
        --nums 10 \
        --workers 8

选项:
    --bench, -b          bench.jsonl 路径 (默认 bench.jsonl)
    --output, -o         输出路径 (默认 quality/outputs/predictions_{mode}_mini_agent.jsonl)
    --mode, -m           实验模式: core_only / core_optional / no_context
    --dataset, -d        数据集过滤: all / swe_bench_verified / swe_rebench / swe_bench
    --model              LLM 模型名称
    --api-key, -k        API key
    --base-url           API base URL
    --nums, -n           只跑前 N 个 instance，不指定则跑全部
    --workers, -w        并发 worker 数量 (默认 8)
    --log-dir            日志目录 (默认 logs/mini_swe_agent；同时保存 jsonl 和 .traj.json)
    --no-log             禁用日志记录
    --max-context-lines  最大 context 行数，超过则截断
    --max-steps          最大 LLM 调用步数，超过则自动终止
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

# 添加 mini-swe-agent 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "mini-swe-agent" / "src"))

from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.config import get_config_from_spec
from minisweagent.exceptions import FormatError
from minisweagent.utils.serialize import recursive_merge

from minisweagent.environments.docker import DockerEnvironment

from quality.gen_patches import (
    DEFAULT_BASE_URL,
    DEFAULT_API_KEY,
    DATASET_CHOICES,
    _extract_regions_metadata,
    _extract_diff,
    _fix_patch_format,
    _format_context_block,
    _load_dataset_instance_ids,
    _load_problem_statements,
)
from quality.gen_patches_agent import _truncate_context

console = Console()


def _build_debug_trajectory(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造简化 trajectory，用于 jsonl 调试日志。"""
    trajectory = []
    for msg in messages:
        content = msg.get("content") or ""
        entry = {"role": msg.get("role", ""), "content": content[:2000]}
        if msg.get("extra", {}).get("actions"):
            entry["actions"] = [a.get("command", "")[:500] for a in msg["extra"]["actions"]]
        trajectory.append(entry)
    return trajectory


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """原子写入 JSON 文件，避免实时日志留下半截内容。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp_path.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """追加一条 JSONL 记录并立即 flush。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _serialize_agent_state(agent: DefaultAgent, instance_id: str) -> dict[str, Any]:
    """序列化当前 agent 状态，供实时轨迹分析。"""
    last_message = agent.messages[-1] if agent.messages else {}
    last_extra = last_message.get("extra", {}) if isinstance(last_message, dict) else {}
    return agent.serialize(
        {
            "info": {
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "instance_id": instance_id,
        }
    )


class RealtimeLoggingDefaultAgent(DefaultAgent):
    """在每次 LLM 返回后实时落盘 trajectory，便于中断后分析。"""

    def __init__(
        self,
        model: LitellmModel,
        env: DockerEnvironment,
        *,
        instance_id: str,
        live_traj_path: Path | None = None,
        live_step_log_path: Path | None = None,
        **kwargs,
    ):
        super().__init__(model, env, **kwargs)
        self.instance_id = instance_id
        self.live_traj_path = live_traj_path
        self.live_step_log_path = live_step_log_path

    def _persist_live_state(self, *, event: str, message: dict[str, Any] | None = None) -> None:
        if self.live_traj_path is not None:
            _write_json_atomic(
                self.live_traj_path,
                _serialize_agent_state(self, self.instance_id),
            )

        if self.live_step_log_path is None or message is None:
            return

        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        actions = []
        for action in message.get("extra", {}).get("actions", []) or []:
            command = action.get("command", "")
            if isinstance(command, str):
                actions.append(command[:500])

        record = {
            "instance_id": self.instance_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "n_calls": self.n_calls,
            "cost": self.cost,
            "role": message.get("role", ""),
            "content": content[:2000],
            "actions": actions,
        }
        exit_status = message.get("extra", {}).get("exit_status")
        if exit_status:
            record["exit_status"] = exit_status
        _append_jsonl(self.live_step_log_path, record)

    def query(self) -> dict:
        message = super().query()
        self._persist_live_state(event="llm_response", message=message)
        return message

    def add_messages(self, *messages: dict) -> list[dict]:
        added = super().add_messages(*messages)
        if any(msg.get("role") == "exit" for msg in added):
            self._persist_live_state(event="exit", message=added[-1] if added else None)
        return added


class RealtimeLoggingLitellmModel(LitellmModel):
    """在 tool-call 解析前保存模型原始返回，便于分析失败样本。"""

    def __init__(
        self,
        *,
        instance_id: str,
        raw_response_log_path: Path | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.instance_id = instance_id
        self.raw_response_log_path = raw_response_log_path
        self._response_idx = 0

    def _log_raw_response(
        self,
        response: Any,
        *,
        event: str,
        error: str | None = None,
    ) -> None:
        if self.raw_response_log_path is None:
            return

        self._response_idx += 1
        record: dict[str, Any] = {
            "instance_id": self.instance_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "response_index": self._response_idx,
            "event": event,
        }

        try:
            record["raw_response"] = response.model_dump()
        except Exception:
            record["raw_response"] = str(response)

        try:
            message = response.choices[0].message
            content = getattr(message, "content", "") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            record["content"] = content[:4000]
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                serialized_tool_calls = []
                for tc in tool_calls:
                    if hasattr(tc, "model_dump"):
                        serialized_tool_calls.append(tc.model_dump())
                    else:
                        serialized_tool_calls.append(str(tc))
                record["tool_calls"] = serialized_tool_calls
        except Exception:
            pass

        if error is not None:
            record["error"] = error

        _append_jsonl(self.raw_response_log_path, record)

    def _parse_actions(self, response) -> list[dict]:
        self._log_raw_response(response, event="llm_raw_response")
        try:
            return super()._parse_actions(response)
        except FormatError as e:
            self._log_raw_response(response, event="llm_parse_error", error=str(e))
            raise


def _build_stats_trajectory(
    agent: DefaultAgent,
    instance_id: str,
    exit_status: str,
    submission: str,
) -> dict[str, Any]:
    """构造 mini-swe-agent 原生轨迹，供 tmp/stats.py 分析。"""
    return agent.serialize(
        {
            "info": {
                "exit_status": exit_status,
                "submission": submission,
            },
            "instance_id": instance_id,
        }
    )

def _extract_patch_from_trajectory(messages: list[dict]) -> str:
    """从 mini-swe-agent trajectory 中提取 patch。"""
    # 首先从最后一个 exit message 的 submission 中提取
    for msg in reversed(messages):
        if msg.get("role") == "exit":
            submission = msg.get("extra", {}).get("submission", "")
            if submission and "diff --git" in submission:
                return submission.strip()

    # 如果 exit message 没有 submission，从 tool messages 中查找
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            # 移除 XML 标签
            import re
            content = re.sub(r'<returncode>.*?</returncode>', '', content, flags=re.DOTALL)
            content = re.sub(r'<output>', '', content)
            content = re.sub(r'</output>', '', content)
            content = content.strip()

            if "diff --git" in content:
                return content

    return ""


def _get_docker_image_name(instance_id: str, dataset: str = "verified", bench_meta: dict[str, Any] | None = None) -> str:
    """根据 instance_id 获取 docker image 名称。

    - verified: swebench/sweb.eval.x86_64.<id>:latest (按 1776 替换规则)
    - swepro:   读取 bench meta 中的 docker_image 字段 (例 jefzda/sweap-images:<tag>)
    """
    if dataset == "swepro" and bench_meta:
        img = bench_meta.get("docker_image")
        if img:
            return img
    id_docker = instance_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{id_docker}:latest".lower()


def _run_test_in_isolated_container(
    image: str,
    patch: str,
    pytest_args: str,
    timeout: int = 300,
    *,
    workdir: str = "/testbed",
    entrypoint_empty: bool = False,
    before_repo_set_cmd: str | None = None,
) -> dict[str, Any]:
    """在临时隔离容器中运行测试。"""
    import docker
    import uuid

    client = docker.from_env()
    container = None

    try:
        run_kwargs: dict[str, Any] = dict(
            image=image,
            command=["sleep", "infinity"],
            detach=True,
            remove=False,
            name=f"swetest-{uuid.uuid4().hex[:8]}",
            working_dir=workdir,
        )
        if entrypoint_empty:
            run_kwargs["entrypoint"] = [""]
        container = client.containers.run(**run_kwargs)

        if before_repo_set_cmd:
            exit_code, output = container.exec_run(
                ["bash", "-lc", before_repo_set_cmd],
                workdir=workdir,
            )
            if exit_code != 0:
                return {
                    "output": f"before_repo_set_cmd failed:\n{output.decode(errors='replace')}",
                    "returncode": 1,
                }

        if patch.strip():
            exit_code, output = container.exec_run(
                f"bash -c 'cd {shlex.quote(workdir)} && git apply << \"PATCH_EOF\"\n{patch}\nPATCH_EOF'",
                workdir=workdir
            )
            if exit_code != 0:
                return {
                    "output": f"Failed to apply patch:\n{output.decode(errors='replace')}",
                    "returncode": 1
                }

        exit_code, output = container.exec_run(
            f"bash -c 'cd {shlex.quote(workdir)} && pytest {pytest_args}'",
            workdir=workdir
        )

        output_str = output.decode(errors='replace')
        lines = output_str.split('\n')
        if len(lines) > 2000:
            output_str = '\n'.join(lines[:2000]) + f'\n\n... (truncated {len(lines) - 2000} lines)'

        return {"output": output_str, "returncode": exit_code}

    except Exception as e:
        return {"output": f"Test execution failed: {str(e)}", "returncode": 1}

    finally:
        if container:
            try:
                container.stop(timeout=5)
                container.remove()
            except Exception:
                pass


def _read_regions_from_docker(
    env: DockerEnvironment,
    regions: list[dict[str, Any]],
    workdir: str = "/testbed",
) -> list[dict[str, Any]]:
    """从 Docker 容器读取 region 内容，填充 content 字段。"""
    enriched = []
    for r in regions:
        path, start, end = r["path"], r["start"], r["end"]
        if end == -1:
            cmd = f"sed -n '{start},$p' {workdir}/{path}"
        else:
            cmd = f"sed -n '{start},{end}p' {workdir}/{path}"
        result = env.execute({"command": cmd}, timeout=30)
        content = result.get("output", "")
        if content:
            enriched.append({"path": path, "start": start, "end": end, "content": content})
    return enriched


class TestEnabledDockerEnvironment(DockerEnvironment):
    """支持 swetest 命令的 DockerEnvironment 扩展。"""

    def __init__(self, *args, workdir: str = "/testbed", entrypoint_empty: bool = False,
                 before_repo_set_cmd: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_image = kwargs.get('image')
        self._workdir = workdir
        self._entrypoint_empty = entrypoint_empty
        self._before_repo_set_cmd = before_repo_set_cmd

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")

        if command.strip().startswith("swetest"):
            return self._execute_test(command, timeout)

        return super().execute(action, cwd, timeout=timeout)

    def _execute_test(self, command: str, timeout: int | None) -> dict[str, Any]:
        """执行 swetest 命令。"""
        pytest_args = command.replace("swetest", "").strip() or "tests/"

        patch_result = super().execute({"command": f"cd {shlex.quote(self._workdir)} && git diff HEAD"}, timeout=30)
        patch = patch_result.get("output", "")

        console.log(f"[cyan]Running tests in isolated container: pytest {pytest_args}[/cyan]")
        result = _run_test_in_isolated_container(
            image=self._test_image,
            patch=patch,
            pytest_args=pytest_args,
            timeout=timeout or 300,
            workdir=self._workdir,
            entrypoint_empty=self._entrypoint_empty,
            before_repo_set_cmd=self._before_repo_set_cmd,
        )

        return result


_SANITIZE_SCRIPT = r'''
import json, os, sys

workdir = sys.argv[1]
region_map = json.loads(sys.argv[2])
keep_files = set(region_map.keys())

# Phase 1: delete non-context files, empty __init__.py
for root, dirs, files in os.walk(workdir):
    parts = root.split(os.sep)
    if '.git' in parts:
        continue
    for fname in files:
        abs_path = os.path.join(root, fname)
        rel_path = os.path.relpath(abs_path, workdir)
        if rel_path in keep_files:
            continue
        if fname == '__init__.py':
            open(abs_path, 'w').close()
            continue
        try:
            os.remove(abs_path)
        except OSError:
            pass

# Phase 2: blank non-context lines in context files
for rel_path, regions in region_map.items():
    abs_path = os.path.join(workdir, rel_path)
    if not os.path.isfile(abs_path):
        continue
    with open(abs_path, 'r', errors='replace') as f:
        lines = f.readlines()
    kept = set()
    n = len(lines)
    for start, end in regions:
        # Treat end <= 0 (e.g. -1 sentinel) or end > n as "to end of file".
        s = max(1, start)
        e = n if (end is None or end < 0 or end > n) else end
        for i in range(s, e + 1):
            kept.add(i)
    new_lines = [line if (i + 1) in kept else '\n' for i, line in enumerate(lines)]
    with open(abs_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

print(f'sanitized: kept {len(keep_files)} files, deleted others')
'''


def _sanitize_testbed(
    env: DockerEnvironment,
    context_regions: list[dict[str, Any]],
    workdir: str = "/testbed",
) -> None:
    """清理 workdir (默认 /testbed) 环境：删除非 context 文件，非 context 行替换为空行。

    执行流程:
    1. 注入 Python 脚本执行 sanitize
    2. 清理空目录
    3. git add + commit 建立 baseline
    """
    # Step 1: 构建 region_map 并注入脚本执行
    region_map: dict[str, list[list[int]]] = {}
    for r in context_regions:
        region_map.setdefault(r["path"], []).append([r["start"], r["end"]])

    region_json = json.dumps(region_map)

    # 写脚本到容器内临时文件，避免 shell 转义问题
    env.execute(
        {"command": f"cat > /tmp/_sanitize.py << 'SANITIZE_EOF'\n{_SANITIZE_SCRIPT}\nSANITIZE_EOF"},
        timeout=10,
    )
    result = env.execute(
        {"command": f"python3 /tmp/_sanitize.py {shlex.quote(workdir)} {shlex.quote(region_json)}"},
        timeout=120,
    )
    console.log(f"Sanitize: {result.get('output', '').strip()}")

    # Step 2: 清理空目录
    qw = shlex.quote(workdir)
    env.execute(
        {"command": (
            f"find {qw} -type d -empty "
            f"-not -path '{workdir}/.git/*' -not -path '{workdir}' -delete"
        )},
        timeout=30,
    )

    # Step 3: git baseline
    env.execute({"command": (
        f"cd {qw} && git add -A && "
        "git -c user.email=x@x.com -c user.name=x commit -m 'sanitized baseline' --allow-empty"
    )}, timeout=30)


def run_mini_agent_single(
    instance: dict[str, Any],
    problem_statement: str,
    mode: str,
    model_name: str,
    api_key: str,
    base_url: str,
    config_path: Path,
    max_context_lines: int | None,
    max_steps: int | None,
    live_traj_path: Path | None = None,
    live_step_log_path: Path | None = None,
    raw_response_log_path: Path | None = None,
    explorer_regions: list[dict[str, Any]] | None = None,
    dataset: str = "verified",
) -> dict[str, Any]:
    """运行单个 instance（Docker 模式）。"""
    instance_id = instance["instance_id"]

    # 提取 region 元数据
    modified_files = instance["ground_truth"].get("modified_core_files") or []
    if mode == "explorer" and explorer_regions is not None:
        region_metadata = explorer_regions
    else:
        region_metadata = _extract_regions_metadata(instance, mode)

    if max_context_lines:
        region_metadata = _truncate_context(region_metadata, modified_files, max_context_lines)

    # SWE-bench-Pro vs Verified branch
    bench_meta = instance.get("meta") or {}
    is_pro = (dataset == "swepro") or bool(bench_meta.get("docker_image"))
    workdir = bench_meta.get("workdir") if is_pro else "/testbed"
    workdir = workdir or "/testbed"
    before_repo_set_cmd = bench_meta.get("before_repo_set_cmd") if is_pro else None

    # 加载配置
    image_name = _get_docker_image_name(
        instance_id,
        dataset="swepro" if is_pro else "verified",
        bench_meta=bench_meta if is_pro else None,
    )
    config = get_config_from_spec(str(config_path))
    agent_config: dict[str, Any] = {"output_path": None}
    if max_steps is not None:
        agent_config["step_limit"] = max_steps
    env_overrides: dict[str, Any] = {"image": image_name, "cwd": workdir}
    if is_pro:
        # Pro images have entrypoint=/bin/bash; clear it so `sleep 2h` runs as the cmd.
        env_overrides["run_args"] = ["--rm", "--entrypoint", ""]
    config = recursive_merge(config, {
        "model": {
            "model_name": model_name,
            "model_kwargs": {
                "api_key": api_key,
                "api_base": base_url,
            }
        },
        "agent": agent_config,
        "environment": env_overrides,
    })

    # 创建 Docker environment（支持测试）
    env_config = config.get("environment", {})
    env_config.pop("environment_class", None)
    env = TestEnabledDockerEnvironment(
        **env_config,
        workdir=workdir,
        entrypoint_empty=is_pro,
        before_repo_set_cmd=before_repo_set_cmd,
    )

    try:
        # Pro: container starts at base_commit (image baked from it). We skip
        # `before_repo_set_cmd` here because (a) the agent doesn't need the
        # test_patch (eval applies it in a fresh container), and (b) sanitize
        # would delete the freshly-checked-out test files anyway.

        # 在容器内 git init（docker image 可能没有 .git；Pro 已有）
        git_check = env.execute({"command": "test -d .git && echo HAS_GIT || echo NO_GIT"})
        if "NO_GIT" in git_check.get("output", ""):
            env.execute({"command": (
                "git init && git add -A && "
                "git -c user.email=x@x.com -c user.name=x commit -m init --allow-empty"
            )})

        # 从 Docker 容器读取 context 内容
        context_regions = []
        core_context = ""
        if mode in ("core_only", "core_optional", "explorer") and region_metadata:
            context_regions = _read_regions_from_docker(env, region_metadata, workdir=workdir)
            core_context = _format_context_block(context_regions)

        # sanitize：清理环境，只保留 explorer 声明的 region 路径。
        # 注意：必须基于 region_metadata 而非 context_regions，否则当所有声明
        # 路径都不存在（如 α=0% 的空 sentinel）时，sanitize 会被跳过，testbed
        # 反而会保留完整代码，破坏受控实验语义。
        if mode in ("core_only", "core_optional", "explorer") and region_metadata:
            _sanitize_testbed(env, region_metadata, workdir=workdir)

        model = RealtimeLoggingLitellmModel(
            instance_id=instance_id,
            raw_response_log_path=raw_response_log_path,
            **config.get("model", {}),
        )
        agent = RealtimeLoggingDefaultAgent(
            model,
            env,
            instance_id=instance_id,
            live_traj_path=live_traj_path,
            live_step_log_path=live_step_log_path,
            **config.get("agent", {}),
        )

        # 运行 agent
        def _run():
            return agent.run(task=problem_statement, core_context=core_context)

        result = _run()

        # 提取 patch
        raw_patch = _extract_patch_from_trajectory(agent.messages)
        patch = _fix_patch_format(_extract_diff(raw_patch)) if raw_patch else ""

        exit_status = result.get("exit_status", "")
        submission = result.get("submission", "")

        return {
            "instance_id": instance_id,
            "patch": patch,
            "raw_patch": raw_patch,
            "exit_status": exit_status,
            "num_steps": agent.n_calls,
            "cost": agent.cost,
            "trajectory": _build_debug_trajectory(agent.messages),
            "trajectory_data": _build_stats_trajectory(
                agent,
                instance_id,
                exit_status,
                submission,
            ),
        }
    finally:
        env.cleanup()


async def generate_patches_mini_agent(
    bench_path: Path,
    problem_statements: dict[str, str],
    output_path: Path,
    mode: str,
    model: str,
    api_key: str,
    base_url: str,
    config_path: Path,
    nums: int | None = None,
    dataset_ids: set[str] | None = None,
    instance_ids_filter: set[str] | None = None,
    log_dir: Path | None = None,
    max_context_lines: int | None = None,
    max_steps: int | None = None,
    workers: int = 8,
    regions_map: dict[str, list[dict[str, Any]]] | None = None,
    dataset_kind: str = "verified",
) -> int:
    """批量使用 mini-swe-agent 生成 patch（支持并发）。"""

    # 加载 bench
    items: list[dict[str, Any]] = []
    with open(bench_path) as f:
        for line in f:
            item = json.loads(line)
            if item["ground_truth"].get("read_core_regions"):
                items.append(item)
    console.log(f"Loaded {len(items)} instances with core_regions from {bench_path}")

    # 按数据集过滤
    if dataset_ids is not None:
        items = [it for it in items if it["instance_id"] in dataset_ids]
        console.log(f"Filtered to {len(items)} instances in selected dataset")

    # 按 instance list 过滤
    if instance_ids_filter is not None:
        items = [it for it in items if it["instance_id"] in instance_ids_filter]
        console.log(f"Filtered to {len(items)} instances by instance list")

    if nums is not None:
        items = items[:nums]
        console.log(f"Limited to first {nums} instances")

    # 断点续跑
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        console.log(f"Already completed: {len(done_ids)}, skipping")

    pending = [it for it in items if it["instance_id"] not in done_ids]
    pending = [it for it in pending if it["instance_id"] in problem_statements]
    if mode == "explorer" and regions_map is not None:
        pending = [it for it in pending if it["instance_id"] in regions_map]
    console.log(f"Pending: {len(pending)}")

    if not pending:
        return 0

    model_label = f"{model}__quality_{mode}_mini_agent"
    console.log(f"Workers: {workers}")

    written = 0
    errors = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_path, "a")

    # 日志文件
    log_f = None
    traj_dir = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"mini_swe_agent_{mode}_{ts}.jsonl"
        traj_dir = log_dir / f"mini_swe_agent_{mode}_{ts}_traj"
        traj_dir.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a")
        console.log(f"Logging debug jsonl to {log_path}")
        console.log(f"Logging traj json to {traj_dir}")

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(workers)

    async def _process_one(item: dict[str, Any]) -> bool:
        """处理单个 instance，返回是否成功。"""
        nonlocal written, errors
        iid = item["instance_id"]
        ps = problem_statements[iid]

        async with semaphore:
            try:
                live_traj_path = None
                live_step_log_path = None
                raw_response_log_path = None
                if traj_dir is not None:
                    live_traj_path = traj_dir / f"{iid}.traj.json"
                    live_step_log_path = traj_dir / f"{iid}.steps.jsonl"
                    raw_response_log_path = traj_dir / f"{iid}.llm_responses.jsonl"

                explorer_regions_for_instance = None
                if mode == "explorer" and regions_map is not None:
                    explorer_regions_for_instance = regions_map.get(iid, [])

                agent_result = await asyncio.to_thread(
                    run_mini_agent_single,
                    item,
                    ps,
                    mode,
                    model,
                    api_key,
                    base_url,
                    config_path,
                    max_context_lines,
                    max_steps,
                    live_traj_path=live_traj_path,
                    live_step_log_path=live_step_log_path,
                    raw_response_log_path=raw_response_log_path,
                    explorer_regions=explorer_regions_for_instance,
                    dataset=dataset_kind,
                )

                async with write_lock:
                    result = {
                        "instance_id": iid,
                        "model_name_or_path": model_label,
                        "model_patch": agent_result["patch"],
                    }
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    written += 1

                    if log_f is not None:
                        log_entry = {
                            "instance_id": iid,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "exit_status": agent_result["exit_status"],
                            "num_steps": agent_result["num_steps"],
                            "cost": agent_result["cost"],
                            "raw_patch": agent_result["raw_patch"],
                            "fixed_patch": agent_result["patch"],
                            "trajectory": agent_result.get("trajectory", []),
                        }
                        log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        log_f.flush()

                    if traj_dir is not None:
                        traj_path = traj_dir / f"{iid}.traj.json"
                        traj_path.write_text(
                            json.dumps(
                                agent_result["trajectory_data"],
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                return True
            except Exception as e:
                import traceback
                async with write_lock:
                    errors += 1
                    console.print(f"[red]Error {iid}: {e}[/red]")
                    if log_f is not None:
                        log_entry = {
                            "instance_id": iid,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "error": f"{e}\n{traceback.format_exc()}",
                        }
                        log_f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        log_f.flush()
                return False

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        ptask = progress.add_task(f"Mini-SWE-agent generating patches ({mode})", total=len(pending))

        async def _wrapped(item: dict[str, Any]) -> bool:
            result = await _process_one(item)
            progress.advance(ptask, 1)
            return result

        await asyncio.gather(*[_wrapped(item) for item in pending])

    out_f.close()
    if log_f is not None:
        log_f.close()
    console.print(f"\n[green]Written: {written}[/green]")
    if errors:
        console.print(f"[red]Errors: {errors}[/red]")
    return written


def main(
    bench: Path = typer.Option("bench.jsonl", "--bench", "-b", help="bench.jsonl 路径"),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="输出路径，默认 quality/outputs/predictions_{mode}_{model}mini_agent.jsonl",
    ),
    mode: str = typer.Option(
        "core_only", "--mode", "-m",
        help="实验模式: core_only / core_optional / no_context",
    ),
    model: str = typer.Option(
        "gpt-5-mini-2025-08-07", "--model",
        help="LLM 模型名称",
    ),
    api_key: str = typer.Option(
        DEFAULT_API_KEY, "--api-key", "-k",
        help="API key",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL, "--base-url",
        help="API base URL",
    ),
    nums: int = typer.Option(
        None, "--nums", "-n",
        help="只跑前 N 个 instance，不指定则跑全部",
    ),
    dataset: str = typer.Option(
        "all", "--dataset", "-d",
        help="数据集过滤: all / swe_bench_verified / swe_rebench / swe_bench",
    ),
    log_dir: Path = typer.Option(
        "logs/mini_swe_agent", "--log-dir",
        help="日志目录，保存调试 jsonl 和可供 tmp/stats.py 分析的 .traj.json",
    ),
    no_log: bool = typer.Option(
        False, "--no-log",
        help="禁用日志记录",
    ),
    max_context_lines: int = typer.Option(
        None, "--max-context-lines",
        help="最大 context 行数，超过则截断",
    ),
    max_steps: int = typer.Option(
        None, "--max-steps",
        help="最大 LLM 调用步数，超过则自动终止；默认使用配置文件中的 step_limit",
    ),
    workers: int = typer.Option(
        8, "--workers", "-w",
        help="并发 worker 数量",
    ),
    regions_file: Path = typer.Option(
        None, "--regions-file",
        help="Explorer results JSONL with regions (required when mode=explorer)",
    ),
    instance_list: Path = typer.Option(
        None, "--instance-list",
        help="Text file with instance IDs (one per line, auto-strips 'swebench.' prefix)",
    ),
    dataset_kind: str = typer.Option(
        "verified", "--dataset-kind",
        help="基准类型: verified (SWE-bench Verified, /testbed) 或 swepro (SWE-bench-Pro, /app)",
    ),
):
    """使用 mini-swe-agent 生成 patch (core context 作为初始上下文)。"""
    if mode not in ("core_only", "core_optional", "no_context", "explorer"):
        console.print(f"[red]Invalid mode: {mode}[/red]")
        raise typer.Exit(1)

    if mode == "explorer" and (regions_file is None or not regions_file.is_file()):
        console.print(f"[red]--regions-file is required and must exist when mode=explorer[/red]")
        raise typer.Exit(1)

    if dataset not in DATASET_CHOICES:
        console.print(f"[red]Invalid dataset: {dataset}. Choose from: {', '.join(DATASET_CHOICES)}[/red]")
        raise typer.Exit(1)

    if dataset_kind not in ("verified", "swepro"):
        console.print(f"[red]Invalid --dataset-kind: {dataset_kind} (must be verified or swepro)[/red]")
        raise typer.Exit(1)

    if output is None:
        output = Path(f"quality/outputs/predictions_{mode}_{model}mini_agent.jsonl")

    config_path = Path(__file__).parent / "configs" / "mini_swe_agent.yaml"
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise typer.Exit(1)

    # Load explorer regions map
    regions_map: dict[str, list[dict[str, Any]]] | None = None
    if mode == "explorer" and regions_file is not None:
        regions_map = {}
        with open(regions_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                iid = rec["instance_id"]
                regions_map[iid] = rec.get("regions", [])
        console.log(f"Loaded explorer regions for {len(regions_map)} instances from {regions_file}")

    dataset_ids = _load_dataset_instance_ids(dataset)

    # --instance-list 过滤
    instance_ids_filter: set[str] | None = None
    if instance_list is not None and instance_list.is_file():
        instance_ids_filter = set()
        for line in instance_list.read_text().splitlines():
            line = line.strip()
            if line:
                instance_ids_filter.add(line.removeprefix("swebench."))
        console.log(f"Instance list filter: {len(instance_ids_filter)} IDs from {instance_list}")

    ps = _load_problem_statements()
    if dataset_kind == "swepro":
        try:
            from datasets import load_dataset as _hf_load
            console.log("Loading ScaleAI/SWE-bench_Pro problem_statements (test split)...")
            ds = _hf_load("ScaleAI/SWE-bench_Pro", split="test")
            for row in ds:
                ps[row["instance_id"]] = row.get("problem_statement", "") or ""
            console.log(f"Loaded {len(ps)} problem_statements (incl. Pro)")
        except Exception as e:
            console.print(f"[red]Failed to load Pro problem_statements: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(generate_patches_mini_agent(
        bench, ps, output, mode, model, api_key, base_url, config_path,
        nums=nums,
        dataset_ids=dataset_ids,
        instance_ids_filter=instance_ids_filter,
        log_dir=None if no_log else log_dir,
        max_context_lines=max_context_lines,
        max_steps=max_steps,
        workers=workers,
        regions_map=regions_map,
        dataset_kind=dataset_kind,
    ))


if __name__ == "__main__":
    typer.run(main)
