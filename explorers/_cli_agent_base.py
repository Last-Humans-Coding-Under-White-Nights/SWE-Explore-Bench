"""Shared base for explorers that shell out to a coding-agent CLI."""
from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List

from .base import Explorer, ExplorerResult
from .parsing import (
    TokenUsage,
    extract_usage_from_jsonl,
    iter_events,
    parse_relevant_files,
    report_usage,
)

EXPLORE_PROMPT = """Explore this repository to find the source files and line ranges most relevant to understanding and fixing the following issue. Do NOT make any code changes.

Use available read-only repository navigation tools. Focus on finding the ROOT CAUSE, not just symptom locations.

VERY IMPORTANT: After exploration, output your findings in EXACTLY this format:
```
RELEVANT_FILES:
- path/to/file1.py:10-20
- path/to/file2.py:1-10
- path/to/file3.py:2-2
- path/to/file3.py:5-5
```

Focus on the root cause. Limit to top {top_k} most relevant regions.
{prompt_additions}

ISSUE DESCRIPTION FROM USER (very important):
{issue}

Do exactly this, but without modifications. You are planner, so you just provide ranges. Use SMALLER ranges whenever possible.
"""


ANSWER_MARKER = "RELEVANT_FILES:"

# The temp home holds only this run's sessions, sub-agents included.
SESSION_USAGE_QUERY = (
    "select parent_id is not null as sub, sum(tokens_input) as input,"
    " sum(tokens_output) as output, sum(tokens_reasoning) as reasoning,"
    " sum(tokens_cache_read) as cache_read, sum(tokens_cache_write) as cache_write"
    " from session group by sub"
)

XDG_VARS = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")
#: Subdirectory of a profile whose files are placed into the checkout per run.
CHECKOUT_SEED_DIR = "checkout"
#: A file written into the checkout for one run, with the content written.
SeededFile = tuple[Path, bytes]


@dataclass
class CheckoutSeed:
    files: list[SeededFile] = field(default_factory=list)
    directories: list[Path] = field(default_factory=list)


LOG_LEVELS = {"info": 0, "debug": 1, "trace": 2}
_log_level = 0


def set_log_level(level: str) -> None:
    """Set console log verbosity for CLI-agent explorers (info|debug|trace)."""
    global _log_level
    _log_level = LOG_LEVELS.get(level.lower(), LOG_LEVELS["info"])


def _log(message: str, level: str = "debug") -> None:
    if LOG_LEVELS.get(level, 0) > _log_level:
        return
    timestamp = time.strftime("%H:%M:%S")
    sys.stderr.write(f"\n  [{level} {timestamp}] {message}\n")
    sys.stderr.flush()


def _extract_output_text(raw: str) -> str:
    """Helper to return the agent's answer text from ``--format json`` stdout."""
    raw = raw.strip()
    if not raw:
        return ""

    texts: list[str] = []
    saw_event = False
    for event in iter_events(raw):
        if "type" not in event:
            continue
        saw_event = True
        if event["type"] != "text":
            continue
        part = event.get("part")
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("text"):
            texts.append(part["text"])

    if not texts:
        return "" if saw_event else raw
    for i in reversed(range(len(texts))):
        if ANSWER_MARKER in texts[i]:
            return "\n".join(texts[i:])
    return texts[-1]


@dataclass
class BaseCliAgentExplorer(Explorer):
    """Template-method base for explorers that drive a coding-agent CLI."""

    repo_root: Path
    bin_path: str = ""
    timeout: int = 2400
    config_dir: Path | None = None
    prompt_additions: str = ""

    # ── subclass hooks ──
    cli_display_name: ClassVar[str] = "CLI agent"
    config_env_var: ClassVar[str] = ""
    config_filename: ClassVar[str] = ""
    install_hint: ClassVar[str] = ""
    prompt_template: ClassVar[str] = EXPLORE_PROMPT

    config_override_vars: ClassVar[tuple[str, ...]] = ()

    def build_cmd(self) -> list[str]:
        """Return the argv for one run. The prompt is delivered on stdin."""
        raise NotImplementedError

    def _collect_usage(self, stdout_text: str, env: dict[str, str]) -> TokenUsage | None:
        """Token usage of the finished run; called before the temp home is removed."""
        try:
            proc = subprocess.run(
                [self.bin_path, "db", SESSION_USAGE_QUERY, "--format", "json"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                env=env,
            )
            usage = TokenUsage()
            for row in json.loads(proc.stdout):
                part = TokenUsage(
                    input_tokens=row["input"],
                    output_tokens=row["output"],
                    cache_read_tokens=row["cache_read"],
                    cache_write_tokens=row["cache_write"],
                    reasoning_tokens=row["reasoning"],
                    # Reasoning is reported beside output, not inside it.
                    separate_reasoning_tokens=row["reasoning"],
                )
                if row["sub"]:
                    part.subagent_tokens = part.total
                usage.add(part)
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, KeyError):
            usage = None
        if usage is None or not usage.has_any():
            _log(
                f"{self.cli_display_name}: session store unreadable, "
                "token usage excludes sub-agents if there were any",
                level="info",
            )
            return extract_usage_from_jsonl(stdout_text)
        return usage

    def _format_prompt(self, query: str, top_k: int) -> str:
        return self.prompt_template.format(
            issue=query, top_k=top_k, prompt_additions=self.prompt_additions
        ).strip()

    def _isolate_env(self, env: dict[str, str], tmp_home: Path) -> None:
        """Redirect config discovery to an empty temporary home."""
        env["HOME"] = str(tmp_home)
        env["USERPROFILE"] = str(tmp_home)
        # Unset, so each of these defaults to a path under HOME (tempdir)
        for var in XDG_VARS + self.config_override_vars:
            env.pop(var, None)

    def _prepare_config(self, env: dict[str, str]) -> None:
        """Select ``config_dir`` as the CLI's profile."""
        if self.config_dir is None:
            return
        config_src = self.config_dir / self.config_filename
        if not config_src.exists():
            raise FileNotFoundError(f"{self.config_filename} not found at {config_src}")
        env[self.config_env_var] = str(self.config_dir.resolve())

    def _seed_checkout(self) -> CheckoutSeed:
        """Copy profile files, recording only paths created by this run.

        Roll back here on failure, before a partial seed can be lost or a
        source-file error can be mistaken for a missing CLI binary.
        """
        seeded = CheckoutSeed()
        if self.config_dir is None:
            return seeded
        seed_root = self.config_dir / CHECKOUT_SEED_DIR
        if not seed_root.is_dir():
            return seeded
        try:
            for source in sorted(p for p in seed_root.rglob("*") if p.is_file()):
                target = self.repo_root / source.relative_to(seed_root)
                if target.exists() or target.is_symlink():
                    continue
                parents = []
                parent = target.parent
                while parent != self.repo_root:
                    parents.append(parent)
                    parent = parent.parent
                # Never write through a checkout directory symlink.
                if any(parent.is_symlink() for parent in parents):
                    continue
                content = source.read_bytes()
                for parent in reversed(parents):
                    if not parent.exists():
                        parent.mkdir()
                        seeded.directories.append(parent)
                # Exclusive creation also protects paths created concurrently.
                try:
                    stream = target.open("xb")
                except FileExistsError:
                    continue
                try:
                    with stream:
                        stream.write(content)
                except BaseException:
                    try:
                        target.unlink(missing_ok=True)
                    except OSError as exc:
                        _log(f"Could not remove partial seed {target}: {exc}", "info")
                    raise
                seeded.files.append((target, content))
        except BaseException:
            self._unseed_checkout(seeded)
            raise
        return seeded

    def _unseed_checkout(self, seeded: CheckoutSeed) -> None:
        """Best-effort removal of unchanged seed files and created empty dirs."""
        for target, content in seeded.files:
            try:
                if (not target.is_symlink() and target.is_file()
                        and target.read_bytes() == content):
                    target.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                _log(f"Could not remove seed {target}: {exc}", "info")
        for directory in reversed(seeded.directories):
            try:
                directory.rmdir()
            except OSError as exc:
                if exc.errno not in (errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST):
                    _log(f"Could not remove seed directory {directory}: {exc}", "info")

    def explore(
        self, *, instance_id: str, query: str, top_k: int = 5
    ) -> List[ExplorerResult]:
        prompt = self._format_prompt(query, top_k)
        cmd = self.build_cmd()
        env = dict(os.environ)

        proc_t0 = time.perf_counter()
        _log(
            f"{self.cli_display_name} {instance_id}: launching "
            f"`{' '.join(cmd)}` (cwd={self.repo_root}, "
            f"prompt={len(prompt)} chars, timeout={self.timeout}s)"
        )

        # A temp HOME keeps the run from picking up the user's own config.
        with tempfile.TemporaryDirectory(prefix="cli-agent-home-") as tmp_home:
            self._isolate_env(env, Path(tmp_home))
            self._prepare_config(env)
            seeded = self._seed_checkout()

            try:
                completed = subprocess.run(
                    cmd,
                    input=prompt,
                    cwd=str(self.repo_root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout if self.timeout > 0 else None,
                    env=env,
                )
            except FileNotFoundError:
                _log(f"{self.cli_display_name} {instance_id}: binary not found")
                raise RuntimeError(
                    f"{self.cli_display_name} not found. {self.install_hint}".strip()
                )
            except subprocess.TimeoutExpired:
                _log(
                    f"{self.cli_display_name} {instance_id}: timed out "
                    f"after {self.timeout}s"
                )
                raise RuntimeError(
                    f"{self.cli_display_name} timed out after {self.timeout}s"
                )
            finally:
                self._unseed_checkout(seeded)

            stderr_text = completed.stderr or ""
            stdout_text = completed.stdout or ""
            tail = f" stderr_tail={stderr_text[-300:]!r}" if stderr_text else ""
            _log(
                f"{self.cli_display_name} {instance_id}: rc={completed.returncode} "
                f"in {time.perf_counter() - proc_t0:.1f}s "
                f"(stdout={len(stdout_text)}B stderr={len(stderr_text)}B){tail}"
            )
            usage = self._collect_usage(stdout_text, env)

        output = _extract_output_text(stdout_text)
        # Best-effort: report token usage before the rc check below can
        # raise, so failed-but-expensive runs still hand their spend to the
        # collector (see _eval_one).
        report_usage(usage)

        if completed.returncode != 0:
            stdout_preview = (completed.stdout or "")[-2000:]
            stderr_preview = (completed.stderr or "")[-2000:]
            detail = stderr_preview
            if stdout_preview:
                detail = f"STDOUT:\n{stdout_preview}\nSTDERR:\n{stderr_preview}"
            raise RuntimeError(
                f"{self.cli_display_name} failed (rc={completed.returncode}):\n{detail}"
            )

        if not output:
            return []
        _log(
            f"{self.cli_display_name} {instance_id}: agent output:\n{output}",
            level="trace",
        )
        return parse_relevant_files(
            output, instance_id, top_k=top_k, repo_path=self.repo_root,
        )
