"""Shared base for explorers that shell out to a coding-agent CLI."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, List

from ._cli_process import run_cli
from .base import (
    BINARY_NOT_FOUND,
    ERROR,
    INVALID_OUTPUT,
    PROVIDER_ERROR,
    TIMEOUT,
    Explorer,
    ExplorerFailure,
    ExplorerResult,
)
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


def _stream_errors(raw: str) -> list[str]:
    """Messages of the ``error`` events on a ``--format json`` stream.

    OpenCode reports a failed provider call as
    ``{"type": "error", "error": {"name": ..., "data": {"message": ...}}}``,
    with an exit code of 1 or, depending on where it failed, 0.

    An agent that retried one failing call reports it once per attempt, so
    identical messages are collapsed: the row records what went wrong, not how
    many times the same sentence was printed.
    """
    messages: list[str] = []
    for event in iter_events(raw):
        if event.get("type") != "error":
            continue
        # Every other event carries its payload under `part`, and an error
        # event is written both ways in the wild; read either, since a row
        # that says "error event" tells nobody anything.
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        error = event.get("error") if event.get("error") is not None else part.get("error")
        if not isinstance(error, dict):
            message = str(error or "error event")
        else:
            data = error.get("data") if isinstance(error.get("data"), dict) else {}
            text = data.get("message") or error.get("message")
            name = error.get("name")
            message = ": ".join(str(p) for p in (name, text) if p) or "error event"
        if message not in messages:
            messages.append(message)
    return messages


#: How many `{` positions `_first_json_object` will try before giving up, so
#: output that is all braces and no JSON cannot make the scan quadratic.
_JSON_SCAN_LIMIT = 64


def _first_json_object(text: str) -> dict | None:
    """The first complete JSON object in ``text``, ignoring what surrounds it.

    A CLI prints its configuration among its own chatter: a plugin's
    ``[plugin] loaded {foo}`` line before it, a ``Done in 12ms`` after it.
    Decoding the whole tail rejects the trailing line outright, and starting
    at the first ``{`` can land in the leading one, so each candidate is
    decoded in turn and the first that yields an object wins. A line-initial
    ``{`` is tried first, being where a pretty-printed object starts.
    """
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    starts.sort(key=lambda i: not (i == 0 or text[i - 1] == "\n"))
    decoder = json.JSONDecoder()
    for start in starts[:_JSON_SCAN_LIMIT]:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _decode(stream: str | bytes | None) -> str:
    """``TimeoutExpired`` carries raw bytes even for a ``text=True`` run."""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream or ""


def _output_detail(stdout_text: str, stderr_text: str) -> str:
    """The tail of what the CLI printed, naming only the streams it used."""
    parts = []
    if stdout_text:
        parts.append(f"STDOUT:\n{stdout_text[-2000:]}")
    if stderr_text:
        parts.append(f"STDERR:\n{stderr_text[-2000:]}")
    return "\n".join(parts)


# Config keys whose values are credentials. The manifest hashes the resolved
# configuration with these replaced, so it neither holds a key nor changes
# when one is rotated.
_SECRET_SUFFIXES = (
    "key", "keys", "keyid", "secret", "secrets", "password", "passwords",
    "passwd", "passphrase", "passphrases", "token", "tokens", "credential",
    "credentials", "cookie", "cookies", "authorization",
)
# ...but a key that counts something is a limit, not a credential. `maxTokens`
# ends in `tokens` and is configuration: redacting it would hide a real change
# to the run from the manifest. Over-redaction costs identity, not safety, so
# this list stays short and deliberate.
_QUANTITY_PREFIXES = ("max", "min", "num", "total", "count", "limit")
# Names that are a credential in full but not as a suffix: `oauth` ends in
# `auth` and names a provider block, not a secret.
_SECRET_NAMES = frozenset({"auth"})
# Maps whose values are handed to a process or a server verbatim; only their
# names are kept.
_OPAQUE_MAPS = {"environment", "env", "headers"}
REDACTED = "<redacted>"


def _is_secret_key(key: str) -> bool:
    # apiKey, api_keys, accessKeyId, accessToken, password, auth — but not
    # maxTokens, keybinds or oauth.
    normalized = re.sub(r"[^a-z]", "", key.lower())
    if normalized.startswith(_QUANTITY_PREFIXES):
        return False
    return normalized in _SECRET_NAMES or normalized.endswith(_SECRET_SUFFIXES)


def redact_config(node: Any, key: str = "") -> Any:
    """``node`` with every credential-shaped value replaced by ``REDACTED``."""
    if isinstance(node, dict):
        if key.lower() in _OPAQUE_MAPS:
            return {k: REDACTED for k in sorted(node)}
        # A credential can be a whole object — `{"auth": {"data": "sk-1"}}`
        # names the secret only at the top — so the key travels down to any
        # child that does not name one of its own. A child that names an
        # opaque map keeps its own name too, or the branch above could never
        # be reached from inside a configuration.
        return {
            k: redact_config(v, k if _is_secret_key(k) or k.lower() in _OPAQUE_MAPS else key)
            for k, v in node.items()
        }
    if isinstance(node, list):
        # The key travels into the list too: `{"api_keys": ["sk-1", "sk-2"]}`
        # is two credentials, and dropping the key here left them in the clear.
        return [redact_config(v, key) for v in node]
    # Credentials are strings. A number or a flag under a credential-shaped
    # key (`"tokens": 4096`) is a setting, and hiding it would keep a real
    # change to the run out of the manifest.
    return REDACTED if isinstance(node, str) and _is_secret_key(key) else node


def sha256_json(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class BaseCliAgentExplorer(Explorer):
    """Template-method base for explorers that drive a coding-agent CLI."""

    repo_root: Path
    bin_path: str = ""
    timeout: int = 2400
    config_dir: Path | None = None
    prompt_additions: str = ""
    #: ``provider/model`` sent as ``--model``; empty leaves it to the config.
    model: str = ""
    #: Agent sent as ``--agent``; empty leaves it to the config.
    agent: str = ""

    # ── subclass hooks ──
    cli_display_name: ClassVar[str] = "CLI agent"
    config_env_var: ClassVar[str] = ""
    config_filename: ClassVar[str] = ""
    install_hint: ClassVar[str] = ""
    prompt_template: ClassVar[str] = EXPLORE_PROMPT
    #: The agent the CLI runs when none is named and the configuration sets
    #: no default. Its model outranks the top-level one just as a named
    #: agent's does, so the manifest has to know it exists.
    implicit_agent: ClassVar[str | None] = None

    config_override_vars: ClassVar[tuple[str, ...]] = ()
    session_usage_query: ClassVar[str | None] = None

    def build_cmd(self) -> list[str]:
        """Return the argv for one run. The prompt is delivered on stdin."""
        raise NotImplementedError

    def _selection_args(self) -> list[str]:
        """``--model`` / ``--agent`` for the selections made explicitly."""
        args = []
        if self.model:
            args += ["--model", self.model]
        if self.agent:
            args += ["--agent", self.agent]
        return args

    # ── run manifest ──

    def _probe(self, args: list[str], env: dict[str, str], cwd: str) -> str | None:
        """Stdout of a short side command, or None if it could not run.

        A missing binary is not "could not run": it is a run that cannot
        produce a single result, and `describe()` learns it at startup, so it
        is raised rather than logged. Every other failure leaves the field
        unknown — see `describe()`.
        """
        try:
            proc = run_cli(
                [self.bin_path, *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                env=env,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{self.cli_display_name} not found at {self.bin_path!r}. "
                f"{self.install_hint}".strip()
            ) from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            _log(f"{self.cli_display_name}: `{' '.join(args)}` failed: {exc}", "info")
            return None
        if proc.returncode != 0:
            _log(
                f"{self.cli_display_name}: `{' '.join(args)}` rc={proc.returncode} "
                f"stderr={(proc.stderr or '')[-300:]!r}",
                "info",
            )
            return None
        return proc.stdout

    def _profile_config(self) -> dict | None:
        if self.config_dir is None:
            return None
        try:
            data = json.loads(
                (self.config_dir / self.config_filename).read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def describe(self) -> dict[str, Any]:
        """What configuration this explorer runs with, for the run manifest.

        The CLI merges its configuration from several sources, so the profile
        file alone does not pin it down: this asks the CLI for the resolved
        configuration (``debug config``) under the same isolated home a case
        runs in, and records a hash of it with every credential redacted. No
        value in the result is a secret.

        ``--pure`` resolves the configuration without plugins, so a plugin's
        config hook is not in ``resolved_config_sha256`` even though a run
        would load it. The shipped profiles have no plugins. Seed files are not
        in ``debug config``; ``seed_sha256`` covers them.

        A probe that fails for any reason other than a missing binary — a
        timeout on a loaded machine, a CLI that has no ``debug config`` —
        leaves its field ``None``, which means *unknown*, not *absent*.
        `_manifest_diff` treats it as unknown too, so one slow startup does
        not refuse every later resume.
        """
        version = resolved = None
        with tempfile.TemporaryDirectory(prefix="cli-agent-home-") as tmp_home:
            env = dict(os.environ)
            self._isolate_env(env, Path(tmp_home))
            self._prepare_config(env)
            # An empty cwd: a checkout's own project config is not part of
            # the configuration the run was given.
            workdir = Path(tmp_home) / "workdir"
            workdir.mkdir()
            stdout = self._probe(["--version"], env, str(workdir))
            if stdout is not None:
                version = stdout.strip() or None
            stdout = self._probe(["debug", "config", "--pure"], env, str(workdir))
            if stdout is not None:
                resolved = _first_json_object(stdout)
        # `resolved or ...` would discard a configuration that resolves to {}.
        config = resolved if resolved is not None else self._profile_config() or {}
        # An agent may name its own model, and it outranks the top-level one:
        # pinning the global model on the command line would otherwise
        # override the agent's choice. That holds for the agent the CLI falls
        # back to as much as for one `--agent` named, so `implicit_agent` is
        # resolved last but consulted all the same.
        agent = self.agent or config.get("default_agent") or self.implicit_agent or None
        agents = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        agent_config = agents.get(agent) if isinstance(agents.get(agent), dict) else {}
        agent_model = agent_config.get("model") or None
        model = self.model or agent_model or config.get("model") or None
        if self.model:
            model_source = "flag"
        elif agent_model:
            model_source = "agent"
        else:
            model_source = "config" if model else None
        return {
            "cli": self.cli_display_name,
            "cli_version": version,
            "model": model,
            "model_source": model_source,
            "agent": agent,
            "prompt_sha256": hashlib.sha256(
                f"{self.prompt_template}\0{self.prompt_additions}".encode("utf-8")
            ).hexdigest(),
            "resolved_config_sha256": (
                sha256_json(redact_config(resolved)) if resolved is not None else None
            ),
            "seed_sha256": sha256_json(
                {rel: sha256_file(path) for rel, path in self._seed_files().items()}
            ),
        }

    def _collect_usage(self, stdout_text: str, env: dict[str, str]) -> TokenUsage | None:
        """Token usage of the finished run; called before the temp home is removed."""
        stream = extract_usage_from_jsonl(stdout_text)
        if not self.session_usage_query:
            return stream
        # The session columns are flat; only the stdout events show whether
        # reasoning sits beside output or inside it.
        separate_reasoning = not (
            stream and stream.reasoning_tokens and not stream.separate_reasoning_tokens
        )
        usage = None
        try:
            proc = run_cli(
                # --pure: plugin chatter on stdout would break the JSON parse.
                [self.bin_path, "db", "--pure", self.session_usage_query,
                 "--format", "json"],
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
                reasoning = int(row.get("reasoning") or 0)
                part = TokenUsage(
                    input_tokens=int(row.get("input") or 0),
                    output_tokens=int(row.get("output") or 0),
                    cache_read_tokens=int(row.get("cache_read") or 0),
                    cache_write_tokens=int(row.get("cache_write") or 0),
                    reasoning_tokens=reasoning,
                    separate_reasoning_tokens=reasoning if separate_reasoning else 0,
                )
                if row.get("sub"):
                    part.subagent = replace(part)
                usage.add(part)
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError,
                AttributeError) as exc:
            usage, reason = None, f"{type(exc).__name__}: {exc}"
        else:
            reason = f"rc={proc.returncode} stderr={(proc.stderr or '')[-300:]!r}"
        if usage is None or not usage.has_any():
            _log(
                f"{self.cli_display_name}: no session-store usage ({reason}); "
                "falling back to the stdout count, which excludes sub-agents",
                level="info",
            )
            return stream
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

    def _seed_files(self) -> dict[str, Path]:
        """Profile files placed into the checkout per run, by relative path."""
        if self.config_dir is None:
            return {}
        seed_root = self.config_dir / CHECKOUT_SEED_DIR
        return {
            path.relative_to(seed_root).as_posix(): path
            for path in sorted(seed_root.rglob("*")) if path.is_file()
        }

    def _seed_checkout(self) -> CheckoutSeed:
        """Copy profile files, recording only paths created by this run.

        Roll back here on failure, before a partial seed can be lost or a
        source-file error can be mistaken for a missing CLI binary.
        """
        seeded = CheckoutSeed()
        try:
            for rel, source in self._seed_files().items():
                target = self.repo_root / rel
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

            returncode: int | None = None
            try:
                completed = run_cli(
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
                raise ExplorerFailure(
                    BINARY_NOT_FOUND,
                    f"{self.cli_display_name} not found. {self.install_hint}".strip(),
                )
            except subprocess.TimeoutExpired as exc:
                # What the CLI printed before it was killed holds the usage
                # events and the reason it stalled; keep it.
                stdout_text, stderr_text = _decode(exc.stdout), _decode(exc.stderr)
            else:
                returncode = completed.returncode
                stdout_text, stderr_text = completed.stdout or "", completed.stderr or ""
            finally:
                self._unseed_checkout(seeded)

            tail = f" stderr_tail={stderr_text[-300:]!r}" if stderr_text else ""
            status = "timed out" if returncode is None else f"rc={returncode}"
            _log(
                f"{self.cli_display_name} {instance_id}: {status} "
                f"in {time.perf_counter() - proc_t0:.1f}s "
                f"(stdout={len(stdout_text)}B stderr={len(stderr_text)}B){tail}"
            )
            usage = self._collect_usage(stdout_text, env)

        # Report token usage before any failure below is raised, so a failed
        # but expensive run still hands its spend to the collector.
        report_usage(usage)
        # The streams are the agent's raw output — model text among the
        # events — so they go to the log, never into a result row that may
        # be published. A row carries the classified message instead.
        _log(
            f"{self.cli_display_name} {instance_id}: output\n"
            f"{_output_detail(stdout_text, stderr_text)}",
            level="trace",
        )

        if returncode is None:
            raise ExplorerFailure(
                TIMEOUT, f"{self.cli_display_name} timed out after {self.timeout}s"
            )
        if returncode != 0:
            # A non-zero exit says the run failed, not whose fault it was: a
            # bad flag and a refused API key both land here. The provider is
            # only named when the stream says so.
            errors = _stream_errors(stdout_text)
            if errors:
                raise ExplorerFailure(
                    PROVIDER_ERROR,
                    f"{self.cli_display_name} failed (rc={returncode}): "
                    + "; ".join(errors),
                )
            raise ExplorerFailure(
                ERROR, f"{self.cli_display_name} failed (rc={returncode})"
            )

        output = _extract_output_text(stdout_text)
        results = []
        if output:
            _log(
                f"{self.cli_display_name} {instance_id}: agent output:\n{output}",
                level="trace",
            )
            results = parse_relevant_files(
                output, instance_id, top_k=top_k, repo_path=self.repo_root,
            )
        # An answer that names no region is still an answer when it uses the
        # output contract; anything else is a run that never produced one.
        if results or ANSWER_MARKER in output:
            return results
        # Only reached without an answer, so the stream is parsed a second
        # time on the one path that reads the result.
        errors = _stream_errors(stdout_text)
        if errors:
            raise ExplorerFailure(
                PROVIDER_ERROR,
                f"{self.cli_display_name} reported errors and no answer (rc=0): "
                + "; ".join(errors),
            )
        raise ExplorerFailure(
            INVALID_OUTPUT,
            f"{self.cli_display_name} exited cleanly without a {ANSWER_MARKER} answer",
        )
