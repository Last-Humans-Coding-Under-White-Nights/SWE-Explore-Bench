"""Run CLI agents with bounded shutdown and captured output."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
import signal
import subprocess
import tempfile
from threading import Event
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._windows_job import WindowsJob


# Worker threads do not receive Ctrl+C directly.
cli_cancel_event: ContextVar[Event | None] = ContextVar("cli_cancel_event", default=None)
_active_trees: dict[int, _ProcessTree] = {}
_STOP_GRACE = 1.0
_POLL_INTERVAL = 0.1
logger = logging.getLogger(__name__)


def kill_active_cli_trees():
    """POSIX hard-exit cleanup: no locks, logging, or waits in signal handlers."""
    if os.name == "posix":
        for tree in tuple(_active_trees.values()):
            if not tree.gone:
                try:
                    os.killpg(tree.process.pid, signal.SIGKILL)
                except OSError:
                    pass


@dataclass
class _ProcessTree:
    process: subprocess.Popen
    job: WindowsJob | None = None
    gone: bool = False

    def exists(self) -> bool:
        # Never signal a group ID after observing that it is available for reuse.
        if self.gone:
            return False
        if self.job is not None:
            # Assignment can fail while the direct child is still suspended.
            self.gone = not self.job.has_active_processes() and self.process.poll() is not None
        else:
            self.process.poll()
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                self.gone = True
            except PermissionError:
                pass  # EPERM does not prove that the group has disappeared.
        return not self.gone

    def wait(self, deadline: float) -> None:
        while self.exists() and time.monotonic() < deadline:
            time.sleep(min(_POLL_INTERVAL, max(0, deadline - time.monotonic())))

    def signal(self, *, force: bool, interrupt=False) -> None:
        if self.gone:
            return
        try:
            if self.job is None:
                sig = signal.SIGINT if interrupt else signal.SIGTERM
                os.killpg(self.process.pid, signal.SIGKILL if force else sig)
            elif force:
                self.job.terminate()
                self.process.kill()  # Also cover a child whose assignment failed.
            else:
                # Popen.send_signal skips groups whose leader has exited.
                os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
        except ProcessLookupError:
            self.gone = True
        except PermissionError as exc:
            # macOS can report EPERM for zombie-only groups.
            logger.debug("Could not signal CLI group %s: %s", self.process.pid, exc)
        except OSError:
            if self.job is None or force:
                raise
            # Some Windows hosts have no console. The job still owns the tree.


def _stop(
    tree: _ProcessTree, *, interrupt: bool = False,
) -> tuple[str | bytes | None, str | bytes | None]:
    process = tree.process
    deadline = time.monotonic() + _STOP_GRACE
    if tree.exists():
        tree.signal(force=False, interrupt=interrupt)
    try:
        output = process.communicate(timeout=_STOP_GRACE)
    except subprocess.TimeoutExpired:
        output = None
    # Helpers with redirected streams can survive leader exit and EOF.
    tree.wait(deadline)
    if tree.exists():
        tree.signal(force=True)
        tree.wait(time.monotonic() + _STOP_GRACE)
    if output is not None:
        return output
    try:
        return process.communicate(timeout=_STOP_GRACE)
    except subprocess.TimeoutExpired as exc:
        # A detached descendant can retain a pipe indefinitely.
        process.kill()
        process.wait(timeout=_STOP_GRACE)
        return exc.output, exc.stderr


def _force_reap(tree: _ProcessTree, error):
    """Last-resort cleanup; failures become notes on the original exception."""
    process = tree.process
    try:
        tree.signal(force=True)
    except Exception as exc:
        error.add_note(f"CLI group cleanup failed: {exc}")
    try:
        process.kill()
        process.wait(timeout=_STOP_GRACE)
    except Exception as exc:
        error.add_note(f"CLI child cleanup failed: {exc}")
    try:
        tree.wait(time.monotonic() + _STOP_GRACE)
    except Exception as exc:
        error.add_note(f"Waiting for CLI group cleanup failed: {exc}")


def _captured_output(process, files, stdout, stderr, *, on_error=False):
    streams = {"stdout": stdout, "stderr": stderr}
    for name, stream in files.items():
        stream.seek(0)
        streams[name] = stream.read()
    for name, data in streams.items():
        if process.text_mode and isinstance(data, bytes):
            errors = "replace" if on_error else (process.errors or "strict")
            data = data.decode(process.encoding, errors=errors)
            data = data.replace("\r\n", "\n").replace("\r", "\n")
        streams[name] = data
    return streams["stdout"], streams["stderr"]


def run_cli(cmd, *, input=None, timeout=None, capture_output=False, check=False, **kwargs):
    """Run a CLI with subprocess.run's input/capture/check conventions.

    POSIX descendants that stay in the new session's process group receive
    SIGINT on cancellation or SIGTERM on other failures, then SIGKILL after
    a grace period. Windows uses CTRL_BREAK and a job object.
    """
    cancel = cli_cancel_event.get()
    if cancel is not None and cancel.is_set():
        raise KeyboardInterrupt
    if input is not None:
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    job = None
    files = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:
        from ._windows_job import WindowsJob
        kwargs["creationflags"] = (kwargs.get("creationflags", 0)
                                   | subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000004)  # CREATE_SUSPENDED
    try:
        if os.name == "nt":
            job = WindowsJob()
            # Windows pipe readers provide no partial output on timeout.
            for name in ("stdout", "stderr"):
                if kwargs.get(name) == subprocess.PIPE:
                    files[name] = tempfile.TemporaryFile()
                    kwargs[name] = files[name]
        process = subprocess.Popen(cmd, **kwargs)
    except BaseException:
        if job is not None:
            job.close()
        for stream in files.values():
            stream.close()
        raise
    tree = _ProcessTree(process, job)
    _active_trees[id(tree)] = tree
    error = None
    stdout = stderr = None
    try:
        if job is not None:
            job.assign_and_resume(process)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancel is not None and cancel.is_set():
                raise KeyboardInterrupt
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            if remaining == 0:
                raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
            # Short waits keep cancellation prompt (worker threads, Windows SIGINT).
            interval = _POLL_INTERVAL if remaining is None else min(remaining, _POLL_INTERVAL)
            try:
                stdout, stderr = process.communicate(input=input, timeout=interval)
                break
            except subprocess.TimeoutExpired as exc:
                input = None  # communicate resumes buffered stdin; never resend it.
                stdout, stderr = exc.output, exc.stderr
        if check and process.returncode:
            # Preserve the exit status if cleanup also fails.
            raise subprocess.CalledProcessError(process.returncode, cmd)
        # Stop helpers before seeking files they may still be writing to.
        if tree.exists():
            stdout, stderr = _stop(tree)
        stdout, stderr = _captured_output(process, files, stdout, stderr)
        result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    except BaseException as exc:
        error = exc
        try:
            stdout, stderr = _stop(tree, interrupt=isinstance(exc, KeyboardInterrupt))
        except BaseException as cleanup_error:
            if isinstance(cleanup_error, Exception):
                exc.add_note(f"CLI cleanup failed: {cleanup_error}")
            else:
                error = cleanup_error
            _force_reap(tree, error)
        try:
            error.stdout, error.stderr = _captured_output(process, files, stdout, stderr, on_error=True)
        except Exception as output_error:
            error.add_note(f"Reading captured CLI output failed: {output_error}")
        if error is not exc:
            raise error
        raise
    finally:
        _active_trees.pop(id(tree), None)
        resources = [job, process.stdin, process.stdout, process.stderr, *files.values()]
        for resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except Exception as close_error:
                    if error is None:
                        raise
                    error.add_note(f"Closing CLI resource failed: {close_error}")
    return result
