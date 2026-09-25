"""Cleanup must preserve the original failure and reap the direct child."""
import os
import asyncio
import shlex
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

from explorers import _cli_process as runner
from test_cli_process_tree import alive, wait_until


@pytest.fixture
def launched(monkeypatch):
    processes = []
    popen = subprocess.Popen

    def record(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, 'Popen', record)
    yield processes
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)



_POPEN = subprocess.Popen


@pytest.mark.parametrize('error', [ValueError('reader failed'), KeyboardInterrupt()])
def test_unexpected_failure_reaps_child(monkeypatch, launched, error):
    original = _POPEN.communicate

    def fail_once(process, *args, **kwargs):
        if not getattr(process, '_injected', False):
            process._injected = True
            raise error
        return original(process, *args, **kwargs)

    monkeypatch.setattr(_POPEN, 'communicate', fail_once)
    with pytest.raises(type(error)) as caught:
        runner.run_cli([sys.executable, '-c', 'import time; time.sleep(60)'],
                       capture_output=True)
    assert caught.value is error
    assert launched[0].poll() is not None


def test_cleanup_error_does_not_replace_cancellation(monkeypatch, launched):
    error = KeyboardInterrupt()

    def interrupt(*args, **kwargs):
        raise error

    def broken_stop(*args, **kwargs):
        raise subprocess.TimeoutExpired('cleanup', 1)

    monkeypatch.setattr(_POPEN, 'communicate', interrupt)
    monkeypatch.setattr(runner, '_stop', broken_stop)
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.run_cli([sys.executable, '-c', 'import time; time.sleep(60)'])
    assert caught.value is error
    assert launched[0].poll() is not None
    assert any('cleanup' in note.lower() for note in error.__notes__)


@pytest.mark.parametrize('interrupt', [KeyboardInterrupt(), SystemExit(130)])
def test_cancellation_during_timeout_cleanup_wins(monkeypatch, launched, interrupt):
    def cancel_cleanup(*args, **kwargs):
        raise interrupt

    monkeypatch.setattr(runner, '_stop', cancel_cleanup)
    with pytest.raises(type(interrupt)) as caught:
        runner.run_cli([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=.1)
    assert caught.value is interrupt
    assert launched[0].poll() is not None


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permission probe')
def test_permission_denied_probe_does_not_cache_group_as_gone(monkeypatch):
    def denied(pid, sig):
        raise PermissionError('group exists but cannot be signalled')

    monkeypatch.setattr(runner.os, 'killpg', denied)
    tree = runner._ProcessTree(SimpleNamespace(pid=123, poll=lambda: 0))
    assert tree.exists()
    assert not tree.gone


def test_empty_job_does_not_hide_unassigned_child():
    tree = runner._ProcessTree(
        SimpleNamespace(poll=lambda: None),
        SimpleNamespace(has_active_processes=lambda: False),
    )
    assert tree.exists()


def test_force_cleanup_does_not_depend_on_job_query():
    terminated = []

    def broken_query():
        raise OSError('job query failed')

    tree = runner._ProcessTree(
        SimpleNamespace(kill=lambda: None, wait=lambda **kw: None),
        SimpleNamespace(has_active_processes=broken_query, terminate=lambda: terminated.append(True)),
    )
    runner._force_reap(tree, KeyboardInterrupt())
    assert terminated == [True]


def test_windows_polite_signal_reaches_group_after_leader_exit(monkeypatch):
    sent = []
    monkeypatch.setattr(runner.signal, 'CTRL_BREAK_EVENT', 1, raising=False)
    monkeypatch.setattr(runner.os, 'kill', lambda pid, sig: sent.append((pid, sig)))
    tree = runner._ProcessTree(
        SimpleNamespace(pid=123, returncode=0, send_signal=lambda sig: None),
        SimpleNamespace(has_active_processes=lambda: True),
    )
    tree.signal(force=False)
    assert sent == [(123, 1)]


def test_stop_waits_for_helpers_after_force(monkeypatch):
    from threading import Event, Timer

    stopped = Event()
    timers = []

    def terminate():
        timer = Timer(.1, stopped.set)
        timers.append(timer)
        timer.start()

    tree = runner._ProcessTree(
        SimpleNamespace(pid=123, poll=lambda: 0, communicate=lambda **kw: ('out', 'err'),
                        send_signal=lambda sig: None, kill=lambda: None),
        SimpleNamespace(has_active_processes=lambda: not stopped.is_set(), terminate=terminate),
    )
    monkeypatch.setattr(runner.signal, 'CTRL_BREAK_EVENT', 1, raising=False)
    monkeypatch.setattr(runner.os, 'kill', lambda pid, sig: None)
    try:
        assert runner._stop(tree) == ('out', 'err')
        assert stopped.is_set(), 'cleanup returned before forced termination finished'
    finally:
        for timer in timers:
            timer.join()


def test_partial_pipe_output_is_decoded_in_text_mode():
    process = SimpleNamespace(text_mode=True, encoding='utf-8', errors=None)
    assert runner._captured_output(
        process, {}, b'caf\xc3\xa9\r\n', b'incomplete \xc3', on_error=True,
    ) == ('café\n', 'incomplete \ufffd')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX group signal errors')
def test_permission_error_does_not_replace_timeout(monkeypatch, launched):
    killpg = os.killpg

    def deny_term(pid, sig):
        if sig == signal.SIGTERM:
            raise PermissionError('zombie group')
        return killpg(pid, sig)

    monkeypatch.setattr(runner.os, 'killpg', deny_term)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_cli([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=.1)
    assert launched[0].poll() is not None


def test_no_force_signal_after_group_disappears(monkeypatch):
    def signal_tree(*args, **kwargs):
        pytest.fail('must not signal a group already observed gone')

    process = SimpleNamespace(communicate=lambda **kwargs: ('out', 'err'))
    tree = runner._ProcessTree(process, gone=True)
    monkeypatch.setattr(tree, 'signal', signal_tree)
    assert runner._stop(tree) == ('out', 'err')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX group identity')
def test_fallback_keeps_group_disappearance(monkeypatch):
    def gone(pid, sig):
        raise ProcessLookupError

    process = SimpleNamespace(pid=123, poll=lambda: 0, kill=lambda: None, wait=lambda **kw: None)
    tree = runner._ProcessTree(process)
    monkeypatch.setattr(runner.os, 'killpg', gone)
    assert not tree.exists()

    def reused(pid, sig):
        pytest.fail('fallback must not probe or signal the reused group ID')

    monkeypatch.setattr(runner.os, 'killpg', reused)
    runner._force_reap(tree, KeyboardInterrupt())


def test_run_cli_check_reports_captured_failure():
    with pytest.raises(subprocess.CalledProcessError) as caught:
        runner.run_cli([sys.executable, '-c', "print('failure'); exit(3)"],
                       capture_output=True, check=True, text=True, encoding='utf-8')
    assert caught.value.returncode == 3
    assert caught.value.stdout == 'failure\n'


@pytest.mark.parametrize('errors', [None, 'strict', 'replace'])
def test_file_capture_decodes_text_with_default_or_explicit_errors(errors):
    with tempfile.TemporaryFile() as output:
        with subprocess.Popen(
            [sys.executable, '-c', "import os; os.write(1, b'caf\\xc3\\xa9\\r\\n')"],
            stdout=output, text=True, encoding='utf-8', errors=errors,
        ) as process:
            process.wait(timeout=5)
            stdout, stderr = runner._captured_output(process, {'stdout': output}, None, None)
    assert stdout == 'café\n'
    assert stderr is None


def test_file_capture_default_errors_rejects_invalid_text():
    with tempfile.TemporaryFile() as output:
        with subprocess.Popen(
            [sys.executable, '-c', "import os; os.write(1, b'\\xff')"],
            stdout=output, text=True, encoding='utf-8',
        ) as process:
            process.wait(timeout=5)
            with pytest.raises(UnicodeDecodeError):
                runner._captured_output(process, {'stdout': output}, None, None)


def test_default_capture_does_not_replace_supplied_stdout(tmp_path):
    path = tmp_path / 'output'
    with path.open('wb') as output:
        result = runner.run_cli([sys.executable, '-c', "print('kept')"], stdout=output)
    assert result.stdout is None
    assert path.read_bytes().strip() == b'kept'


@pytest.mark.parametrize(('returncode', 'fail_cleanup'), [(0, False), (3, False), (3, True)])
@pytest.mark.parametrize('attempt', range(10) if os.name == 'nt' else [0])
def test_completed_command_stops_redirected_helper(tmp_path, monkeypatch, returncode, fail_cleanup, attempt):
    script = """
import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')
print('failure output', flush=True)
sys.exit(int(sys.argv[2]))
"""
    pid_file = tmp_path / 'child'
    if fail_cleanup:
        def broken_cleanup(*args, **kwargs):
            raise subprocess.TimeoutExpired('cleanup', 1)

        monkeypatch.setattr(runner, '_stop', broken_cleanup)
    try:
        cmd = [sys.executable, '-c', script, str(pid_file), str(returncode)]
        if returncode:
            with pytest.raises(subprocess.CalledProcessError) as caught:
                runner.run_cli(cmd, capture_output=True, text=True, encoding='utf-8', check=True)
            result = caught.value
        else:
            result = runner.run_cli(cmd, capture_output=True, text=True, encoding='utf-8',
                                    check=True, timeout=10)
        assert result.returncode == returncode
        assert result.stdout == 'failure output\n'
        if fail_cleanup:
            assert any('cleanup' in note.lower() for note in result.__notes__)
        assert not alive(int(pid_file.read_text(encoding='utf-8')))
    finally:
        if pid_file.exists():
            pid = int(pid_file.read_text(encoding='utf-8'))
            if alive(pid):
                os.kill(pid, signal.SIGTERM if os.name == 'nt' else signal.SIGKILL)


@pytest.mark.skipif(os.name != 'posix', reason='Bash session requires POSIX')
def test_awe_task_cancellation_reaps_its_command(tmp_path):
    from explorers.awe_agent_explorer import LocalBashSession

    pid_file = tmp_path / 'command-pid'
    script = """
import os, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()), encoding='utf-8')
time.sleep(60)
"""
    command = shlex.join([sys.executable, '-c', script, str(pid_file)])

    async def cancel_command():
        task = asyncio.create_task(LocalBashSession(str(tmp_path)).execute(command))
        try:
            async with asyncio.timeout(10):
                while not pid_file.exists() or not pid_file.stat().st_size:
                    await asyncio.sleep(.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            pid = int(pid_file.read_text(encoding='utf-8'))
            assert not alive(pid), 'cancelled command is still running'
        finally:
            if pid_file.exists():
                pid = int(pid_file.read_text(encoding='utf-8'))
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)
            if not task.done():
                task.cancel()

    asyncio.run(cancel_command())
