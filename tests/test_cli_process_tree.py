"""Real CLI descendants must stop before timeout/cancellation returns."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]

CHILD = """
import os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
mode = sys.argv[2]
def stop(*args):
    (root / 'child-term').touch()
    if mode == 'polite':
        sys.exit(0)
signal.signal(signal.SIGBREAK if os.name == 'nt' else signal.SIGTERM, stop)
if os.name == 'posix':
    signal.signal(signal.SIGINT, stop)
(root / 'child-pid.tmp').write_text(str(os.getpid()), encoding='utf-8')
(root / 'child-pid.tmp').replace(root / 'child-pid')
while True:
    time.sleep(0.05)
"""

CLI = """
import os, signal, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
mode = sys.argv[2]
child = subprocess.Popen(
    [sys.executable, str(root / 'child.py'), str(root), mode],
    stdout=subprocess.DEVNULL if mode == 'closed-pipes' else None,
    stderr=subprocess.DEVNULL if mode == 'closed-pipes' else None,
)
def stop(*args):
    print('polite stop', flush=True)
    (root / 'parent-term').touch()
    if mode == 'polite':
        child.wait(timeout=2)
    sys.exit(0)
signal.signal(signal.SIGBREAK if os.name == 'nt' else signal.SIGTERM, stop)
if os.name == 'posix':
    signal.signal(signal.SIGINT, stop)
while not (root / 'child-pid').exists():
    time.sleep(0.01)
print('partial output café', flush=True)
print('partial error', file=sys.stderr, flush=True)
(root / 'parent-pid.tmp').write_text(str(os.getpid()), encoding='utf-8')
(root / 'parent-pid.tmp').replace(root / 'parent-pid')
if mode == 'leader-exit':
    print('RELEVANT_FILES:', flush=True)
    sys.exit(0)
while True:
    time.sleep(0.05)
"""

DRIVER = """
import json, os, subprocess, sys, time
from pathlib import Path
from explorers.opencode import OpenCodeExplorer
root = Path(sys.argv[1])
mode, stop, explorer_name = sys.argv[2:]
class FakeExplorer(OpenCodeExplorer):
    session_usage_query = None
    def build_cmd(self):
        return [sys.executable, str(root / 'cli.py'), str(root), mode]
timeout = 1 if stop == 'timeout' else 60
if explorer_name == 'opencode':
    explorer = FakeExplorer(repo_root=root, timeout=timeout)
elif explorer_name != 'runner':
    from explorers.claude_code import ClaudeCodeExplorer
    from explorers.cursor_agent import CursorAgentExplorer
    os.environ['CURSOR_AGENT_BIN'] = 'fake-cursor'
    cls = ClaudeCodeExplorer if explorer_name == 'claude' else CursorAgentExplorer
    explorer = cls(repo_root=root, timeout=timeout)

def wait_for_fixture():
    deadline = time.monotonic() + 10
    while not (root / 'parent-pid').exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('fake CLI did not become ready')
        time.sleep(.01)

# Exclude fixture startup from the timeout under test. In production the
# timeout likewise starts after Popen and Windows job setup return.
real_popen = subprocess.Popen
def fake_cli(cmd, **kwargs):
    if cmd[0] in ('claude', 'fake-cursor'):
        cmd = [sys.executable, str(root / 'cli.py'), str(root), mode]
    process = real_popen(cmd, **kwargs)
    if stop == 'timeout' and os.name == 'posix':
        wait_for_fixture()
    return process
subprocess.Popen = fake_cli
if stop == 'timeout' and os.name == 'nt':
    from explorers._windows_job import WindowsJob
    resume = WindowsJob.assign_and_resume
    def resume_ready(job, process):
        resume(job, process)
        wait_for_fixture()
    WindowsJob.assign_and_resume = resume_ready

def run():
    if explorer_name == 'runner':
        from explorers._cli_process import run_cli
        return run_cli([sys.executable, str(root / 'cli.py'), str(root), mode],
                       timeout=timeout, capture_output=True, text=True, encoding='utf-8')
    return explorer.explore(instance_id='test', query='test')
try:
    if stop in ('worker', 'worker-cancel', 'double-interrupt', 'terminate', 'hangup'):
        from eval_runner import _interruptible_pool
        with _interruptible_pool(1) as pool:
            future = pool.submit(run)
            if stop == 'worker-cancel':
                while not (root / 'parent-pid').exists():
                    time.sleep(0.01)
                raise KeyboardInterrupt
            future.result()
    else:
        run()
except BaseException as exc:
    (root / 'result.json').write_text(json.dumps({
        'type': type(exc).__name__,
        'stdout': getattr(exc, 'stdout', None),
        'stderr': getattr(exc, 'stderr', None),
        'outcome': getattr(exc, 'outcome', None),
    }), encoding='utf-8')
else:
    (root / 'result.json').write_text(json.dumps({'type': 'Success'}), encoding='utf-8')
"""


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), "process did not stop or become ready before deadline"


def alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            assert ctypes.get_last_error() == 87  # PID no longer exists
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
        finally:
            kernel.CloseHandle(handle)
    # Orphans can briefly remain zombies until the OS reaps them; a zombie
    # holds no pipes and cannot run. Do not depend on init's reaping latency.
    result = subprocess.run(
        ['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True,
        text=True, encoding='utf-8',
    )
    return bool(result.stdout.strip()) and not result.stdout.lstrip().startswith('Z')


@pytest.mark.parametrize('stop', ['timeout', 'interrupt', 'worker', 'worker-cancel'])
@pytest.mark.parametrize('mode', ['polite', 'stubborn', 'closed-pipes', 'leader-exit'])
def test_explorer_stops_whole_tree(tmp_path, stop, mode):
    check_explorer_tree(tmp_path, stop, mode, 'opencode')


@pytest.mark.parametrize('explorer', ['claude', 'cursor'])
@pytest.mark.parametrize('stop', ['timeout', 'interrupt', 'worker-cancel'])
def test_standalone_explorer_stops_whole_tree(tmp_path, stop, explorer):
    check_explorer_tree(tmp_path, stop, 'stubborn', explorer)


def test_runner_keeps_partial_timeout_output(tmp_path):
    check_explorer_tree(tmp_path, 'timeout', 'stubborn', 'runner')


@pytest.mark.skipif(os.name != 'nt', reason='Windows job handle lifetime')
def test_runner_death_stops_whole_tree(tmp_path):
    check_explorer_tree(tmp_path, 'runner-kill', 'stubborn', 'opencode')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX runner signals')
@pytest.mark.parametrize('stop', ['double-interrupt', 'terminate', 'hangup'])
def test_runner_hard_exit_stops_whole_tree(tmp_path, stop):
    check_explorer_tree(tmp_path, stop, 'stubborn', 'opencode')


def check_explorer_tree(tmp_path, stop, mode, explorer):
    if os.name != 'posix' and stop in ('interrupt', 'worker'):
        pytest.skip('Sending SIGINT from a separate process requires POSIX')
    for name, source in [('child', CHILD), ('cli', CLI), ('driver', DRIVER)]:
        (tmp_path / f'{name}.py').write_text(source, encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING='utf-8')
    driver = subprocess.Popen(
        [sys.executable, str(tmp_path / 'driver.py'), str(tmp_path), mode, stop, explorer],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', start_new_session=os.name == 'posix',
    )
    try:
        wait_until(lambda: (tmp_path / 'parent-pid').exists())
        pids = [int((tmp_path / f'{name}-pid').read_text(encoding='utf-8'))
                for name in ('parent', 'child')]
        if stop in ('interrupt', 'worker'):
            driver.send_signal(signal.SIGINT)
        elif stop == 'double-interrupt':
            driver.send_signal(signal.SIGINT)
            wait_until(lambda: (tmp_path / 'child-term').exists())
            driver.send_signal(signal.SIGINT)
        elif stop in ('terminate', 'hangup'):
            driver.send_signal(signal.SIGTERM if stop == 'terminate' else signal.SIGHUP)
        elif stop == 'runner-kill':
            driver.kill()
        stdout, stderr = driver.communicate(timeout=15)
        hard_exit = stop in ('runner-kill', 'double-interrupt', 'terminate', 'hangup')
        if stop in ('double-interrupt', 'terminate', 'hangup'):
            expected_signal = {'double-interrupt': signal.SIGINT,
                               'terminate': signal.SIGTERM, 'hangup': signal.SIGHUP}[stop]
            assert driver.returncode == 128 + expected_signal, (stdout, stderr)
        elif not hard_exit:
            assert driver.returncode == 0, (stdout, stderr)
        wait_until(lambda: all(not alive(pid) for pid in pids))
        if hard_exit:
            return
        if os.name == 'posix':
            if mode != 'leader-exit':
                assert (tmp_path / 'parent-term').exists()
            assert (tmp_path / 'child-term').exists()
        result = json.loads((tmp_path / 'result.json').read_text(encoding='utf-8'))
        if os.name == 'nt' and mode == 'leader-exit' and stop == 'timeout':
            # File-backed output is complete when the leader exits. Remaining
            # helpers are cleaned up without changing its successful result.
            assert result['type'] == 'Success'
            return
        if stop == 'timeout':
            if explorer == 'opencode':
                assert result['outcome'] == 'timeout'
            elif explorer == 'runner':
                assert result['type'] == 'TimeoutExpired'
            else:
                assert result['type'] == 'RuntimeError'
        else:
            assert result['type'] == 'KeyboardInterrupt'
        if stop == 'interrupt' or explorer == 'runner':
            assert 'partial output café' in result['stdout']
            if os.name == 'posix' and mode != 'leader-exit':
                assert 'polite stop' in result['stdout']
            assert 'partial error' in result['stderr']
    finally:
        # Even a broken implementation must not leak our fixture processes.
        for name in ('child', 'parent'):
            path = tmp_path / f'{name}-pid'
            if path.exists():
                try:
                    pid = int(path.read_text(encoding='utf-8'))
                    if alive(pid):
                        os.kill(pid, signal.SIGTERM if os.name == 'nt' else signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if driver.poll() is None:
            driver.kill()
        driver.communicate(timeout=5)
