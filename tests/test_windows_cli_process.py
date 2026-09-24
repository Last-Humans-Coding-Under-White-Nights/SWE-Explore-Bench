"""Native Windows coverage for job ownership and file-backed output capture."""
import os
import subprocess
import sys

import pytest

from explorers import _cli_process as runner


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows job APIs')


def test_job_tracks_active_processes():
    from explorers._windows_job import WindowsJob

    job = WindowsJob()
    process = None
    try:
        assert not job.has_active_processes()
        process = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(60)'],
            creationflags=0x00000004,  # CREATE_SUSPENDED
        )
        job.assign_and_resume(process)
        assert job.has_active_processes()
        job.terminate()
        process.wait(timeout=5)
        assert not job.has_active_processes()
    finally:
        job.close()
        if process is not None:
            process.wait(timeout=5)


def test_empty_job_list_still_waits_for_process_handle(monkeypatch):
    import ctypes
    from explorers import _windows_job
    from test_cli_process_tree import wait_until

    job = _windows_job.WindowsJob()
    process = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'], creationflags=0x00000004,
    )
    try:
        job.assign_and_resume(process)
        assert job.has_active_processes()

        def empty_job(handle, info_class, buffer, size, returned):
            ctypes.memset(buffer, 0, size)
            return True

        monkeypatch.setattr(_windows_job, '_query_info', empty_job)
        assert job.has_active_processes(), 'empty accounting does not mean process exit'
        job.terminate()
        wait_until(lambda: not job.has_active_processes())
        assert process.poll() is not None
    finally:
        job.close()
        process.kill()
        process.wait(timeout=5)


def test_termination_survives_handle_discovery_failure(monkeypatch):
    from explorers._windows_job import WindowsJob

    job = WindowsJob()
    process = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'], creationflags=0x00000004,
    )
    try:
        job.assign_and_resume(process)

        def broken_query():
            raise OSError('job query failed')

        monkeypatch.setattr(job, 'has_active_processes', broken_query)
        with pytest.raises(OSError, match='job query failed'):
            job.terminate()
        assert process.wait(timeout=5) is not None
    finally:
        job.close()
        process.kill()
        process.wait(timeout=5)


def test_cleanup_failure_keeps_partial_output(monkeypatch):
    def broken_cleanup(*args, **kwargs):
        raise subprocess.TimeoutExpired('cleanup', 1)

    monkeypatch.setattr(runner, '_stop', broken_cleanup)
    # Let the process finish normally, then trigger check=True cleanup. This
    # makes output readiness deterministic while exercising the same fallback
    # that handles failed timeout/cancellation cleanup.
    with pytest.raises(subprocess.CalledProcessError) as caught:
        runner.run_cli(
            [sys.executable, '-c', "import os; os.write(1, b'paid output'); os.write(2, b'detail'); exit(3)"],
            capture_output=True, text=True, encoding='utf-8', check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    assert caught.value.stdout == 'paid output'
    assert caught.value.stderr == 'detail'
    assert any('cleanup' in note for note in caught.value.__notes__)


def test_assignment_failure_reaps_suspended_child(monkeypatch):
    from explorers import _windows_job

    processes = []
    popen = subprocess.Popen

    def record(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, 'Popen', record)
    monkeypatch.setattr(_windows_job, '_assign', lambda *args: False)
    try:
        with pytest.raises(OSError):
            runner.run_cli([sys.executable, '-c', 'import time; time.sleep(60)'])
        assert processes[0].poll() is not None
    finally:
        for process in processes:
            process.kill()
            process.wait(timeout=5)


def test_job_can_be_closed_twice_without_closing_another_job():
    from explorers._windows_job import WindowsJob

    job = WindowsJob()
    job.close()
    other = WindowsJob()
    try:
        job.close()
        assert not other.has_active_processes()
        assert job.handle is None
    finally:
        other.close()


def test_main_thread_interrupt_does_not_wait_for_command_timeout(tmp_path):
    import _thread
    from threading import Thread
    import time

    from eval_runner import _cli_cancellation
    from test_cli_process_tree import wait_until

    ready = tmp_path / 'ready'
    interrupted_at = []

    def interrupt_when_ready():
        wait_until(ready.exists)
        interrupted_at.append(time.monotonic())
        _thread.interrupt_main()

    thread = Thread(target=interrupt_when_ready, daemon=True)
    with _cli_cancellation():
        thread.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                runner.run_cli([
                    sys.executable, '-c',
                    'import pathlib, sys, time; pathlib.Path(sys.argv[1]).touch(); time.sleep(60)',
                    str(ready),
                ])
            assert time.monotonic() - interrupted_at[0] < 5
        finally:
            thread.join(timeout=10)
