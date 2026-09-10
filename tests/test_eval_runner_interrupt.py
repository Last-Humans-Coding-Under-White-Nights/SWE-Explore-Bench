"""Ctrl+C must abandon the queued backlog instead of draining it."""
import signal
import time

import pytest

import eval_runner
from eval_runner import _interruptible_pool


def test_queued_work_is_cancelled_not_drained():
    with pytest.raises(KeyboardInterrupt):
        with _interruptible_pool(1) as pool:
            futures = [pool.submit(time.sleep, 0.5) for _ in range(10)]
            raise KeyboardInterrupt

    assert sum(f.cancelled() for f in futures) >= 9


def test_handler_is_restored_on_exit():
    before = signal.getsignal(signal.SIGINT)
    with _interruptible_pool(1):
        assert signal.getsignal(signal.SIGINT) is not before
    assert signal.getsignal(signal.SIGINT) is before


def test_second_interrupt_exits_immediately(monkeypatch):
    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(eval_runner.os, "_exit", fake_exit)
    with _interruptible_pool(1):
        handler = signal.getsignal(signal.SIGINT)
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
        with pytest.raises(SystemExit) as exc:
            handler(signal.SIGINT, None)

    assert exc.value.code == 130
