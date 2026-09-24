"""Ctrl+C must abandon the queued backlog instead of draining it."""
import os
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
    signals = [signal.SIGINT]
    if os.name == 'posix':
        signals.extend((signal.SIGTERM, signal.SIGHUP))
    before = {sig: signal.getsignal(sig) for sig in signals}
    with _interruptible_pool(1):
        for sig in signals:
            assert signal.getsignal(sig) is not before[sig]
    assert {sig: signal.getsignal(sig) for sig in signals} == before


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


def test_sequential_cancellation_scope_restores_context_and_handler():
    before = signal.getsignal(signal.SIGINT)
    previous_event = eval_runner.cli_cancel_event.get()
    with pytest.raises(KeyboardInterrupt):
        with eval_runner._cli_cancellation() as cancel:
            assert eval_runner.cli_cancel_event.get() is cancel
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
    assert cancel.is_set()
    assert eval_runner.cli_cancel_event.get() is previous_event
    assert signal.getsignal(signal.SIGINT) is before
