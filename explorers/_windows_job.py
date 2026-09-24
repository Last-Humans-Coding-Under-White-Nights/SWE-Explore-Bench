"""Windows job ownership for a CLI tree, independent of its leader's lifetime.

Imported only on Windows. Start suspended, assign to the job, then resume so
even immediately spawned helpers belong to the job. See Microsoft's guidance:
https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
"""
import ctypes
from ctypes import wintypes
import time


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", ctypes.c_uint64 * 6),  # IO_COUNTERS: six ULONGLONGs
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _api(name, restype, *argtypes):
    fn = getattr(kernel32, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


_create = _api("CreateJobObjectW", wintypes.HANDLE, ctypes.c_void_p, wintypes.LPCWSTR)
_assign = _api("AssignProcessToJobObject", wintypes.BOOL, wintypes.HANDLE, wintypes.HANDLE)
_terminate = _api("TerminateJobObject", wintypes.BOOL, wintypes.HANDLE, wintypes.UINT)
_set_info = _api("SetInformationJobObject", wintypes.BOOL, wintypes.HANDLE,
                 ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
_query_info = _api("QueryInformationJobObject", wintypes.BOOL, wintypes.HANDLE,
                   ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p)
_close = _api("CloseHandle", wintypes.BOOL, wintypes.HANDLE)
_snapshot = _api("CreateToolhelp32Snapshot", wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD)
_first = _api("Thread32First", wintypes.BOOL, wintypes.HANDLE, ctypes.POINTER(_ThreadEntry))
_next = _api("Thread32Next", wintypes.BOOL, wintypes.HANDLE, ctypes.POINTER(_ThreadEntry))
_open_thread = _api("OpenThread", wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_resume = _api("ResumeThread", wintypes.DWORD, wintypes.HANDLE)
_open_process = _api("OpenProcess", wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_in_job = _api("IsProcessInJob", wintypes.BOOL, wintypes.HANDLE, wintypes.HANDLE,
               ctypes.POINTER(wintypes.BOOL))
_wait = _api("WaitForSingleObject", wintypes.DWORD, wintypes.HANDLE, wintypes.DWORD)


class WindowsJob:
    def __init__(self):
        self._processes = {}
        self.handle = _create(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not _set_info(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def has_active_processes(self):
        pids = self._process_ids()
        for pid in pids:
            if pid in self._processes:
                continue
            handle = _open_process(0x00101000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
            if not handle:
                error = ctypes.get_last_error()
                if error == 87:  # Process disappeared since enumeration.
                    continue
                raise ctypes.WinError(error)
            belongs = wintypes.BOOL()
            if not _in_job(handle, self.handle, ctypes.byref(belongs)):
                error = ctypes.WinError(ctypes.get_last_error())
                _close(handle)
                raise error
            if belongs.value:
                self._processes[pid] = handle
            else:
                _close(handle)  # The PID was reused before OpenProcess.
        # Empty job accounting can precede process exit and file release.
        for pid, handle in list(self._processes.items()):
            status = _wait(handle, 0)
            if status == 0:  # WAIT_OBJECT_0
                _close(self._processes.pop(pid))
            elif status != 258:  # WAIT_TIMEOUT
                raise ctypes.WinError(ctypes.get_last_error())
        return bool(pids or self._processes)

    def _process_ids(self):
        capacity = 16
        while True:
            class ProcessIds(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("count", wintypes.DWORD),
                            ("pids", ctypes.c_size_t * capacity)]

            info = ProcessIds()
            if _query_info(self.handle, 3, ctypes.byref(info), ctypes.sizeof(info), None):
                if info.count == info.assigned:
                    return list(info.pids[:info.count])
            elif ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            capacity = max(capacity * 2, info.assigned)

    def assign_and_resume(self, process):
        if not _assign(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Popen closes the initial thread handle. Enumerate the suspended
        # process's thread and reopen it with THREAD_SUSPEND_RESUME access.
        snapshot = _snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
        if not snapshot or snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = _ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            found = _first(snapshot, ctypes.byref(entry))
            while found:
                if entry.th32OwnerProcessID == process.pid:
                    thread = _open_thread(0x0002, False, entry.th32ThreadID)
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if _resume(thread) == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        _close(thread)
                    return
                found = _next(snapshot, ctypes.byref(entry))
            raise OSError("Could not find suspended CLI thread")
        finally:
            _close(snapshot)

    def terminate(self):
        try:
            self.has_active_processes()  # Retain handles before termination removes PIDs.
        finally:
            if not _terminate(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            # Retained handles remain usable even if job queries failed.
            deadline = time.monotonic() + 1.0
            for handle in self._processes.values():
                status = _wait(handle, max(0, int((deadline - time.monotonic()) * 1000)))
                if status not in (0, 258):
                    raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle is not None:
            handle, self.handle = self.handle, None
            _close(handle)
        for handle in self._processes.values():
            _close(handle)
        self._processes.clear()
