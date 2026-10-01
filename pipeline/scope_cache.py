"""Bounded, process-local reuse of immutable topic membership tables."""
from collections import OrderedDict
from concurrent.futures import Future
import threading


class ScopeCache:
    def __init__(self, max_bytes=32 * 1024 * 1024, max_entries=128):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._pending = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def get_or_compute(self, key, compute):
        if self.max_bytes <= 0 or self.max_entries <= 0:
            return compute()
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached[0].clone()
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                pending = self._pending[key] = Future()
        if not owner:
            return pending.result().clone()
        try:
            frame = compute()
            size = frame.estimated_size()
            with self._lock:
                if size <= self.max_bytes:
                    while self._entries and (self._bytes + size > self.max_bytes
                                             or len(self._entries) >= self.max_entries):
                        self._bytes -= self._entries.popitem(last=False)[1][1]
                    self._entries[key] = (frame, size)
                    self._bytes += size
                pending.set_result(frame)
                del self._pending[key]
            return frame.clone()
        except BaseException as error:
            with self._lock:
                pending.set_exception(error)
                del self._pending[key]
            raise
