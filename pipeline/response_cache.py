"""Small process-local cache for results from an immutable dataset snapshot."""
from collections import OrderedDict
from concurrent.futures import Future
import json
import threading
import time


class ResponseCache:
    def __init__(self, max_bytes=16 * 1024 * 1024, max_entries=256, ttl=300, clock=time.monotonic):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.ttl = ttl
        self.clock = clock
        self._entries = OrderedDict()
        self._pending = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def get_or_compute(self, key, compute):
        if self.max_bytes <= 0 or self.max_entries <= 0 or self.ttl <= 0:
            return compute()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                expires, payload = entry
                if expires > self.clock():
                    self._entries.move_to_end(key)
                    return json.loads(payload)
                self._bytes -= len(payload)
                del self._entries[key]
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                pending = self._pending[key] = Future()
        if not owner:
            return json.loads(pending.result())
        try:
            result = compute()
            payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
            cacheable = not isinstance(result, dict) or not result.get("error")
            with self._lock:
                if cacheable and len(payload) <= self.max_bytes:
                    # Reclaim expired entries as well as least recently used ones.
                    now = self.clock()
                    for stale in [k for k, (expiry, _) in self._entries.items() if expiry <= now]:
                        self._bytes -= len(self._entries.pop(stale)[1])
                    while self._entries and (self._bytes + len(payload) > self.max_bytes
                                             or len(self._entries) >= self.max_entries):
                        self._bytes -= len(self._entries.popitem(last=False)[1][1])
                    self._entries[key] = (now + self.ttl, payload)
                    self._bytes += len(payload)
                # Settle before removing the in-flight record so waiting callers
                # share this computation even if the result is too big to retain.
                pending.set_result(payload)
                del self._pending[key]
            # Each caller owns its result; API annotations must not alter a
            # later response or another concurrent user's data.
            return json.loads(payload)
        except BaseException as error:
            with self._lock:
                pending.set_exception(error)
                del self._pending[key]
            raise
