"""
utils/metrics.py
----------------
Lightweight in-process request metrics for the health endpoint.

No external dependencies — uses a simple thread-safe rolling counter.
Metrics reset on process restart (not persisted to disk).

Public API
----------
record_request(endpoint, elapsed_s, success) — record one request
get_metrics_summary()                         — return dict of stats
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _EndpointStats:
    total:    int   = 0
    success:  int   = 0
    failed:   int   = 0
    total_elapsed_s: float = 0.0

    @property
    def avg_latency_s(self) -> float:
        return round(self.total_elapsed_s / self.total, 3) if self.total else 0.0

    @property
    def success_rate(self) -> float:
        return round(self.success / self.total, 3) if self.total else 0.0


_stats: dict[str, _EndpointStats] = defaultdict(_EndpointStats)
_lock  = threading.Lock()


def record_request(endpoint: str, elapsed_s: float, success: bool) -> None:
    """
    Record one completed request.

    Parameters
    ----------
    endpoint:  One of "score", "rank", "explain".
    elapsed_s: Wall-clock seconds for the request.
    success:   True if the request completed without a 5xx error.
    """
    with _lock:
        s = _stats[endpoint]
        s.total    += 1
        s.success  += int(success)
        s.failed   += int(not success)
        s.total_elapsed_s += elapsed_s


def get_metrics_summary() -> dict:
    """
    Return a snapshot of all endpoint stats.

    Returns
    -------
    dict  keyed by endpoint name, values contain total/success/failed/
          avg_latency_s/success_rate.
    """
    with _lock:
        return {
            endpoint: {
                "total":         s.total,
                "success":       s.success,
                "failed":        s.failed,
                "avg_latency_s": s.avg_latency_s,
                "success_rate":  s.success_rate,
            }
            for endpoint, s in _stats.items()
        }
