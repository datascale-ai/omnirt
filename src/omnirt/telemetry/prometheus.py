"""Minimal Prometheus exposition for OmniRT."""

from __future__ import annotations

from collections import defaultdict
import threading
from typing import Dict, Iterable, Tuple


_DEFAULT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


def _labels_key(labels: Dict[str, str] | None) -> Tuple[Tuple[str, str], ...]:
    payload = labels or {}
    return tuple(sorted((str(key), str(value)) for key, value in payload.items()))


def _labels_text(labels_key: Tuple[Tuple[str, str], ...]) -> str:
    if not labels_key:
        return ""
    parts = [f'{key}="{value}"' for key, value in labels_key]
    return "{" + ",".join(parts) + "}"


class PrometheusMetrics:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = defaultdict(dict)
        self._gauges: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[Tuple[Tuple[str, str], ...], Dict[str, object]]] = defaultdict(dict)
        self.set_queue_depth(priority="default", depth=0)

    def observe_job(self, *, task: str, model: str, execution_mode: str, state: str) -> None:
        self._inc(
            "omnirt_jobs_total",
            {
                "task": task,
                "model": model,
                "execution_mode": execution_mode,
                "state": state,
            },
        )

    def observe_stage_duration(self, *, stage: str, model: str, seconds: float) -> None:
        self._observe_histogram(
            "omnirt_stage_duration_seconds",
            {"stage": stage, "model": model},
            max(float(seconds), 0.0),
            buckets=_DEFAULT_BUCKETS,
        )

    def observe_cache_hit(self, *, cache_type: str) -> None:
        self._inc("omnirt_cache_hits_total", {"cache_type": cache_type})

    def set_queue_depth(self, *, priority: str, depth: int) -> None:
        self._set_gauge("omnirt_queue_depth", {"priority": priority}, max(int(depth), 0))

    def set_vram_peak_bytes(self, *, device: str, bytes_value: float) -> None:
        self._set_gauge("omnirt_vram_peak_bytes", {"device": str(device)}, max(float(bytes_value), 0.0))

    def set_worker_inflight(self, *, worker_id: str, model: str, count: int) -> None:
        self._set_gauge(
            "omnirt_worker_inflight",
            {"worker_id": worker_id, "model": model},
            max(int(count), 0),
        )

    def set_worker_queue_depth(self, *, worker_id: str, model: str, depth: int) -> None:
        self._set_gauge(
            "omnirt_worker_queue_depth",
            {"worker_id": worker_id, "model": model},
            max(int(depth), 0),
        )

    def observe_worker_chunk_duration(self, *, worker_id: str, model: str, seconds: float) -> None:
        self._observe_histogram(
            "omnirt_worker_chunk_duration_seconds",
            {"worker_id": worker_id, "model": model},
            max(float(seconds), 0.0),
            buckets=_DEFAULT_BUCKETS,
        )

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            lines.extend(self._render_counter("omnirt_jobs_total", "counter"))
            lines.extend(self._render_histogram("omnirt_stage_duration_seconds", "histogram"))
            lines.extend(self._render_counter("omnirt_cache_hits_total", "counter"))
            lines.extend(self._render_gauge("omnirt_queue_depth", "gauge"))
            lines.extend(self._render_gauge("omnirt_vram_peak_bytes", "gauge"))
            lines.extend(self._render_gauge("omnirt_worker_inflight", "gauge"))
            lines.extend(self._render_gauge("omnirt_worker_queue_depth", "gauge"))
            lines.extend(self._render_histogram("omnirt_worker_chunk_duration_seconds", "histogram"))
        return "\n".join(lines) + "\n"

    def _inc(self, name: str, labels: Dict[str, str] | None = None, value: float = 1.0) -> None:
        key = _labels_key(labels)
        with self._lock:
            current = float(self._counters[name].get(key, 0.0))
            self._counters[name][key] = current + float(value)

    def _set_gauge(self, name: str, labels: Dict[str, str] | None, value: float) -> None:
        key = _labels_key(labels)
        with self._lock:
            self._gauges[name][key] = float(value)

    def _observe_histogram(
        self,
        name: str,
        labels: Dict[str, str] | None,
        value: float,
        *,
        buckets: Iterable[float],
    ) -> None:
        key = _labels_key(labels)
        with self._lock:
            bucket_edges = tuple(sorted(float(edge) for edge in buckets))
            record = self._histograms[name].get(key)
            if record is None:
                record = {
                    "buckets": bucket_edges,
                    "counts": {edge: 0.0 for edge in bucket_edges},
                    "count": 0.0,
                    "sum": 0.0,
                }
                self._histograms[name][key] = record
            for edge in bucket_edges:
                if value <= edge:
                    record["counts"][edge] += 1.0  # type: ignore[index]
            record["count"] += 1.0  # type: ignore[index]
            record["sum"] += float(value)  # type: ignore[index]

    def _render_counter(self, name: str, kind: str) -> list[str]:
        lines = [f"# TYPE {name} {kind}"]
        for labels, value in sorted(self._counters.get(name, {}).items()):
            lines.append(f"{name}{_labels_text(labels)} {value}")
        return lines

    def _render_gauge(self, name: str, kind: str) -> list[str]:
        lines = [f"# TYPE {name} {kind}"]
        for labels, value in sorted(self._gauges.get(name, {}).items()):
            lines.append(f"{name}{_labels_text(labels)} {value}")
        return lines

    def _render_histogram(self, name: str, kind: str) -> list[str]:
        lines = [f"# TYPE {name} {kind}"]
        for labels, record in sorted(self._histograms.get(name, {}).items()):
            counts = record["counts"]  # type: ignore[index]
            cumulative = 0.0
            for edge in record["buckets"]:  # type: ignore[index]
                cumulative = float(counts[edge])  # type: ignore[index]
                bucket_labels = dict(labels)
                bucket_labels["le"] = str(edge)
                lines.append(f"{name}_bucket{_labels_text(_labels_key(bucket_labels))} {cumulative}")
            inf_labels = dict(labels)
            inf_labels["le"] = "+Inf"
            lines.append(f"{name}_bucket{_labels_text(_labels_key(inf_labels))} {record['count']}")
            lines.append(f"{name}_count{_labels_text(labels)} {record['count']}")
            lines.append(f"{name}_sum{_labels_text(labels)} {record['sum']}")
        return lines
