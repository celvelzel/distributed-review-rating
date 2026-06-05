"""Timing helpers and lightweight metrics persistence."""

from __future__ import annotations

import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional


class StageTimer:
    """Collect elapsed timings per stage and metric key."""

    def __init__(self) -> None:
        self._stages: Dict[str, Dict[str, float]] = {}

    def record(self, stage_name: str, metric_key: str, elapsed_sec: float) -> None:
        self._stages.setdefault(stage_name, {})[metric_key] = (
            self._stages.get(stage_name, {}).get(metric_key, 0.0) + float(elapsed_sec)
        )

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {stage: dict(metrics) for stage, metrics in self._stages.items()}

    @property
    def stages(self) -> Dict[str, Dict[str, float]]:
        return self._stages


def timed(stage_name: str, metric_key: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that measures elapsed wall-clock time in seconds.

    The wrapped function can accept a ``stage_timer`` or ``timer`` keyword
    argument with a :class:`StageTimer` instance to collect the measurement.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            stage_timer = kwargs.pop("stage_timer", None) or kwargs.pop("timer", None)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if isinstance(stage_timer, StageTimer):
                    stage_timer.record(stage_name, metric_key, elapsed)

        return wrapper

    return decorator


def _deep_merge(base: MutableMapping[str, Any], incoming: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in incoming.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_metrics(output_path: str) -> Dict[str, Any]:
    path = Path(output_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_metrics(output_path: str, data: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_metrics(output_path)
    merged = _deep_merge(existing if isinstance(existing, MutableMapping) else {}, data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return merged
