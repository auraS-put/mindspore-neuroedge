"""Huawei ModelArts monitoring integration.

Logs metrics to the ModelArts training console so experiments
running on Ascend hardware can be tracked in the Huawei Cloud UI.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from auras.monitoring.base_logger import BaseLogger


class ModelArtsLogger(BaseLogger):
    """Log metrics visible in the ModelArts training job UI.

    ModelArts reads structured print statements from stdout to
    display real-time charts in the console.
    """

    def __init__(self, cfg=None):
        self._cfg = cfg

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        # ModelArts parses lines like: [ModelArts-Metric]{"metric_name": value}
        import json
        payload = dict(metrics)
        if step is not None:
            payload["step"] = step
        print(f"[ModelArts-Metric]{json.dumps(payload)}")

    def log_config(self, config: Dict[str, Any]) -> None:
        import json
        print(f"[ModelArts-Config]{json.dumps(config, default=str)}")

    def finish(self) -> None:
        print("[ModelArts] Training finished.")
