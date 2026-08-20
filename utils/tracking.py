"""Resilient local and Weights & Biases experiment tracking."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Write every event locally and mirror it to W&B when available."""

    def __init__(
        self,
        local_path: str | Path,
        config: dict[str, Any],
        *,
        wandb_enabled: bool,
        wandb_project: str,
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        self.local_path = Path(local_path)
        if self.local_path.exists():
            raise FileExistsError(f"Tracking log already exists: {self.local_path}")
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self._run: Any | None = None
        self._run_id: str | None = None
        self._wandb_logging_enabled = False
        self._append({"event": "run_started", "config": config})

        if not wandb_enabled:
            self._append({"event": "wandb_disabled"})
            return

        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config=config,
                dir=str(self.local_path.parent),
            )
            if run is None:
                raise RuntimeError("wandb.init returned no run")
            self._run = run
            self._run_id = str(run.id)
            self._wandb_logging_enabled = True
        except Exception as error:
            self._record_wandb_failure("init", error)
            return

        self._append(
            {
                "event": "wandb_initialized",
                "run_id": self.run_id,
                "project": wandb_project,
            }
        )

    @property
    def run_id(self) -> str | None:
        """Return the active W&B run identifier, if initialization succeeded."""
        return self._run_id

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Persist metrics locally before attempting to send them to W&B."""
        self._append({"event": "metrics", "step": step, "metrics": metrics})
        if self._run is None or not self._wandb_logging_enabled:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception as error:
            self._record_wandb_failure("log", error)
            self._wandb_logging_enabled = False

    def finish(self, summary: dict[str, Any]) -> None:
        """Record the terminal status locally and close the W&B run."""
        self._append({"event": "run_finished", "summary": summary})
        if self._run is None:
            return
        try:
            self._run.summary.update(summary)
            self._run.finish()
        except Exception as error:
            self._record_wandb_failure("finish", error)
        finally:
            self._run = None
            self._wandb_logging_enabled = False

    def _append(self, event: dict[str, Any]) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.local_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _record_wandb_failure(self, operation: str, error: Exception) -> None:
        error_type = type(error).__name__
        self._append(
            {
                "event": "wandb_failure",
                "operation": operation,
                "error_type": error_type,
            }
        )
        print(
            f"W&B {operation} failed ({error_type}); continuing with local tracking.",
            flush=True,
        )
