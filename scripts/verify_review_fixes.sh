#!/usr/bin/env bash
set -euo pipefail

rectal_conda_base="$(conda info --base)"
source "$rectal_conda_base/etc/profile.d/conda.sh"
conda activate vision

python -m compileall -q train.py test.py inference.py models datasets utils scripts
python train.py --help >/dev/null
python test.py --help >/dev/null
python inference.py --help >/dev/null

mkdir -p logs
RECTAL_VALIDATION_DIR="logs/review-fix-validation-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export RECTAL_VALIDATION_DIR
mkdir "$RECTAL_VALIDATION_DIR"

python - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from utils import ExperimentTracker, load_checkpoint


root = Path(os.environ["RECTAL_VALIDATION_DIR"])


class FakeRun:
    def __init__(self) -> None:
        self.id = "fake-run-id"
        self.logged: list[tuple[dict[str, float], int]] = []
        self.summary: dict[str, object] = {}
        self.finished = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logged.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


fake_run = FakeRun()
fake_wandb = SimpleNamespace(init=lambda **_: fake_run)
with patch.dict(sys.modules, {"wandb": fake_wandb}):
    tracker = ExperimentTracker(
        root / "tracking-success.jsonl",
        {"seed": 42, "splits": {"train": "sha256"}},
        wandb_enabled=True,
        wandb_project="rectallitenet",
    )
    assert tracker.run_id == "fake-run-id"
    tracker.log({"train/loss": 0.25}, step=1)
    tracker.finish({"status": "completed", "best_checkpoint": "best.pt"})
    assert tracker.run_id == "fake-run-id"
assert fake_run.logged == [({"train/loss": 0.25}, 1)]
assert fake_run.summary["best_checkpoint"] == "best.pt"
assert fake_run.finished


def fail_init(**_: object) -> object:
    raise RuntimeError("simulated offline mode")


with patch.dict(sys.modules, {"wandb": SimpleNamespace(init=fail_init)}):
    fallback = ExperimentTracker(
        root / "tracking-fallback.jsonl",
        {"seed": 42},
        wandb_enabled=True,
        wandb_project="rectallitenet",
    )
    fallback.log({"train/loss": 0.5}, step=1)
    fallback.finish({"status": "completed"})
fallback_events = [
    json.loads(line)
    for line in (root / "tracking-fallback.jsonl").read_text().splitlines()
]
assert [event["event"] for event in fallback_events] == [
    "run_started",
    "wandb_failure",
    "metrics",
    "run_finished",
]

bad_checkpoint = root / "unsupported-context.pt"
torch.save(
    {"model_state": {}, "model_config": {"context_slices": 5}},
    bad_checkpoint,
)
try:
    load_checkpoint(bad_checkpoint)
except ValueError as error:
    assert "require context_slices=3" in str(error)
else:
    raise AssertionError("Unsupported checkpoint context was accepted")

bad_config = json.loads(Path("configs/default.json").read_text())
bad_config["data"]["context_slices"] = 5
bad_config_path = root / "bad-context.json"
bad_config_path.write_text(json.dumps(bad_config))
bad_output = root / "must-not-be-created"
bad_context = subprocess.run(
    [
        sys.executable,
        "train.py",
        "--config",
        str(bad_config_path),
        "--train-manifest",
        str(root / "missing.csv"),
        "--data-root",
        str(root),
        "--output-dir",
        str(bad_output),
        "--no-wandb",
    ],
    capture_output=True,
    text=True,
)
assert bad_context.returncode != 0
assert "data.context_slices must be 3" in bad_context.stderr
assert not bad_output.exists()

valid_manifest = root / "valid-manifest.csv"
valid_manifest.write_text(
    "path,patient_id,slice_id\n"
    "case1_slice0.npz,case1,0\n"
)
existing_training = root / "existing-training"
existing_training.mkdir()
(existing_training / "sentinel.txt").write_text("preserve me\n")
training_collision = subprocess.run(
    [
        sys.executable,
        "train.py",
        "--train-manifest",
        str(valid_manifest),
        "--data-root",
        str(root),
        "--output-dir",
        str(existing_training),
        "--no-pretrained",
        "--no-wandb",
    ],
    capture_output=True,
    text=True,
)
assert training_collision.returncode != 0
assert "Output directory already exists" in training_collision.stderr
assert (existing_training / "sentinel.txt").read_text() == "preserve me\n"

existing_result = root / "existing-result.json"
existing_result.write_text("preserve me\n")
test_collision = subprocess.run(
    [
        sys.executable,
        "test.py",
        "--manifest",
        str(root / "missing.csv"),
        "--data-root",
        str(root),
        "--checkpoint",
        str(root / "missing.pt"),
        "--output",
        str(existing_result),
    ],
    capture_output=True,
    text=True,
)
assert test_collision.returncode != 0
assert "Output file already exists" in test_collision.stderr
assert existing_result.read_text() == "preserve me\n"

existing_predictions = root / "existing-predictions"
existing_predictions.mkdir()
inference_collision = subprocess.run(
    [
        sys.executable,
        "inference.py",
        "--input-dir",
        str(root),
        "--checkpoint",
        str(root / "missing.pt"),
        "--output-dir",
        str(existing_predictions),
    ],
    capture_output=True,
    text=True,
)
assert inference_collision.returncode != 0
assert "Output directory already exists" in inference_collision.stderr

(root / "validation-success.json").write_text(
    json.dumps(
        {
            "status": "passed",
            "checks": [
                "compile_and_cli_help",
                "wandb_success_and_fallback",
                "unsupported_context",
                "training_output_collision",
                "evaluation_output_collision",
                "inference_output_collision",
            ],
        },
        indent=2,
    )
    + "\n"
)
print(f"Focused review-fix checks passed; artifacts: {root}")
PY
