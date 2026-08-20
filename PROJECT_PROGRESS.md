# Project Progress

This append-oriented record tracks work performed by coding agents. Runtime
details for long-running jobs belong in their corresponding files under
`logs/`; progress entries should link to those files without copying secrets or
large command output.

## 2026-08-19 09:47:55 UTC — Repository agent guidance

- Objective: inspect RectalLiteNet and establish repository-specific operating
  instructions for future agents.
- Files changed: `AGENTS.md`, `PROJECT_PROGRESS.md`.
- Work completed: documented the project layout, scientific/data invariants,
  checkpoint compatibility, coding conventions, verification expectations,
  container and Conda environment, A100 GPU slice, safe secret handling, W&B
  tracking, Bash/`nohup` training launches, log-based process tracking,
  mandatory progress recording, and non-destructive operation rules.
- Validation: Python compilation and CLI help checks passed; synthetic CPU model
  and CARE dataset smoke checks passed; the seed-42 checkpoint loaded and ran;
  all committed checkpoint SHA-256 hashes matched `checkpoints/checksums.json`;
  `git diff --check` passed. The `vision` Conda environment, NVIDIA
  A100-SXM4-40GB device, and secrets-file path were confirmed without reading
  secret contents.
- Active processes: none launched.
- W&B run: none.
- Logs/checkpoints created: none.
- Status: completed.
- Next step: use these instructions for subsequent implementation and runtime
  work.

## 2026-08-20 19:09:55 UTC — Review findings corrected

- Objective: resolve the three review findings covering W&B experiment
  tracking, unsupported context-slice configuration, and overwrite-prone output
  defaults.
- Files changed: `train.py`, `test.py`, `inference.py`,
  `configs/default.json`, `utils/tracking.py`, `utils/checkpoint.py`,
  `utils/__init__.py`, `requirements.txt`, `README.md`,
  `scripts/verify_review_fixes.sh`, and this progress record.
- Work completed: restored default-on W&B tracking with CLI/config overrides;
  added a durable local `events.jsonl` fallback and resolved configuration with
  manifest hashes, split cardinalities, checkpoint paths, seed, and metrics;
  rejected non-three-slice training configs and checkpoints; required explicit
  new output paths and collision-safe writes for training, evaluation, and
  inference; updated dependency and user documentation.
- Commands/scripts run: `bash scripts/verify_review_fixes.sh` in the `vision`
  Conda environment. The wrapper ran Python compilation, all three CLI help
  checks, and focused Python/subprocess assertions. `git diff --check`, final
  status, diff, and process checks were also inspected.
- Validation: passed. Covered successful W&B mirroring with a fake run,
  simulated W&B initialization failure with continued local metrics, stable
  run IDs, unsupported training/checkpoint context rejection, and preservation
  of existing training/evaluation/inference targets. Evidence is in
  `logs/review-fix-validation-20260820T190908Z-896094/validation-success.json`.
- Active processes: none.
- W&B run: no external run created; focused checks used fake success/failure
  clients and did not access the network.
- Logs/checkpoints created: ignored validation artifacts under
  `logs/review-fix-validation-20260820T190908Z-896094/`; no model checkpoint or
  CARE output was created.
- Status: completed.
- Failures/limitations: no full training, private CARE evaluation, GPU workload,
  pretrained-weight download, or live W&B network call was requested or run.
- Next step: review and commit the focused changes when ready.

## 2026-08-20 19:47:54 UTC — Review fixes published

- Objective: commit and synchronize the completed review fixes with GitHub.
- Files changed: `PROJECT_PROGRESS.md` for this publishing record; the
  implementation scope is recorded in the preceding entry.
- Commands run: created branch `codex/fix-review-reproducibility`, staged the
  12 explicit project files, ran `git diff --cached --check`, committed as
  `5c42d49` (`Fix experiment reproducibility safeguards`), and pushed the
  branch to `origin` with upstream tracking.
- Validation: the staged whitespace check passed; the implementation validation
  remains recorded in
  `logs/review-fix-validation-20260820T190908Z-896094/validation-success.json`.
- Active processes: none.
- W&B run: none.
- Logs/checkpoints created: none during publishing.
- Status: completed; the implementation commit is available on
  `origin/codex/fix-review-reproducibility`.
- Failures: none.
- Next step: open a pull request if review/merge through GitHub is desired.
