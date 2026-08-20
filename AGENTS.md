# AGENTS.md

## Scope

These instructions apply to the entire repository. A more deeply nested
`AGENTS.md` may add stricter instructions for its subtree.

## Project purpose

RectalLiteNet is a small PyTorch project for 2.5D semantic segmentation of
axial CARE CT slices. The model predicts three classes for the center slice:

- `0`: background
- `1`: normal rectum
- `2`: rectal tumor

Treat this as medical-imaging research software. Preserve reproducibility,
patient separation, preprocessing compatibility, and honest metric naming.
Do not describe smoke checks or development-set results as clinical validation.

## Start here

Before changing code:

1. Run `git status --short` and preserve unrelated user changes.
2. Read `README.md`, the relevant entry point, and every module it calls.
3. Inspect `configs/default.json` before changing preprocessing, model, training,
   or inference behavior.
4. Check whether the requested work needs CARE data, a pretrained checkpoint,
   network access for timm weights, a GPU, or a long-running training job.
5. Do not start full training, evaluation over private data, or a checkpoint
   download unless the user requested it.

## Repository map

```text
models/rectal_lite_net.py   ConvNeXt encoder, decoder, SCSE, classifier
datasets/care_dataset.py    manifests, 3-slice loading, preprocessing, augmentation
utils/                      config, checkpoints, loss, metrics, inference, seeding
scripts/prepare_manifest.py build a slice manifest from CARE-style NPZ files
scripts/split_manifest.py   make a deterministic patient-disjoint train/val split
train.py                    training and validation entry point
test.py                     labeled evaluation and optional checkpoint ensemble
inference.py                directory inference and mask export
configs/default.json        canonical preprocessing and runtime defaults
checkpoints/                Git LFS pretrained weights plus checksums
data/                       user-provided CARE data; ignored by Git
logs/                       generated runs, metrics, and predictions; ignored by Git
corpus/                     generated/extracted literature artifacts, not runtime code
docs/TROUBLESHOOTING.md     common runtime failures
```

Unless the task is specifically about literature assets, avoid scanning or
rewriting `corpus/`; it is large, generated, and unrelated to the Python runtime.

## Environment and commands

This repository is being developed inside a container. Use the existing Conda
environment named `vision`; the required packages are already installed. Do not
create a new virtual environment or reinstall dependencies unless the user
explicitly requests an environment change.

```bash
conda activate vision
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

The container has access to a 40 GB NVIDIA A100 GPU slice. Confirm the actual
GPU and free memory with `nvidia-smi` before starting a GPU workload; do not
assume the full slice is idle merely because it is visible.

Secrets and API tokens are available in `/workspace/.secrets/api_keys.env`.
Treat that file as sensitive:

- never print, inspect, copy, commit, or include its contents in logs;
- never expose secret values in commands, diffs, progress notes, or handoffs;
- source it only inside a runtime script that genuinely requires a token;
- refer to environment-variable names, never their values;
- do not modify the secrets file unless the user explicitly requests it.

Experiment tracking is performed with Weights & Biases (W&B). Training scripts
must record the resolved configuration, seed, dataset/split identity, checkpoint
path, and key metrics in W&B. Keep a local log as the durable source for process
output even when W&B is enabled; a network or W&B failure must not erase the
run record.

The underlying Python CLI arguments are documented in `README.md`. Wrap project
workloads in task-specific Bash scripts rather than invoking those entry points
directly. Representative launcher shapes are:

```bash
bash scripts/<prepare-manifest-wrapper>.sh
bash scripts/<split-manifest-wrapper>.sh
nohup bash scripts/<training-script>.sh > logs/<run-id>/train.log 2>&1 &
bash scripts/<evaluation-script>.sh
bash scripts/<inference-script>.sh
```

Use `--workers 0`, a small batch, `--no-pretrained`, and one epoch for a bounded
training smoke check launched by its Bash wrapper. CPU is supported but full
training is expected to be slow. The first pretrained training run may access
the network for timm's ImageNet weights.

## Running project workloads

Use Bash scripts to run project workloads. Put reusable launchers under
`scripts/`, make all paths and run identifiers explicit, activate the `vision`
environment in the script, and fail on errors. Do not hide configuration in an
interactive shell history.

Never execute training interactively or by calling `python train.py` directly.
Training is allowed only when the user has explicitly requested it, and every
training run must be launched through a Bash script using `nohup`, with both
stdout and stderr written to a unique file under `logs/`. Record the background
PID without overwriting another run's PID file. A typical launch shape is:

```bash
mkdir -p logs/<run-id>
nohup bash scripts/<training-script>.sh > logs/<run-id>/train.log 2>&1 &
run_pid=$!
printf '%s\n' "$run_pid" > logs/<run-id>/train.pid
```

Before launching, verify the command, manifests, output directory, seed, GPU
availability, W&B configuration, and that the run ID is new. After launching,
confirm the PID is alive and inspect the log for startup errors. Do not claim a
background run succeeded merely because `nohup` returned a PID.

Logs are the authoritative local record of running scripts. Every long-running
script must have its own stable, descriptive log path. Monitor work by reading
or tailing that log and checking the recorded PID; do not repeatedly relaunch a
job when progress is merely slow. Preserve logs, PID files, W&B run identifiers,
resolved configurations, metrics, and failure traces.

## Progress recording

Maintaining project progress is mandatory. Every agent must record what it did,
not only summarize it in chat. Update the project's existing progress document
when one exists; otherwise create and maintain `PROJECT_PROGRESS.md` at the
repository root. Each entry should include the UTC date/time, objective, files
changed, commands or Bash scripts run, validation results, W&B run ID when
applicable, log and checkpoint paths, current status, failures, and concrete
next steps. Keep entries factual and append-oriented; never erase prior history
or represent an incomplete/background run as complete.

For active workloads, the progress record should point to the corresponding
file under `logs/` rather than duplicating its full output. Before handing off,
update the progress record and ensure another agent can determine what is
running, where its logs are, and how to verify its status.

## Non-destructive operations

Do not run `rm` or any command/API operation whose purpose is to delete, clean,
purge, or otherwise make project data difficult to recover. This includes
cleanup commands such as `git clean`, destructive Git resets, file replacement
through checkout, log truncation, and deletion through Python or shell
utilities. Do not add automatic cleanup traps to Bash scripts, and do not
manually overwrite an existing run directory or user artifact. Normal append
logging and the project's atomic `last.pt`/`history.json` refreshes are allowed
only inside a newly created, unique run directory.

When an obsolete artifact must be moved out of the way, stop and ask the user
for direction. Prefer a new versioned filename or output directory so existing
data, checkpoints, logs, manifests, and results remain untouched. This rule
also applies to temporary artifacts created by agents: place them in a unique
location and leave them intact unless the user explicitly authorizes deletion.

## Scientific and data invariants

- CARE filenames must match `case<number>_slice<number>.npz` so patient and
  adjacent-slice identities are unambiguous.
- Each NPZ requires a 2D `image`; labeled training/testing samples also require
  a `label`.
- Split by `patient_id`, never by individual slices. Train/validation patient
  overlap is leakage and must fail loudly.
- The loader forms channels from offsets `(-1, 0, +1)` and repeats the center
  slice when a neighbor is absent.
- The dataset currently always creates three channels. Changing
  `data.context_slices` requires coordinated changes to the dataset, model,
  smoke tests, and checkpoint metadata.
- `fixed_crop_box` is ordered `[x1, y1, x2, y2]`. The default is
  `[88, 54, 440, 406]`, followed by resize to `384 x 384` and normalization
  with mean `0.1364736` and standard deviation `0.23238614`.
- Label decoding clamps values below zero to background and values above two to
  tumor. Preserve integer class IDs and use nearest-neighbor interpolation for
  masks.
- Image geometry and label geometry must remain synchronized during training
  augmentation. Image interpolation may be smooth; label interpolation may not.
- Evaluation and validation restore crop probabilities to the original image
  grid before taking `argmax`. Pixels outside the crop are background.
- A DataLoader batch used by restoration must contain one original spatial
  shape; current entry points reject mixed shapes within a batch.
- Global pixel-count metrics and per-patient metrics answer different
  questions. Do not relabel one as the other. The foreground macro averages
  normal-rectum and tumor scores, not background.

## Checkpoint compatibility

Project checkpoints store `model_state`, `model_config`, `metadata`, and
training `history`. Keep new checkpoints loadable through
`utils.checkpoint.load_checkpoint` and load state dictionaries strictly.

Inference preprocessing is taken from checkpoint `model_config`, with legacy
defaults only when fields are absent. Ensembles must have identical image size,
crop, mean, and standard deviation. If model structure or stored metadata
changes, test both a new project checkpoint and the legacy plain-state-dict
path.

The committed `checkpoints/*.pt` files are Git LFS objects and are about 120 MB
each. Do not rewrite them casually. If an explicitly requested checkpoint
change is made, update `checkpoints/checksums.json`, verify byte sizes and
SHA-256 values, and confirm `git lfs ls-files`.

## Implementation conventions

- Follow the existing Python style: `from __future__ import annotations`,
  standard-library/third-party/local import groups, `pathlib.Path`, type hints,
  concise docstrings, and explicit validation errors.
- Keep reusable behavior in `datasets/`, `models/`, or `utils/`; keep entry
  points focused on argument parsing and orchestration.
- Export public helpers through the package `__init__.py` when callers should
  import them from `datasets`, `models`, or `utils`.
- Make configurable behavior explicit in `configs/default.json` and preserve
  command-line overrides. Update `README.md` when a user-facing command,
  default, output schema, or data requirement changes.
- Preserve deterministic sorting and seeded splitting. Use `seed_everything`
  for training randomness.
- Keep checkpoint writes atomic and create parent directories for generated
  outputs.
- Do not commit datasets, run logs, predictions, virtual environments,
  bytecode, or temporary artifacts. Respect `.gitignore` and `.gitattributes`.
- Run training and other long-running project commands through auditable Bash
  scripts; use `nohup` and per-run log files for training.
- Update the project progress record for every completed, failed, or active
  piece of agent work.
- Avoid broad formatting or generated-corpus churn in focused changes.

## Verification

There is currently no committed automated test suite or configured formatter.
Run the smallest checks that cover the change and report exactly what ran.

For every Python change, run:

```bash
python -m compileall -q train.py test.py inference.py models datasets utils scripts
python train.py --help >/dev/null
python test.py --help >/dev/null
python inference.py --help >/dev/null
```

Then add focused checks as appropriate:

- Model changes: instantiate with `pretrained=False`, run a small CPU tensor,
  and assert output shape `[batch, 3, height, width]` and finite logits/loss.
- Dataset or augmentation changes: use temporary synthetic CARE-style NPZ
  files; verify channel order, edge-slice fallback, shapes, dtypes, crop bounds,
  label IDs, and synchronized geometry.
- Manifest changes: verify deterministic ordering, required CSV columns,
  patient-disjoint output, invalid filenames, empty input, and repeatability for
  a fixed seed.
- Loss or metric changes: cover empty foreground, perfect overlap, false
  positives, false negatives, both foreground classes, and finite gradients.
- Inference changes: check `tta=none` and `tta=flips`, probability normalization,
  crop restoration, single and ensemble checkpoints, and output mask values.
- Checkpoint changes: round-trip a temporary checkpoint on CPU and confirm
  strict state loading plus stored preprocessing.

Use real CARE data only when it is available and the requested task warrants
it. Keep generated test artifacts in a unique location outside the repository
or under ignored `logs/`. Do not delete them without explicit authorization.

## Handoff

Before finishing, inspect `git diff --check`, `git status --short`, and the final
diff. Update the project progress record. Summarize changed files, validation
performed, active Bash processes and their log paths, W&B run IDs, and any
checks that could not run because data, checkpoints, GPU access, or network
access was unavailable.
