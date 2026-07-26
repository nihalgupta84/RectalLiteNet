# RectalLiteNet

RectalLiteNet is the code release for joint segmentation of normal rectum and
rectal tumor in contrast-enhanced CARE CT. The evaluated model uses a fixed
pelvic field of view, three adjacent axial slices, a ConvNeXt-Tiny encoder, and
a compact SCSE decoder.

## Results

The final protocol trains five independent models on all 318 CARE development
patients for 30 epochs and evaluates them once on 81 held-out patients. Values
below use the source-compatible global-pixel CARE profile.

| Profile | Normal Dice | Tumor Dice | Macro Dice | Macro IoU | Parameters | MACs |
|---|---:|---:|---:|---:|---:|---:|
| Single model, mean ± SD | 67.84 ± 0.27 | 76.45 ± 0.20 | 72.14 ± 0.21 | 56.60 ± 0.25 | 31.37M | 15.66G |
| Three-model ensemble | 68.57 | 76.92 | 72.75 | 57.33 | 94.10M | 46.97G |

The single model uses 69.7% fewer parameters than paper-reported automatic
U-SAM, 72.3% fewer than HHF-SAM, and 83.7% fewer than GLocalSeg. Published
comparisons are contextual because final comparator predictions were not
available under this evaluator.

## Installation

```bash
conda create -n rectallitenet python=3.11 -y
conda activate rectallitenet
pip install -r requirements.txt
```

The released checkpoints use Git LFS:

```bash
git lfs install
git lfs pull
```

## CARE Data

Obtain CARE from the release linked by the
[U-SAM paper](https://arxiv.org/abs/2308.08283). The source archive can be
downloaded with:

```bash
gdown 1X_JTfD8Ch-IxmG5VHtKk_xGZT336Fl1Q
```

After extraction, the expected directories are:

```text
DataV6/
├── train/train_npz/*.npz
├── train/train_bbox.csv
├── test/test_npz/*.npz
└── test/test_bbox.csv
```

Each NPZ must contain `image` and `label`. Labels are decoded as
`0=background`, `1=normal rectum`, and raw values `2/3=tumor`.

Create manifests:

```bash
python prepare_manifest.py \
  --input-dir /path/to/DataV6/train/train_npz \
  --relative-to /path/to/DataV6 \
  --output manifests/development.csv

python prepare_manifest.py \
  --input-dir /path/to/DataV6/test/test_npz \
  --relative-to /path/to/DataV6 \
  --output manifests/test.csv
```

CARE images are not redistributed by this repository.

## Training

The paper configuration is stored in `configs/care.json`. Train one final seed:

```bash
python train.py \
  --manifest manifests/development.csv \
  --data-root /path/to/DataV6 \
  --output checkpoints/rectallitenet_seed42.pt \
  --epochs 30 \
  --batch-size 24 \
  --workers 16 \
  --seed 42 \
  --pretrained \
  --amp
```

Repeat with seeds `3407`, `7301`, `1337`, and `2026`. W&B logging is optional:
add `--wandb --wandb-project rectallitenet` after authenticating through the
standard W&B environment or CLI.

## Evaluation

Single model with the paper's flip probability averaging:

```bash
python evaluate.py \
  --manifest manifests/test.csv \
  --data-root /path/to/DataV6 \
  --checkpoint checkpoints/rectallitenet_seed42.pt \
  --reference-index-csv /path/to/DataV6/test/test_bbox.csv \
  --output outputs/seed42.json \
  --tta flips \
  --amp
```

Predeclared three-model ensemble:

```bash
python evaluate.py \
  --manifest manifests/test.csv \
  --data-root /path/to/DataV6 \
  --checkpoint checkpoints/rectallitenet_seed42.pt \
  --checkpoint checkpoints/rectallitenet_seed3407.pt \
  --checkpoint checkpoints/rectallitenet_seed7301.pt \
  --reference-index-csv /path/to/DataV6/test/test_bbox.csv \
  --output outputs/ensemble3.json \
  --tta flips \
  --amp
```

The evaluator reports class Dice, IoU, precision, recall, macro overlap, and
per-patient overlap. Omitting `--reference-index-csv` evaluates every manifest
slice rather than the source paper's indexed comparison subset.

## Checkpoints

The five files in `checkpoints/` contain inference model weights, the evaluated
model configuration, seed, epoch, source checkpoint hash, and paper metrics.
`checkpoints/checksums.json` records SHA-256 hashes.

## License

The release code is MIT licensed. PyTorch, torchvision, timm, OpenCV, SciPy,
NumPy, tqdm, and W&B remain subject to their respective licenses. CARE remains
subject to the dataset authors' terms.
