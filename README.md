# RectalLiteNet

RectalLiteNet is a compact 2.5D deep-learning project for segmenting the normal
rectum and rectal tumor in axial CT images. It predicts three pixel classes:

- `0`: background
- `1`: normal rectum
- `2`: rectal tumor

The code in this repository covers the complete workflow: preparing the CARE
dataset, making a patient-disjoint split, training, testing, and producing
segmentation masks. It contains only the implementation needed to run the
project.

## Project structure

```text
RectalLiteNet/
├── models/                 # Neural-network architecture
├── data/                   # Put downloaded data and manifests here
├── datasets/               # CARE loading and preprocessing code
├── configs/                # Reproducible project settings
├── scripts/                # Dataset preparation utilities
├── utils/                  # Loss, metrics, checkpoints, and inference helpers
├── logs/                   # Generated checkpoints, metrics, and predictions
├── docs/                   # Public troubleshooting notes
├── train.py                # Train a model
├── test.py                 # Evaluate labeled data
├── inference.py            # Predict masks for new data
├── README.md
└── requirements.txt
```

## 1. Create the environment

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/nihalgupta84/RectalLiteNet.git
cd RectalLiteNet

python -m venv .venv
source .venv/bin/activate          # Linux or macOS
# .venv\Scripts\activate           # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify that PyTorch can see the GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

CPU execution is supported, but training is much faster on a CUDA GPU.

## 2. Download the CARE dataset

CARE can be downloaded from Google Drive:

- [CARE dataset download](https://drive.google.com/file/d/1X_JTfD8Ch-IxmG5VHtKk_xGZT336Fl1Q/view)

You can download it in a browser, or use `gdown`:

```bash
pip install gdown
gdown 1X_JTfD8Ch-IxmG5VHtKk_xGZT336Fl1Q
```

Extract the archive inside `data/CARE`. The project expects this dataset tree:

```text
data/
└── CARE/
    └── DataV6/
        ├── train/
        │   ├── train_npz/
        │   │   ├── case10000001_slice0.npz
        │   │   ├── case10000001_slice1.npz
        │   │   └── ...
        │   └── train_bbox.csv
        └── test/
            ├── test_npz/
            │   ├── case20000001_slice0.npz
            │   ├── case20000001_slice1.npz
            │   └── ...
            └── test_bbox.csv
```

Each NPZ file must contain:

- `image`: one 2D CT slice, expected in the range `[0, 1]`
- `label`: one 2D segmentation mask for training or testing

The filename must be `case<number>_slice<number>.npz`. This allows the loader to
find adjacent slices automatically. If the previous or next slice does not
exist, the center slice is repeated.

CARE images are not included in this repository. Follow the dataset owner's
terms when downloading and using them.

## 3. Create a development manifest

A manifest is a small CSV that tells the code where every slice is located and
which patient it belongs to.

```bash
python scripts/prepare_manifest.py \
  --input-dir data/CARE/DataV6/train/train_npz \
  --relative-to data/CARE \
  --output data/manifests/development.csv
```

The output has this format:

```csv
path,patient_id,slice_id,height,width
DataV6/train/train_npz/case10000001_slice0.npz,case10000001,0,512,512
```

## 4. Make training and validation splits

Always split by patient, not by image. Splitting individual images can place
slices from one patient in both sets and produce misleading results.

```bash
python scripts/split_manifest.py \
  --manifest data/manifests/development.csv \
  --train-output data/manifests/train.csv \
  --val-output data/manifests/val.csv \
  --val-fraction 0.125 \
  --seed 2026
```

Running this command again with the same seed produces the same split.

Create the separate test manifest without mixing it into training:

```bash
python scripts/prepare_manifest.py \
  --input-dir data/CARE/DataV6/test/test_npz \
  --relative-to data/CARE \
  --output data/manifests/test.csv
```

## 5. Understand the preprocessing

For every center slice, `datasets/care_dataset.py` performs these steps:

1. Loads the previous, center, and next axial slices as three channels.
2. Clips every image to `[0, 1]` and applies CLAHE contrast enhancement.
3. Applies the fixed crop `[x1=88, y1=54, x2=440, y2=406]`.
4. Uses synchronized rotation and flip augmentation during training only.
5. Resizes the crop to `384 × 384`.
6. Normalizes it with mean `0.1364736` and standard deviation `0.23238614`.
7. Converts all label values above `2` to tumor class `2`.

These values are stored in `configs/default.json`. Keep them unchanged when
using a provided checkpoint because preprocessing must match training.

## 6. Train RectalLiteNet

Start one training run:

```bash
python train.py \
  --config configs/default.json \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --data-root data/CARE \
  --output-dir logs/seed42 \
  --seed 42
```

The first run may download ImageNet initialization weights for ConvNeXt-Tiny.
Training produces:

```text
logs/seed42/
├── best.pt       # Highest validation macro Dice
├── last.pt       # Most recent epoch
└── history.json  # Loss and validation metrics for every epoch
```

Useful command-line overrides:

```bash
# Quick code check with one epoch and a small batch
python train.py <the required paths above> --epochs 1 --batch-size 2 --workers 0

# Train without downloading encoder initialization
python train.py <the required paths above> --no-pretrained

# Disable mixed precision
python train.py <the required paths above> --no-amp
```

The default loss is `0.4 × cross-entropy + 0.6 × foreground Dice`. A physical
batch of 8 is accumulated for 3 steps, giving an effective batch of 24. AdamW
uses linear learning-rate warmup followed by cosine decay. All settings are
visible in `configs/default.json`.

## 7. Test a trained checkpoint

Evaluate one model on labeled test data:

```bash
python test.py \
  --manifest data/manifests/test.csv \
  --data-root data/CARE \
  --checkpoint logs/seed42/best.pt \
  --output logs/seed42/test_results.json \
  --tta flips \
  --amp
```

The JSON output contains Dice, IoU, precision, and recall for normal rectum and
tumor, plus per-patient metrics.

To evaluate an ensemble, repeat `--checkpoint`:

```bash
python test.py \
  --manifest data/manifests/test.csv \
  --data-root data/CARE \
  --checkpoint logs/seed42/best.pt \
  --checkpoint logs/seed3407/best.pt \
  --checkpoint logs/seed7301/best.pt \
  --output logs/ensemble/test_results.json \
  --tta flips \
  --amp
```

All ensemble checkpoints must have identical preprocessing settings.

## 8. Run inference

`inference.py` accepts a directory of labeled or unlabeled CARE-style NPZ
files. Adjacent files for the same patient should be in the same directory.

```bash
python inference.py \
  --input-dir data/CARE/DataV6/test/test_npz \
  --checkpoint logs/seed42/best.pt \
  --output-dir logs/predictions \
  --tta flips \
  --amp
```

For every input slice, it saves:

- `<sample_id>.npy`: exact class IDs
- `<sample_id>.png`: grayscale class mask
- `<sample_id>_color.png`: green normal rectum and red tumor preview
- `summary.json`: settings and all generated paths

## Pretrained checkpoints

The checkpoint download URL will be added here after the public upload is
complete:

| File | Download |
|---|---|
| RectalLiteNet pretrained checkpoint | **URL to be updated** |

After downloading a checkpoint, use its local path with `--checkpoint`. The
loader also remains compatible with the earlier RectalLiteNet release
checkpoint format.

## Model summary

The model uses a ConvNeXt-Tiny feature encoder and a lightweight U-Net-style
decoder. Each decoder stage upsamples the features, joins the matching encoder
skip connection, applies convolutional refinement, and applies SCSE attention.
The final layer directly predicts the three segmentation classes for the center
slice.

The implementation is in `models/rectal_lite_net.py` and is deliberately kept
in one file so beginners can follow the full forward pass.

## Common problems

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for memory, path,
filename, worker, and pretrained-weight issues.

## License

The project code is provided under the MIT License. The CARE dataset and Python
dependencies retain their own licenses and usage terms.
