# Troubleshooting

## CUDA runs out of memory

Reduce `training.batch_size` in `configs/default.json`, or pass a smaller value:

```bash
python train.py <the other arguments> --batch-size 2
```

The same solution applies to `test.py` and `inference.py` by changing their
`--batch-size` option.

## A manifest path cannot be found

Manifest paths are interpreted relative to `--data-root`. For example, a row
containing `DataV6/train/train_npz/case1_slice1.npz` requires `--data-root` to
point to the directory that contains `DataV6`.

## A filename is rejected

CARE files must follow this pattern:

```text
case<number>_slice<number>.npz
```

The patient identifier is everything before `_slice`. This naming convention
is how adjacent slices and patient-disjoint splits are constructed.

## Pretrained encoder download fails

The first training run downloads the ConvNeXt-Tiny ImageNet weights used to
initialize the encoder. Check the internet connection, or start without those
weights by adding `--no-pretrained`.

## OpenCV uses too many CPU threads

The dataset loader already limits OpenCV to one thread per worker. If the
machine is still overloaded, reduce `--workers` or use `--workers 0`.
