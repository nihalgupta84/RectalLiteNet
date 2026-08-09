"""CARE dataset loading and preprocessing.

CARE samples are NPZ files named ``case<id>_slice<index>.npz``. Each file must
contain an ``image`` array. Training and testing files must also contain a
``label`` array.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)

DEFAULT_MEAN = 0.1364736
DEFAULT_STD = 0.23238614
DEFAULT_CROP = (88, 54, 440, 406)
SAMPLE_PATTERN = re.compile(r"^(?P<patient>case\d+)_slice(?P<slice>\d+)$")


@dataclass(frozen=True)
class Record:
    """Location and identity of one axial slice."""

    path: Path
    patient_id: str
    slice_id: int


def _record_from_path(path: Path) -> Record:
    match = SAMPLE_PATTERN.match(path.stem)
    if match is None:
        raise ValueError(
            f"Unexpected filename {path.name!r}. Expected case<number>_slice<number>.npz"
        )
    return Record(
        path=path,
        patient_id=match.group("patient"),
        slice_id=int(match.group("slice")),
    )


def records_from_directory(directory: str | Path) -> list[Record]:
    """Create records from all NPZ files in a directory."""
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ files found in {directory}")
    return [_record_from_path(path) for path in paths]


def records_from_manifest(
    manifest: str | Path,
    data_root: str | Path,
) -> list[Record]:
    """Load records from a CSV produced by ``scripts/prepare_manifest.py``."""
    root = Path(data_root)
    records: list[Record] = []
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"path", "patient_id", "slice_id"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Manifest {manifest} is missing columns: {sorted(missing_columns)}"
            )
        for row in reader:
            path = Path(row["path"])
            if not path.is_absolute():
                path = root / path
            records.append(
                Record(
                    path=path,
                    patient_id=row["patient_id"],
                    slice_id=int(row["slice_id"]),
                )
            )
    if not records:
        raise ValueError(f"No records found in {manifest}")
    return records


def decode_label(label: np.ndarray) -> np.ndarray:
    """Map CARE labels to 0=background, 1=normal rectum, 2=tumor."""
    decoded = np.asarray(label).astype(np.int64, copy=True)
    decoded[decoded < 0] = 0
    decoded[decoded > 2] = 2
    return decoded


class CAREDataset(Dataset):
    """Load synchronized three-slice images and center-slice labels."""

    def __init__(
        self,
        records: Sequence[Record],
        image_size: int = 384,
        training: bool = False,
        require_label: bool = True,
        return_original_label: bool = False,
        crop_box: Sequence[int] = DEFAULT_CROP,
        mean: float = DEFAULT_MEAN,
        std: float = DEFAULT_STD,
    ) -> None:
        if not records:
            raise ValueError("CAREDataset requires at least one record")
        if len(crop_box) != 4:
            raise ValueError("crop_box must contain [x1, y1, x2, y2]")
        if std <= 0:
            raise ValueError("std must be positive")

        self.records = list(records)
        self.image_size = int(image_size)
        self.training = training
        self.require_label = require_label
        self.return_original_label = return_original_label
        self.crop_box = tuple(int(value) for value in crop_box)
        self.mean = float(mean)
        self.std = float(std)
        self.by_key = {
            (record.patient_id, record.slice_id): record for record in self.records
        }
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, record: Record) -> np.ndarray:
        if not record.path.is_file():
            raise FileNotFoundError(record.path)
        with np.load(record.path) as sample:
            if "image" not in sample:
                raise ValueError(f"{record.path} has no 'image' array")
            image = np.squeeze(sample["image"]).astype(np.float32)
        if image.ndim != 2:
            raise ValueError(f"Expected a 2D image in {record.path}, got {image.shape}")
        image = np.clip(image, 0.0, 1.0)
        image_8bit = (image * 255).astype(np.uint8)
        return self.clahe.apply(image_8bit).astype(np.float32) / 255.0

    def _load_label(self, record: Record) -> np.ndarray:
        with np.load(record.path) as sample:
            if "label" not in sample:
                raise ValueError(f"{record.path} has no 'label' array")
            label = decode_label(np.squeeze(sample["label"]))
        if label.ndim != 2:
            raise ValueError(f"Expected a 2D label in {record.path}, got {label.shape}")
        return label

    def _augment(
        self,
        image_stack: np.ndarray,
        label: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the same random geometry to images and label."""
        if random.random() < 0.5:
            turns = random.randint(0, 3)
            image_stack = np.rot90(image_stack, turns, axes=(-2, -1)).copy()
            label = np.rot90(label, turns).copy()

        if random.random() < 0.5:
            image_axis = random.choice((-2, -1))
            label_axis = 0 if image_axis == -2 else 1
            image_stack = np.flip(image_stack, axis=image_axis).copy()
            label = np.flip(label, axis=label_axis).copy()

        if random.random() < 0.5:
            angle = random.uniform(-20.0, 20.0)
            height, width = label.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            image_stack = np.stack(
                [
                    cv2.warpAffine(
                        channel,
                        matrix,
                        (width, height),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                    for channel in image_stack
                ]
            )
            label = cv2.warpAffine(
                label.astype(np.int16),
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            ).astype(np.int64)
        return image_stack, label

    def __getitem__(self, index: int) -> dict[str, object]:
        center_record = self.records[index]
        neighbor_records = [
            self.by_key.get(
                (center_record.patient_id, center_record.slice_id + offset),
                center_record,
            )
            for offset in (-1, 0, 1)
        ]
        image_stack = np.stack(
            [self._load_image(record) for record in neighbor_records]
        )
        original_height, original_width = image_stack.shape[-2:]

        x1, y1, x2, y2 = self.crop_box
        if not (0 <= x1 < x2 <= original_width and 0 <= y1 < y2 <= original_height):
            raise ValueError(
                f"Crop {self.crop_box} is outside image shape "
                f"{(original_height, original_width)} for {center_record.path}"
            )
        image_stack = image_stack[:, y1:y2, x1:x2]

        original_label = None
        label = None
        if self.require_label:
            original_label = self._load_label(center_record)
            if original_label.shape != (original_height, original_width):
                raise ValueError(
                    f"Image and label shapes differ in {center_record.path}: "
                    f"{(original_height, original_width)} vs {original_label.shape}"
                )
            label = original_label[y1:y2, x1:x2]
            if self.training:
                image_stack, label = self._augment(image_stack, label)

        resized_stack = np.stack(
            [
                cv2.resize(
                    channel,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                for channel in image_stack
            ]
        )
        resized_stack = (resized_stack - self.mean) / self.std

        item: dict[str, object] = {
            "image": torch.from_numpy(np.ascontiguousarray(resized_stack)).float(),
            "patient_id": center_record.patient_id,
            "sample_id": center_record.path.stem,
            "height": original_height,
            "width": original_width,
        }
        if label is not None:
            resized_label = cv2.resize(
                label.astype(np.int16),
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)
            item["label"] = torch.from_numpy(np.ascontiguousarray(resized_label)).long()
        if original_label is not None and self.return_original_label:
            item["original_label"] = torch.from_numpy(
                np.ascontiguousarray(original_label)
            ).long()
        return item
