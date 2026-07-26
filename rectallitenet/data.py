from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

cv2.setNumThreads(1)

CARE_MEAN = 0.1364736
CARE_STD = 0.23238614
FIXED_CROP = (88, 54, 440, 406)
SAMPLE_PATTERN = re.compile(r"^(?P<patient>case\d+)_slice(?P<slice>\d+)$")


@dataclass(frozen=True)
class Record:
    path: Path
    patient_id: str
    slice_id: int


def decode_label(label: np.ndarray) -> np.ndarray:
    decoded = np.asarray(label).astype(np.int64, copy=True)
    decoded[decoded < 0] = 0
    decoded[decoded > 2] = 2
    return decoded


def records_from_manifest(
    manifest: str | Path, data_root: str | Path
) -> list[Record]:
    root = Path(data_root)
    records = []
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
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
        raise ValueError(f"No records in {manifest}")
    return records


def official_reference_resize(label: np.ndarray, size: int = 224) -> np.ndarray:
    rows = zoom(np.arange(label.shape[0]), size / label.shape[0], order=0).astype(int)
    columns = zoom(np.arange(label.shape[1]), size / label.shape[1], order=0).astype(
        int
    )
    return label[np.ix_(rows, columns)].astype(np.int64, copy=False)


class CAREDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        image_size: int = 384,
        training: bool = False,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.training = training
        self.by_key = {
            (record.patient_id, record.slice_id): record for record in records
        }
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self) -> int:
        return len(self.records)

    def _image(self, record: Record) -> np.ndarray:
        with np.load(record.path) as sample:
            image = np.squeeze(sample["image"]).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return self.clahe.apply((image * 255).astype(np.uint8)).astype(np.float32) / 255

    def _label(self, record: Record) -> np.ndarray:
        with np.load(record.path) as sample:
            return decode_label(np.squeeze(sample["label"]))

    def _augment(
        self, stack: np.ndarray, label: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            turns = random.randint(0, 3)
            stack = np.rot90(stack, turns, axes=(-2, -1)).copy()
            label = np.rot90(label, turns).copy()
        if random.random() < 0.5:
            axis = random.choice((-2, -1))
            stack = np.flip(stack, axis=axis).copy()
            label = np.flip(label, axis=axis + 2).copy()
        if random.random() < 0.5:
            angle = random.uniform(-20.0, 20.0)
            height, width = label.shape
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            stack = np.stack(
                [
                    cv2.warpAffine(
                        channel,
                        matrix,
                        (width, height),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                    for channel in stack
                ]
            )
            label = cv2.warpAffine(
                label.astype(np.int16),
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            ).astype(np.int64)
        return stack, label

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        neighbors = [
            self.by_key.get((record.patient_id, record.slice_id + offset), record)
            for offset in (-1, 0, 1)
        ]
        stack = np.stack([self._image(item) for item in neighbors])
        original_label = self._label(record)
        x1, y1, x2, y2 = FIXED_CROP
        stack = stack[:, y1:y2, x1:x2]
        label = original_label[y1:y2, x1:x2]
        if self.training:
            stack, label = self._augment(stack, label)
        stack = np.stack(
            [
                cv2.resize(
                    channel,
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_CUBIC,
                )
                for channel in stack
            ]
        )
        stack = (stack - CARE_MEAN) / CARE_STD
        label = cv2.resize(
            label.astype(np.int16),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(stack)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "original_label": torch.from_numpy(
                np.ascontiguousarray(original_label)
            ).long(),
            "reference_label": torch.from_numpy(
                np.ascontiguousarray(official_reference_resize(original_label))
            ).long(),
            "patient_id": record.patient_id,
            "sample_id": record.path.stem,
        }
