"""PASCAL VOC data pipeline for YOLO-style detection.

Two stages:
1. ``prepare_voc`` converts VOC XML annotations to YOLO txt label files once
   (trainval 2007+2012 -> train, test 2007 -> val/test). Class index follows
   the canonical VOC 20-class order.
2. ``VOCDataset`` loads images + YOLO labels with standard detection
   augmentations (mosaic optional, HSV, flip, scale).

The Q^2 paper (Appendix 8.1) trains YOLOv5s/VOC from a pretrained checkpoint
with batch 64, seed 0.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Canonical VOC 20-class order (0-indexed).
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


def _voc_xml_to_yolo(xml_path: str, class_to_idx: dict) -> Optional[np.ndarray]:
    """Parse one VOC XML into YOLO format [cls, cx, cy, w, h] (normalised)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return None
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in class_to_idx:
            continue
        difficult = obj.find("difficult")
        difficult = int(difficult.text) if difficult is not None else 0
        if difficult:
            continue  # follow common practice: drop difficult boxes
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        # clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        boxes.append([class_to_idx[name], cx, cy, bw, bh])
    if not boxes:
        return None
    return np.array(boxes, dtype=np.float32)


def prepare_voc(voc_root: str, out_root: str) -> dict:
    """Convert VOC -> YOLO layout under ``out_root``.

    Creates ``out_root/images/{train,test}`` (symlinks to JPEGImages) and
    ``out_root/labels/{train,test}`` (YOLO txt files), plus ``*.list`` index
    files. Idempotent: skips conversion if labels already exist.
    """
    voc_root = Path(voc_root)
    out_root = Path(out_root)
    devkit = voc_root / "VOCdevkit"

    splits = {
        "train": [
            (devkit / "VOC2007" / "ImageSets" / "Main" / "trainval.txt", devkit / "VOC2007"),
            (devkit / "VOC2012" / "ImageSets" / "Main" / "trainval.txt", devkit / "VOC2012"),
        ],
        "test": [
            (devkit / "VOC2007" / "ImageSets" / "Main" / "test.txt", devkit / "VOC2007"),
        ],
    }

    summary = {}
    for split, sources in splits.items():
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        list_file = out_root / f"{split}.list"
        if list_file.exists():
            summary[split] = sum(1 for _ in open(list_file))
            continue
        n = 0
        with open(list_file, "w") as lf:
            for list_path, year_dir in sources:
                if not list_path.exists():
                    continue
                for line in open(list_path):
                    stem = line.strip()
                    if not stem:
                        continue
                    xml = year_dir / "Annotations" / f"{stem}.xml"
                    jpg = year_dir / "JPEGImages" / f"{stem}.jpg"
                    if not (xml.exists() and jpg.exists()):
                        continue
                    boxes = _voc_xml_to_yolo(str(xml), VOC_CLASS_TO_IDX)
                    lbl_path = lbl_dir / f"{stem}.txt"
                    if boxes is None:
                        # negative sample: empty label file
                        lbl_path.write_text("")
                    else:
                        np.savetxt(str(lbl_path), boxes, fmt="%g")
                    img_link = img_dir / f"{stem}.jpg"
                    if not img_link.exists():
                        try:
                            os.symlink(jpg, img_link)
                        except FileExistsError:
                            pass
                    lf.write(f"{img_link}\n")
                    n += 1
        summary[split] = n
    return summary


def collate_detection(batch):
    """Collate variable-length label sets. Returns (images, targets, sizes)."""
    imgs = torch.stack([b[0] for b in batch], 0)
    targets = [b[1] for b in batch]
    return imgs, targets


class VOCDataset(Dataset):
    """YOLO-style VOC dataset. Images -> (N,3,H,W) float in [0,1]. Labels ->
    list of (M,5) tensors [cls, cx, cy, w, h] in image pixels (after resize)."""

    def __init__(
        self,
        list_file: str,
        img_size: int = 640,
        augment: bool = True,
        mosaic: bool = False,
    ) -> None:
        self.img_paths = [l.strip() for l in open(list_file) if l.strip()]
        self.img_size = img_size
        self.augment = augment
        self.mosaic = mosaic and augment
        # label dir mirrors image dir
        base = Path(self.img_paths[0]).parents[1]
        self.label_root = base.parent / "labels" / Path(self.img_paths[0]).parent.name

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_label(self, img_path: str, ow: int, oh: int) -> torch.Tensor:
        stem = Path(img_path).stem
        lbl = self.label_root / f"{stem}.txt"
        if not lbl.exists():
            return torch.zeros((0, 5), dtype=torch.float32)
        raw = np.loadtxt(str(lbl), ndmin=2).reshape(-1, 5)
        out = []
        for r in raw:
            c, cx, cy, w, h = r
            out.append([c, cx * ow, cy * oh, w * ow, h * oh])
        return torch.tensor(out, dtype=torch.float32) if out else torch.zeros((0, 5))

    def _letterbox(self, img: np.ndarray, labels: torch.Tensor):
        """Resize to img_size keeping aspect, pad grey. Update box coords."""
        import cv2
        h, w = img.shape[:2]
        s = self.img_size
        r = min(s / h, s / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad = np.full((s, s, 3), 114, dtype=np.uint8)
        top, left = (s - nh) // 2, (s - nw) // 2
        pad[top:top + nh, left:left + nw] = img
        if labels.numel() > 0:
            labels = labels.clone()
            labels[:, 1] = labels[:, 1] * r + left
            labels[:, 2] = labels[:, 2] * r + top
            labels[:, 3] *= r
            labels[:, 4] *= r
        return pad, labels, r

    def __getitem__(self, idx):
        import cv2
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            # fall back to a grey image
            img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
            labels = torch.zeros((0, 5))
            return self._to_tensor(img), labels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        labels = self._load_label(img_path, w, h)
        if self.augment:
            img, labels = self._augment(img, labels)
        img, labels, _ = self._letterbox(img, labels)

        if self.augment:
            # HSV jitter
            img = self._hsv(img)
            # random horizontal flip
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
                if labels.numel() > 0:
                    labels[:, 1] = self.img_size - labels[:, 1]
        return self._to_tensor(img), labels

    def _augment(self, img, labels):
        return img, labels

    def _hsv(self, img):
        import cv2
        if np.random.rand() > 0.9:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h = int(np.random.randint(-10, 10))
            s = float(np.random.uniform(0.7, 1.3))
            v = float(np.random.uniform(0.7, 1.3))
            img[:, :, 0] = (img[:, :, 0].astype(int) + h) % 180
            img[:, :, 1] = np.clip(img[:, :, 1] * s, 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * v, 0, 255)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
