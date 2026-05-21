from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_REPO_ROOT = Path(__file__).resolve().parent.parent


class BRCAM2CE2ECRDataset(Dataset):
	def __init__(
		self,
		data_root: str | Path,
		phase: str,
		crop_size: int = 384,
	) -> None:
		self.data_root = Path(data_root)
		self.image_root = self.data_root / "images"
		self.gt_root = self.data_root / "gt_custom"
		self.phase = phase
		self.crop_size = crop_size
		self.split_file = self._resolve_split_file(phase)
		self.image_names = np.loadtxt(self.split_file, dtype=str).tolist()
		if isinstance(self.image_names, str):
			self.image_names = [self.image_names]

		if not self.image_names:
			raise ValueError(f"split 文件为空: {self.split_file}")

		if not self.image_root.is_dir():
			raise ValueError(f"图像目录不存在: {self.image_root}")
		if not self.gt_root.is_dir():
			raise ValueError(f"点标注目录不存在: {self.gt_root}")

	def __len__(self) -> int:
		return len(self.image_names)

	def __getitem__(self, index: int):
		image_name = self.image_names[index]
		image = Image.open(self.image_root / image_name).convert("RGB")
		image_np = np.asarray(image, dtype=np.float32) / 255.0
		gt_dots = np.load(self.gt_root / image_name.replace(".png", "_gt_dots.npy"), allow_pickle=True)
		gt_dots = self._normalize_gt_dots(gt_dots)

		if self.phase == "train":
			image_np, gt_dots = self._apply_train_transforms(image_np, gt_dots)

		points, labels = self._extract_points_and_labels(gt_dots)
		image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
		target = {
			"points": torch.from_numpy(points).float(),
			"labels": torch.from_numpy(labels).long(),
			"image_id": torch.tensor(index, dtype=torch.long),
			"image_name": image_name,
		}
		return image_tensor, target

	def _normalize_gt_dots(self, gt_dots: np.ndarray) -> np.ndarray:
		if gt_dots.ndim != 3:
			raise ValueError(f"gt_dots shape 非法，期望 3 维，实际为 {gt_dots.shape}")

		if gt_dots.shape[2] == 4:
			return gt_dots[:, :, 1:4].astype(np.uint8)
		if gt_dots.shape[2] == 3:
			return gt_dots.astype(np.uint8)
		raise ValueError(f"gt_dots 通道数非法，期望 3 或 4，实际为 {gt_dots.shape}")

	def _resolve_split_file(self, phase: str) -> Path:
		dataset_name = self.data_root.name.lower()
		split_dir = _REPO_ROOT / "data_splits" / dataset_name
		phase_to_split = {
			"train": "train_split.txt",
			"val": "val_split.txt",
			"test": "test_split.txt",
		}
		if phase not in phase_to_split:
			raise ValueError(f"不支持的 phase: {phase}")
		split_file = split_dir / phase_to_split[phase]
		if not split_file.is_file():
			raise ValueError(f"split 文件不存在: {split_file}")
		return split_file

	def _apply_train_transforms(self, image_np: np.ndarray, gt_dots: np.ndarray):
		if random.random() < 0.5:
			image_np = image_np[:, ::-1].copy()
			gt_dots = gt_dots[:, ::-1].copy()
		if random.random() < 0.5:
			image_np = image_np[::-1, :].copy()
			gt_dots = gt_dots[::-1, :].copy()

		height, width = image_np.shape[:2]
		crop_height = min(self.crop_size, height)
		crop_width = min(self.crop_size, width)
		if crop_height < height or crop_width < width:
			top = random.randint(0, height - crop_height)
			left = random.randint(0, width - crop_width)
			image_np = image_np[top : top + crop_height, left : left + crop_width]
			gt_dots = gt_dots[top : top + crop_height, left : left + crop_width]

		return image_np, gt_dots

	def _extract_points_and_labels(self, gt_dots: np.ndarray):
		points: list[list[float]] = []
		labels: list[int] = []
		for class_index in range(gt_dots.shape[2]):
			y_coords, x_coords = np.where(gt_dots[:, :, class_index] > 0)
			for x_coord, y_coord in zip(x_coords.tolist(), y_coords.tolist()):
				points.append([float(x_coord), float(y_coord)])
				labels.append(class_index)

		if not points:
			return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
		return np.asarray(points, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def e2ecr_collate_fn(batch):
	images, targets = zip(*batch)
	return list(images), list(targets)


def build_e2ecr_dataset(dataset_type: str, data_root: str | Path, phase: str, crop_size: int = 384):
	dataset_type_normalized = dataset_type.strip().lower()
	dataset_registry = {
		"brca-m2c": BRCAM2CE2ECRDataset,
	}
	if dataset_type_normalized not in dataset_registry:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(dataset_registry.keys()))}"
		)
	dataset_cls = dataset_registry[dataset_type_normalized]
	return dataset_cls(data_root=data_root, phase=phase, crop_size=crop_size)


def get_num_classes(dataset_type: str) -> int:
	dataset_type_normalized = dataset_type.strip().lower()
	class_registry = {
		"brca-m2c": 3,
	}
	if dataset_type_normalized not in class_registry:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(class_registry.keys()))}"
		)
	return class_registry[dataset_type_normalized]