from __future__ import annotations

import csv
from collections import OrderedDict
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


# 通过惰性检测第三方库是否可用，避免在未安装依赖时导入阶段直接报错。
if importlib.util.find_spec("albumentations") is not None:
	A = importlib.import_module("albumentations")
else:
	A = None


class MoNuSACDataset(Dataset):
	"""读取 MoNuSAC patch 检测数据。

	数据来源：
	- annotations/boxes.csv: 每行一个目标框，包含 split 和 image_path。
	- metadata/classes.csv: 可选，仅在训练侧用于统计类别数。

	返回格式与 torchvision detection API 保持一致：
	- image: FloatTensor[C, H, W]，范围为 [0, 1]
	- target: dict，包含 boxes、labels、image_id、area、iscrowd
	"""

	def __init__(self, data_root: str | Path, annotations_csv: str | Path, split: str) -> None:
		self.data_root = Path(data_root)
		self.annotations_csv = Path(annotations_csv)
		self.split = split
		self.samples = self._load_samples()

		if not self.samples:
			raise ValueError(f"split={split} 没有可用样本，请检查 {self.annotations_csv}")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int):
		sample = self.samples[index]
		image = Image.open(sample["image_full_path"]).convert("RGB")
		image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), torch.float32)

		boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
		labels = torch.tensor(sample["labels"], dtype=torch.int64)
		area = torch.tensor(sample["area"], dtype=torch.float32)
		iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)

		target = {
			"boxes": boxes,
			"labels": labels,
			"image_id": torch.tensor([index], dtype=torch.int64),
			"area": area,
			"iscrowd": iscrowd,
		}
		return image_tensor, target

	def _load_samples(self) -> list[dict]:
		grouped: OrderedDict[str, dict] = OrderedDict()
		with self.annotations_csv.open("r", encoding="utf-8", newline="") as handle:
			reader = csv.DictReader(handle)
			for row in reader:
				if row["split"] != self.split:
					continue

				image_path = row["image_path"]
				sample = grouped.setdefault(
					image_path,
					{
						"image_path": image_path,
						"image_full_path": self.data_root / image_path,
						"boxes": [],
						"labels": [],
						"area": [],
					},
				)

				is_negative = int(row["is_negative"])
				if is_negative:
					continue

				xmin = float(row["xmin"])
				ymin = float(row["ymin"])
				xmax = float(row["xmax"])
				ymax = float(row["ymax"])
				sample["boxes"].append([xmin, ymin, xmax, ymax])
				sample["labels"].append(int(row["label_id"]))
				sample["area"].append(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))

		formatted_samples: list[dict] = []
		for sample in grouped.values():
			if sample["boxes"]:
				formatted_samples.append(sample)
				continue

			sample["boxes"] = torch.zeros((0, 4), dtype=torch.float32).tolist()
			sample["labels"] = torch.zeros((0,), dtype=torch.int64).tolist()
			sample["area"] = torch.zeros((0,), dtype=torch.float32).tolist()
			formatted_samples.append(sample)

		return formatted_samples


class BRCAM2CE2ECRDataset(Dataset):
	def __init__(
		self,
		data_root: str | Path,
		phase: str,
		crop_size: int = 384,
		transform: bool = False
	) -> None:
		self.data_root = Path(data_root)
		self.image_root = self.data_root / "images"
		self.gt_root = self.data_root / "gt_custom"
		self.phase = phase
		self.target_size = crop_size
		self.transform = transform
		if self.phase == "train" and self.transform and A is None:
			raise ImportError("训练增强依赖 albumentations，请先安装: pip install albumentations")
		# 训练增强交给第三方库统一管理，避免手写几何变换时漏掉点坐标同步。
		self.train_transform = self._build_train_transform() if self.phase == "train" and self.transform else None
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
		points, labels = self._extract_points_and_labels(gt_dots)

		if self.phase == "train" and self.transform:
			image_np, points = self._apply_train_transforms(image_np, points)

		image_np, points = self._resize_image_and_points(image_np, points)
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
		split_dir = self.data_root / "data_splits"
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

	def _build_train_transform(self):
		# Albumentations 会自动同步图像和 keypoints 的几何变化，
		# 这里优先选择对点标注安全、且对病理图像泛化更有帮助的标准增强。
		return A.Compose(
			[
				A.HorizontalFlip(p=0.5),
				A.VerticalFlip(p=0.5),
				A.RandomRotate90(p=0.75),
				A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
				A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=8, val_shift_limit=6, p=0.5),
			],
			keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
		)

	def _apply_train_transforms(self, image_np: np.ndarray, points: np.ndarray):
		if self.train_transform is None:
			return np.ascontiguousarray(image_np), points.astype(np.float32, copy=False)

		# 对空点集也统一走同一套接口，保证训练代码不需要区分是否有 GT。
		transformed = self.train_transform(
			image=image_np,
			keypoints=points.tolist() if points.shape[0] > 0 else [],
		)
		transformed_points = np.asarray(transformed["keypoints"], dtype=np.float32)
		if transformed_points.size == 0:
			transformed_points = np.zeros((0, 2), dtype=np.float32)
		return np.ascontiguousarray(transformed["image"]), transformed_points

	def _resize_image_and_points(self, image_np: np.ndarray, points: np.ndarray):
		original_height, original_width = image_np.shape[:2]
		target_height = self.target_size
		target_width = self.target_size

		image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
		image_tensor = F.resize(
			image_tensor,
			size=[target_height, target_width],
			interpolation=F.InterpolationMode.BILINEAR,
			antialias=True,
		)
		image_np = image_tensor.permute(1, 2, 0).numpy()

		if points.shape[0] == 0:
			return image_np, points.astype(np.float32)

		scaled_points = points.astype(np.float32).copy()
		scaled_points[:, 0] = scaled_points[:, 0] * (target_width / original_width)
		scaled_points[:, 1] = scaled_points[:, 1] * (target_height / original_height)
		return image_np, scaled_points

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


class CoNSePE2ECRDataset(BRCAM2CE2ECRDataset):
	def __init__(
		self,
		data_root: str | Path,
		phase: str,
		crop_size: int = 384,
		transform: bool = False,
		use_five_fold: bool = False,
		fold_index: int | None = None,
	) -> None:
		self.use_five_fold = use_five_fold
		self.fold_index = fold_index
		super().__init__(data_root=data_root, phase=phase, crop_size=crop_size, transform=transform)

	def __getitem__(self, index: int):
		image_name = self.image_names[index]
		image = Image.open(self.image_root / image_name).convert("RGB")
		image_np = np.asarray(image, dtype=np.float32) / 255.0
		gt_dots = np.load(self.gt_root / image_name.replace(".png", "_gt_dots.npy"), allow_pickle=True)
		gt_dots = self._normalize_gt_dots(gt_dots)
		points, labels = self._extract_points_and_labels(gt_dots)

		if self.phase == "train" and self.transform:
			image_np, points = self._apply_train_transforms(image_np, points)
			# CoNSeP 训练图像数量较少，因此训练阶段额外使用随机裁剪，
			# 让每张整图在不同 epoch 提供更多局部视野，提升数据利用率。
			image_np, points = self._apply_random_crop(image_np, points, self.target_size)

		# CoNSeP 训练样本如果已经被随机裁成目标尺寸，下面 resize 基本是恒等操作；
		# 验证和测试则会统一缩放到目标尺寸，保持与现有训练脚本接口兼容。
		image_np, points = self._resize_image_and_points(image_np, points)
		image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
		target = {
			"points": torch.from_numpy(points).float(),
			"labels": torch.from_numpy(labels).long(),
			"image_id": torch.tensor(index, dtype=torch.long),
			"image_name": image_name,
		}
		return image_tensor, target

	def _resolve_split_file(self, phase: str) -> Path:
		split_dir = self.data_root / "data_splits"
		if self.use_five_fold and phase in {"train", "val"}:
			if self.fold_index is None or self.fold_index < 1:
				raise ValueError("使用 CoNSeP 五折验证时，fold_index 必须是从 1 开始的正整数")
			split_file = split_dir / "five_fold" / f"fold{self.fold_index}_{phase}.txt"
		else:
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

	def _apply_random_crop(self, image_np: np.ndarray, points: np.ndarray, crop_size: int):
		height, width = image_np.shape[:2]
		crop_height = min(crop_size, height)
		crop_width = min(crop_size, width)
		if crop_height == height and crop_width == width:
			return image_np, points.astype(np.float32, copy=False)

		max_top = height - crop_height
		max_left = width - crop_width

		if points.shape[0] > 0:
			# 优先围绕一个随机真实点裁剪，减少裁到纯背景 patch 的概率。
			center_point = points[np.random.randint(0, points.shape[0])]
			min_left = max(0, int(np.floor(center_point[0])) - crop_width + 1)
			max_left_candidate = min(max_left, int(np.floor(center_point[0])))
			min_top = max(0, int(np.floor(center_point[1])) - crop_height + 1)
			max_top_candidate = min(max_top, int(np.floor(center_point[1])))

			if min_left <= max_left_candidate:
				left = int(np.random.randint(min_left, max_left_candidate + 1))
			else:
				left = int(np.random.randint(0, max_left + 1))
			if min_top <= max_top_candidate:
				top = int(np.random.randint(min_top, max_top_candidate + 1))
			else:
				top = int(np.random.randint(0, max_top + 1))
		else:
			left = int(np.random.randint(0, max_left + 1))
			top = int(np.random.randint(0, max_top + 1))

		right = left + crop_width
		bottom = top + crop_height
		cropped_image = np.ascontiguousarray(image_np[top:bottom, left:right])

		if points.shape[0] == 0:
			return cropped_image, points.astype(np.float32, copy=False)

		inside_mask = (
			(points[:, 0] >= left)
			& (points[:, 0] < right)
			& (points[:, 1] >= top)
			& (points[:, 1] < bottom)
		)
		cropped_points = points[inside_mask].astype(np.float32, copy=True)
		if cropped_points.shape[0] > 0:
			cropped_points[:, 0] -= left
			cropped_points[:, 1] -= top
		return cropped_image, cropped_points


def detection_collate_fn(batch):
	images, targets = zip(*batch)
	return list(images), list(targets)


def e2ecr_collate_fn(batch):
	images, targets = zip(*batch)
	return list(images), list(targets)


def get_num_classes(classes_csv: str | Path) -> int:
	classes_csv = Path(classes_csv)
	with classes_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		label_ids = [int(row["label_id"]) for row in reader]
	if not label_ids:
		raise ValueError(f"未能从 {classes_csv} 读取到类别信息")
	return max(label_ids) + 1


def build_e2ecr_dataset(
	dataset_type: str,
	data_root: str | Path,
	phase: str,
	crop_size: int = 384,
	transform: bool = False,
	use_five_fold: bool = False,
	fold_index: int | None = None,
):
	dataset_type_normalized = dataset_type.strip().lower()
	phase_normalized = phase.strip().lower()
	dataset_registry = {
		"brca-m2c": BRCAM2CE2ECRDataset,
		"consep": CoNSePE2ECRDataset,
	}
	if dataset_type_normalized not in dataset_registry:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(dataset_registry.keys()))}"
		)
	dataset_cls = dataset_registry[dataset_type_normalized]
	# 训练集打开增强，验证和测试保持确定性，避免评估口径漂移。
	dataset_kwargs = {
		"data_root": data_root,
		"phase": phase_normalized,
		"crop_size": crop_size,
		"transform": transform,
	}
	if dataset_type_normalized == "consep":
		dataset_kwargs["use_five_fold"] = use_five_fold
		dataset_kwargs["fold_index"] = fold_index
	return dataset_cls(
		data_root=data_root,
		phase=phase_normalized,
		crop_size=crop_size,
		transform=transform,
		**({"use_five_fold": use_five_fold, "fold_index": fold_index} if dataset_type_normalized == "consep" else {}),
	)


def get_e2ecr_num_classes(dataset_type: str) -> int:
	dataset_type_normalized = dataset_type.strip().lower()
	class_registry = {
		"brca-m2c": 3,
		"consep": 3,
	}
	if dataset_type_normalized not in class_registry:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(class_registry.keys()))}"
		)
	return class_registry[dataset_type_normalized]