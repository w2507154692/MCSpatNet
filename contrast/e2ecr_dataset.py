from __future__ import annotations

import csv
import importlib
import importlib.util
from abc import abstractmethod
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

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_PHASE_TO_SPLIT_NAME = {
	"train": "train",
	"val": "val",
	"test": "test",
}


class E2ECRDatasetBase(Dataset):
	"""E2ECR 数据集基类：统一增强、裁剪、缩放与 target 组装逻辑。"""

	use_random_crop: bool = False

	def __init__(
		self,
		data_root: str | Path,
		phase: str,
		crop_size: int = 384,
		transform: bool = False,
	) -> None:
		self.data_root = Path(data_root)
		self.phase = phase.strip().lower()
		if self.phase not in _PHASE_TO_SPLIT_NAME:
			raise ValueError(f"不支持的 phase: {phase}")
		self.target_size = crop_size
		self.transform = transform
		if self.phase == "train" and self.transform and A is None:
			raise ImportError("训练增强依赖 albumentations，请先安装: pip install albumentations")
		self.train_transform = self._build_train_transform() if self.phase == "train" and self.transform else None
		self.image_names: list[str] = []
		self._setup_dataset_index()

	def __len__(self) -> int:
		return len(self.image_names)

	def __getitem__(self, index: int):
		image_np, points, labels, image_name = self._read_sample(index)
		points = points.astype(np.float32, copy=True)
		labels = labels.astype(np.int64, copy=True)

		if self.phase == "train" and self.transform:
			image_np, points = self._apply_train_transforms(image_np, points)
			if self.use_random_crop:
				image_np, points, labels = self._apply_random_crop(image_np, points, labels, self.target_size)

		image_np, points = self._resize_image_and_points(image_np, points)
		image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
		target = {
			"points": torch.from_numpy(points).float(),
			"labels": torch.from_numpy(labels).long(),
			"image_id": torch.tensor(index, dtype=torch.long),
			"image_name": image_name,
		}
		return image_tensor, target

	@abstractmethod
	def _setup_dataset_index(self) -> None:
		"""子类负责填充 self.image_names 及各自所需的索引结构。"""

	@abstractmethod
	def _read_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
		"""返回 image_np(H,W,3)、points(N,2)、labels(N,) 与 image_name。"""

	def _build_train_transform(self):
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

	def _apply_random_crop(self, image_np: np.ndarray, points: np.ndarray, labels: np.ndarray, crop_size: int):
		height, width = image_np.shape[:2]
		crop_height = min(crop_size, height)
		crop_width = min(crop_size, width)
		if crop_height == height and crop_width == width:
			return image_np, points.astype(np.float32, copy=False), labels.astype(np.int64, copy=False)

		max_top = height - crop_height
		max_left = width - crop_width

		if points.shape[0] > 0:
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
			return cropped_image, points.astype(np.float32, copy=False), labels.astype(np.int64, copy=False)

		inside_mask = (
			(points[:, 0] >= left)
			& (points[:, 0] < right)
			& (points[:, 1] >= top)
			& (points[:, 1] < bottom)
		)
		cropped_points = points[inside_mask].astype(np.float32, copy=True)
		cropped_labels = labels[inside_mask].astype(np.int64, copy=True)
		if cropped_points.shape[0] > 0:
			cropped_points[:, 0] -= left
			cropped_points[:, 1] -= top
		return cropped_image, cropped_points, cropped_labels

	def _normalize_gt_dots(self, gt_dots: np.ndarray) -> np.ndarray:
		if gt_dots.ndim != 3:
			raise ValueError(f"gt_dots shape 非法，期望 3 维，实际为 {gt_dots.shape}")

		if gt_dots.shape[2] == 4:
			return gt_dots[:, :, 1:4].astype(np.uint8)
		if gt_dots.shape[2] == 3:
			return gt_dots.astype(np.uint8)
		raise ValueError(f"gt_dots 通道数非法，期望 3 或 4，实际为 {gt_dots.shape}")

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


class SplitFileGtDotsDataset(E2ECRDatasetBase):
	"""通过 data_splits/*.txt 划分样本，并从 gt_custom 读取点标注。"""

	def _setup_dataset_index(self) -> None:
		self.image_root = self.data_root / "images"
		self.gt_root = self.data_root / "gt_custom"
		split_file = self._resolve_split_file(self.phase)
		image_names = np.loadtxt(split_file, dtype=str).tolist()
		if isinstance(image_names, str):
			image_names = [image_names]
		self.image_names = image_names

		if not self.image_names:
			raise ValueError(f"split 文件为空: {split_file}")
		if not self.image_root.is_dir():
			raise ValueError(f"图像目录不存在: {self.image_root}")
		if not self.gt_root.is_dir():
			raise ValueError(f"点标注目录不存在: {self.gt_root}")

	def _read_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
		image_name = self.image_names[index]
		image = Image.open(self.image_root / image_name).convert("RGB")
		image_np = np.asarray(image, dtype=np.float32) / 255.0
		gt_dots = np.load(self.gt_root / image_name.replace(".png", "_gt_dots.npy"), allow_pickle=True)
		gt_dots = self._normalize_gt_dots(gt_dots)
		points, labels = self._extract_points_and_labels(gt_dots)
		return image_np, points, labels, image_name

	def _resolve_split_file(self, phase: str) -> Path:
		split_dir = self.data_root / "data_splits"
		phase_to_split = {
			"train": "train_split.txt",
			"val": "val_split.txt",
			"test": "test_split.txt",
		}
		split_file = split_dir / phase_to_split[phase]
		if not split_file.is_file():
			raise ValueError(f"split 文件不存在: {split_file}")
		return split_file


class BRCAM2CE2ECRDataset(SplitFileGtDotsDataset):
	use_random_crop = False


class CoNSePE2ECRDataset(SplitFileGtDotsDataset):
	use_random_crop = True

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
			split_file = split_dir / phase_to_split[phase]

		if not split_file.is_file():
			raise ValueError(f"split 文件不存在: {split_file}")
		return split_file


class MoNuSACDataset(E2ECRDatasetBase):
	"""读取 MoNuSAC_point：按 images/{phase}/ 目录划分，点标注来自 annotations/points.csv。

	仅保留前两类：1=Epithelial，2=Lymphocyte；Macrophage/Neutrophil 会被忽略。
	"""

	use_random_crop = True
	_ACTIVE_LABEL_IDS = frozenset({1, 2})

	def _setup_dataset_index(self) -> None:
		self.annotations_csv = self.data_root / "annotations" / "points.csv"
		if not self.annotations_csv.is_file():
			raise ValueError(f"点标注文件不存在: {self.annotations_csv}")

		self.phase_image_dir = self.data_root / "images" / self.phase
		if not self.phase_image_dir.is_dir():
			raise ValueError(f"图像目录不存在: {self.phase_image_dir}")

		self.points_by_image = self._load_points_from_csv(self.phase)
		self.image_names = self._list_phase_image_paths()
		if not self.image_names:
			raise ValueError(f"phase={self.phase} 在 {self.phase_image_dir} 下没有可用图像")

	def _read_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
		image_path = self.image_names[index]
		image_full_path = self.data_root / image_path
		if not image_full_path.is_file():
			raise FileNotFoundError(f"图像不存在: {image_full_path}")

		image = Image.open(image_full_path).convert("RGB")
		image_np = np.asarray(image, dtype=np.float32) / 255.0
		points, labels = self.points_by_image.get(
			image_path,
			(np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)),
		)
		return image_np, points, labels, image_path

	def _list_phase_image_paths(self) -> list[str]:
		image_paths: list[str] = []
		for path in sorted(self.phase_image_dir.rglob("*")):
			if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
				continue
			relative_path = path.relative_to(self.data_root).as_posix()
			image_paths.append(relative_path)
		return image_paths

	def _load_points_from_csv(self, phase: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
		phase_prefix = f"images/{phase}/"
		raw_points: dict[str, list[tuple[float, float, int]]] = {}

		with self.annotations_csv.open("r", encoding="utf-8", newline="") as handle:
			reader = csv.DictReader(handle)
			for row in reader:
				image_path = row["image_path"].replace("\\", "/")
				if not image_path.startswith(phase_prefix):
					continue
				if int(row["is_negative"]):
					continue

				label_id = int(row["label_id"])
				if label_id not in self._ACTIVE_LABEL_IDS:
					continue
				raw_points.setdefault(image_path, []).append(
					(float(row["x"]), float(row["y"]), label_id - 1)
				)

		points_by_image: dict[str, tuple[np.ndarray, np.ndarray]] = {}
		for image_path, entries in raw_points.items():
			if entries:
				entry_array = np.asarray(entries, dtype=np.float32)
				points_by_image[image_path] = (entry_array[:, :2], entry_array[:, 2].astype(np.int64))
			else:
				points_by_image[image_path] = (
					np.zeros((0, 2), dtype=np.float32),
					np.zeros((0,), dtype=np.int64),
				)
		return points_by_image


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
		"monusac": MoNuSACDataset,
	}
	if dataset_type_normalized not in dataset_registry:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(dataset_registry.keys()))}"
		)
	dataset_cls = dataset_registry[dataset_type_normalized]
	return dataset_cls(
		data_root=data_root,
		phase=phase_normalized,
		crop_size=crop_size,
		transform=transform,
		**(
			{"use_five_fold": use_five_fold, "fold_index": fold_index}
			if dataset_type_normalized == "consep"
			else {}
		),
	)


def get_e2ecr_num_classes(dataset_type: str) -> int:
	dataset_type_normalized = dataset_type.strip().lower()
	class_registry = {
		"brca-m2c": 3,
		"consep": 3,
		"monusac": 2,
	}
	if dataset_type_normalized not in class_registry:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(class_registry.keys()))}"
		)
	return class_registry[dataset_type_normalized]


_E2ECR_CLASS_VIZ = {
	"brca-m2c": {
		"names": ["inflammatory", "epithelial", "stromal"],
		"gt_colors": ["blue", "red", "yellow"],
		"pred_colors": ["blue", "red", "yellow"],
	},
	"consep": {
		"names": ["inflammatory", "epithelial", "stromal"],
		"gt_colors": ["blue", "red", "yellow"],
		"pred_colors": ["blue", "red", "yellow"],
	},
	"monusac": {
		"names": ["Epithelial", "Lymphocyte"],
		"gt_colors": ["red", "lime"],
		"pred_colors": ["red", "lime"],
	},
}


def get_e2ecr_class_names(dataset_type: str) -> list[str]:
	dataset_type_normalized = dataset_type.strip().lower()
	if dataset_type_normalized not in _E2ECR_CLASS_VIZ:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(_E2ECR_CLASS_VIZ.keys()))}"
		)
	return list(_E2ECR_CLASS_VIZ[dataset_type_normalized]["names"])


def get_e2ecr_gt_colors(dataset_type: str) -> list[str]:
	dataset_type_normalized = dataset_type.strip().lower()
	if dataset_type_normalized not in _E2ECR_CLASS_VIZ:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(_E2ECR_CLASS_VIZ.keys()))}"
		)
	return list(_E2ECR_CLASS_VIZ[dataset_type_normalized]["gt_colors"])


def get_e2ecr_pred_colors(dataset_type: str) -> list[str]:
	dataset_type_normalized = dataset_type.strip().lower()
	if dataset_type_normalized not in _E2ECR_CLASS_VIZ:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(_E2ECR_CLASS_VIZ.keys()))}"
		)
	return list(_E2ECR_CLASS_VIZ[dataset_type_normalized]["pred_colors"])
