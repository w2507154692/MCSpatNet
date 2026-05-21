from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


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


def detection_collate_fn(batch):
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