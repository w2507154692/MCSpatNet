from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class E2ECRConfig:
	"""E2ECR 模型与匹配/推理过程的超参数配置。"""

	input_channels: int = 3
	num_classes: int = 3
	base_channels: int = 32
	alpha: float = 0.05
	beta: float = 0.6
	lambda_reg: float = 2e-3
	candidate_multiplier: int = 8
	min_candidates: int = 128
	max_candidates: int = 2048
	inference_obj_threshold: float = 0.2
	inference_score_threshold: float = 0.15
	nms_radius: float = 6.0
	max_predictions_per_image: int = 512


class ConvBlock(nn.Module):
	"""U-Net 中使用的基础卷积块：两层 3x3 Conv + BN + ReLU。"""

	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)


class UpBlock(nn.Module):
	"""U-Net 解码阶段的上采样块。

	处理流程：
	1. 将低分辨率特征双线性上采样到 skip 特征大小。
	2. 与编码器同层特征拼接。
	3. 通过 1x1 降维和卷积块融合语义与细节。
	"""

	def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
		super().__init__()
		self.reduce = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)
		self.block = ConvBlock(out_channels, out_channels)

	def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
		x = torch.cat([x, skip], dim=1)
		x = self.reduce(x)
		return self.block(x)


class E2ECR(nn.Module):
	def __init__(self, config: E2ECRConfig | None = None) -> None:
		super().__init__()
		self.config = config or E2ECRConfig()

		# 这里使用一个轻量 U-Net 主干，输出共享的高分辨率特征图，
		# 再分别接回归头、检测头、分类头。
		base = self.config.base_channels
		self.enc1 = ConvBlock(self.config.input_channels, base)
		self.enc2 = ConvBlock(base, base * 2)
		self.enc3 = ConvBlock(base * 2, base * 4)
		self.bottleneck = ConvBlock(base * 4, base * 8)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.dec3 = UpBlock(base * 8, base * 4, base * 4)
		self.dec2 = UpBlock(base * 4, base * 2, base * 2)
		self.dec1 = UpBlock(base * 2, base, base)

		self.shared_head = nn.Sequential(
			nn.Conv2d(base, base, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(base),
			nn.ReLU(inplace=True),
		)
		self.reg_head = nn.Conv2d(base, 2, kernel_size=1)
		self.det_head = nn.Conv2d(base, 2, kernel_size=1)
		self.cls_head = nn.Conv2d(base, self.config.num_classes, kernel_size=1)

	def forward(self, images):
		"""模型主入口，只负责输出预测图。

		输入可以是 [B, C, H, W] 张量，也可以是由多张图组成的列表。
		若当前 batch 中图像尺寸不一致，这里会自动补齐到相同大小，
		完成一次真正的 batch 前向后，再按原图大小裁回每张图的预测结果。
		"""

		if isinstance(images, torch.Tensor):
			if images.ndim != 4:
				raise ValueError(f"images 张量应为 4 维 [B, C, H, W]，实际为 {images.shape}")
			image_batch = images
			original_sizes = [(int(images.shape[-2]), int(images.shape[-1]))] * int(images.shape[0])
		else:
			image_list = list(images)
			if not image_list:
				raise ValueError("images 不能为空")
			original_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in image_list]
			max_height = max(height for height, _ in original_sizes)
			max_width = max(width for _, width in original_sizes)
			padded_images = []
			for image, (height, width) in zip(image_list, original_sizes):
				pad_right = max_width - width
				pad_bottom = max_height - height
				padded_images.append(F.pad(image, (0, pad_right, 0, pad_bottom), value=1.0))
			image_batch = torch.stack(padded_images, dim=0)

		x1 = self.enc1(image_batch)
		x2 = self.enc2(self.pool(x1))
		x3 = self.enc3(self.pool(x2))
		x4 = self.bottleneck(self.pool(x3))

		x = self.dec3(x4, x3)
		x = self.dec2(x, x2)
		x = self.dec1(x, x1)
		x = self.shared_head(x)

		reg_batch = self.reg_head(x)
		det_batch = self.det_head(x)
		cls_batch = self.cls_head(x)

		outputs = []
		for sample_index, (height, width) in enumerate(original_sizes):
			outputs.append(
				{
					"reg": reg_batch[sample_index, :, :height, :width],
					"det": det_batch[sample_index, :, :height, :width],
					"cls": cls_batch[sample_index, :, :height, :width],
				}
			)
		return outputs

	def export_config(self) -> dict:
		return asdict(self.config)


def build_e2ecr(config: E2ECRConfig | None = None) -> E2ECR:
	return E2ECR(config=config)