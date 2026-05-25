from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


@dataclass
class E2ECRConfig:
	"""E2ECR 模型与匹配/推理过程的超参数配置。"""

	input_channels: int = 3
	num_classes: int = 3
	base_channels: int = 32
	encoder_name: str = "resnet34"
	pretrained_encoder: bool = False


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


def _build_resnet34_encoder(config: E2ECRConfig):
	# 编码器默认采用 ResNet-34。
	# 这样可以复用成熟的预训练特征提取能力，同时继续保留 U-Net 解码器对高分辨率细节的恢复能力。
	if config.encoder_name.lower() != "resnet34":
		raise ValueError(f"当前仅支持 encoder_name='resnet34'，实际为 {config.encoder_name}")

	weights = ResNet34_Weights.DEFAULT if config.pretrained_encoder and config.input_channels == 3 else None
	backbone = resnet34(weights=weights)

	# 细胞点检测比自然图像分类更依赖高分辨率细节。
	# 因此这里保留 ResNet 的残差主体，但把 stem 的首层步幅改为 1，
	# 同时在外部前向里去掉 maxpool，避免一开始就把空间分辨率压到 1/4。
	backbone.conv1.stride = (1, 1)

	# 如果输入通道数不是 3，则替换第一层卷积。
	# 预训练权重只对 3 通道输入直接可用，因此这里保留普通初始化。
	if config.input_channels != 3:
		backbone.conv1 = nn.Conv2d(
			config.input_channels,
			64,
			kernel_size=7,
			stride=1,
			padding=3,
			bias=False,
		)

	return backbone


class E2ECR(nn.Module):
	def __init__(self, config: E2ECRConfig | None = None) -> None:
		super().__init__()
		self.config = config or E2ECRConfig()

		# 这里将原来的纯 U-Net 编码器替换成 ResNet-34 编码器，
		# 但解码器仍然保留 U-Net 风格的逐层上采样与 skip fusion，
		# 从而兼顾预训练语义特征和点级定位所需的高分辨率细节。
		backbone = _build_resnet34_encoder(self.config)
		base = self.config.base_channels

		self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
		self.maxpool = nn.Identity()
		self.layer1 = backbone.layer1
		self.layer2 = backbone.layer2
		self.layer3 = backbone.layer3
		self.layer4 = backbone.layer4

		self.dec4 = UpBlock(512, 256, 256)
		self.dec3 = UpBlock(256, 128, 128)
		self.dec2 = UpBlock(128, 64, 64)
		self.dec1 = UpBlock(64, 64, 64)

		# 解码器回到 1/2 分辨率后，再上采样回原图大小，
		# 最后用较轻的共享头生成三张密集预测图。
		self.shared_head = nn.Sequential(
			nn.Conv2d(64, base, kernel_size=3, padding=1, bias=False),
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

		# 这里使用高分辨率 stem：
		# stem(1/1), layer1(1/1), layer2(1/2), layer3(1/4), layer4(1/8)。
		# 这样更符合点级密集预测任务对边缘和细粒度定位的要求。
		x1 = self.stem(image_batch)
		x2 = self.layer1(self.maxpool(x1))
		x3 = self.layer2(x2)
		x4 = self.layer3(x3)
		x5 = self.layer4(x4)

		x = self.dec4(x5, x4)
		x = self.dec3(x, x3)
		x = self.dec2(x, x2)
		x = self.dec1(x, x1)
		# dec1 后已经回到原图分辨率；这里保留一次尺寸对齐，主要用于处理奇数尺寸输入时的边界差异。
		x = F.interpolate(x, size=image_batch.shape[-2:], mode="bilinear", align_corners=False)
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