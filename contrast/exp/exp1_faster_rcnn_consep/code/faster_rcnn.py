from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from torch import nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN, anchor_utils, faster_rcnn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d


@dataclass(slots=True)
class FasterRCNNConfig:
	"""Faster R-CNN baseline 的配置项。

	说明：
	- num_classes 需要包含背景类。
	- 这里默认使用 512 输入尺寸，与当前 MoNuSac patch 设置保持一致。
	- anchor 默认采用更小的尺度，适配细胞核这类小目标检测任务。
	- trainable_backbone_layers 表示 ResNet 主干中保留可训练状态的层数。
	"""

	num_classes: int
	pretrained_detector: bool = False
	pretrained_backbone: bool = False
	# torchvision 对 ResNet-FPN 通常允许取值 0-5：
	# 0 表示 backbone 全冻结，5 表示从 stem 到 layer4 全部参与训练。
	# 这里默认设为 3，是一个常见 baseline 折中：保留高层特征适应检测任务，
	# 同时减少小数据集上全量微调带来的不稳定和显存压力。
	trainable_backbone_layers: int = 5
	min_size: int = 512
	max_size: int = 512
	image_mean: Sequence[float] = (0.485, 0.456, 0.406)
	image_std: Sequence[float] = (0.229, 0.224, 0.225)
	rpn_anchor_sizes: tuple[tuple[int, ...], ...] = ((8,), (16,), (32,), (64,), (128,))
	rpn_aspect_ratios: tuple[tuple[float, ...], ...] = field(
		default_factory=lambda: ((0.5, 1.0, 2.0),) * 5
	)
	box_score_thresh: float = 0.05
	box_nms_thresh: float = 0.5
	box_detections_per_img: int = 300


def build_faster_rcnn(config: FasterRCNNConfig | None = None) -> FasterRCNN:
	"""构建一个便于后续扩展的 torchvision Faster R-CNN baseline。

	默认选择：
	- 主干网络：ResNet-50 FPN
	- 检测器主体：直接实例化 torchvision 的 FasterRCNN 类
	- anchor：使用比自然图像默认值更小的尺度，更适合细胞核检测
	"""

	if config is None:
		config = FasterRCNNConfig(num_classes=5)

	weights_backbone, norm_layer = _resolve_backbone_setup(config)
	# 自定义 RPN anchor。这里把尺度压小，减少对超大目标先验的依赖。
	anchor_generator = anchor_utils.AnchorGenerator(
		sizes=config.rpn_anchor_sizes,
		aspect_ratios=config.rpn_aspect_ratios,
	)
	# RoIAlign 负责从 FPN 多尺度特征图中为候选框提取固定尺寸特征。
	box_roi_pool = MultiScaleRoIAlign(
		featmap_names=["0", "1", "2", "3"],
		output_size=7,
		sampling_ratio=2,
	)
	# 使用公开的 backbone 构造接口，避免 torchvision 预置工厂内部重复传参。
	backbone = resnet_fpn_backbone(
		backbone_name="resnet50",
		weights=weights_backbone,
		norm_layer=norm_layer,
		trainable_layers=config.trainable_backbone_layers,
	)

	# 直接实例化 FasterRCNN，保留对 anchor 和 RoIAlign 的完整控制权。
	model = FasterRCNN(
		backbone=backbone,
		num_classes=config.num_classes,
		min_size=config.min_size,
		max_size=config.max_size,
		image_mean=list(config.image_mean),
		image_std=list(config.image_std),
		rpn_anchor_generator=anchor_generator,
		box_roi_pool=box_roi_pool,
		box_score_thresh=config.box_score_thresh,
		box_nms_thresh=config.box_nms_thresh,
		box_detections_per_img=config.box_detections_per_img,
	)
	return model


def replace_box_predictor(model: FasterRCNN, num_classes: int) -> FasterRCNN:
	"""在复用已有 checkpoint 时，替换检测头以适配新的类别集合。"""

	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
	return model


def _resolve_backbone_setup(config: FasterRCNNConfig):
	"""根据配置决定 backbone 权重和归一化层。

	说明：
	- 当前实现走自定义 FasterRCNN 构造路径，因此主要复用 backbone 预训练权重。
	- 若配置了 pretrained_detector，这里退化为使用对应 detector 的 backbone 权重。
	"""

	weights_backbone = None
	norm_layer = nn.BatchNorm2d

	if config.pretrained_detector:
		weights_backbone = ResNet50_Weights.verify(
			FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.backbone
		)
		norm_layer = FrozenBatchNorm2d
	elif config.pretrained_backbone:
		weights_backbone = ResNet50_Weights.DEFAULT
		norm_layer = FrozenBatchNorm2d

	return weights_backbone, norm_layer
