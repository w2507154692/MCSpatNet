from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence

from torch import nn
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.detection import FasterRCNN, anchor_utils, faster_rcnn
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.misc import FrozenBatchNorm2d


class ResNetC4Backbone(nn.Module):
	"""标准 C4 主干：输出 ResNet 的 conv4_x 特征图。"""

	def __init__(self, conv1, bn1, relu, maxpool, layer1, layer2, layer3):
		super().__init__()
		self.conv1 = conv1
		self.bn1 = bn1
		self.relu = relu
		self.maxpool = maxpool
		self.layer1 = layer1
		self.layer2 = layer2
		self.layer3 = layer3
		self.out_channels = 1024

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		return OrderedDict([("0", x)])


class ResNetC4BoxHead(nn.Module):
	"""标准 C4 RoI head：使用 ResNet 的 conv5_x 作为 box head。"""

	def __init__(self, layer4):
		super().__init__()
		self.layer4 = layer4
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

	def forward(self, x):
		x = self.layer4(x)
		x = self.avgpool(x)
		return x


@dataclass(slots=True)
class FasterRCNNConfig:
	"""标准 C4 Faster R-CNN 的配置项。

	说明：
	- num_classes 需要包含背景类。
	- 当前实现固定使用 ResNet-101 作为骨干，并输出 conv4_x 单尺度特征图。
	- RoI head 使用 ResNet-101 的 conv5_x，即标准 C4 结构。
	- anchor 默认采用更小的尺度，适配细胞核这类小目标检测任务。
	- trainable_backbone_layers 表示从高层到低层保留可训练状态的 ResNet stage 数量。
	"""

	num_classes: int
	pretrained_detector: bool = False
	pretrained_backbone: bool = False
	# 0 表示主干全冻结，5 表示从 conv1/bn1 到 layer4 全部参与训练。
	trainable_backbone_layers: int = 5
	min_size: int = 512
	max_size: int = 512
	image_mean: Sequence[float] = (0.485, 0.456, 0.406)
	image_std: Sequence[float] = (0.229, 0.224, 0.225)
	rpn_anchor_sizes: tuple[tuple[int, ...], ...] = ((8, 16, 32, 64, 128),)
	rpn_aspect_ratios: tuple[tuple[float, ...], ...] = field(
		default_factory=lambda: ((0.5, 1.0, 2.0),)
	)
	box_score_thresh: float = 0.05
	box_nms_thresh: float = 0.5
	box_detections_per_img: int = 1000


def build_faster_rcnn(config: FasterRCNNConfig | None = None) -> FasterRCNN:
	"""构建标准 C4 + ResNet-101 的 Faster R-CNN。

	默认选择：
	- 主干网络：ResNet-101 C4
	- box head：ResNet-101 的 conv5_x
	- anchor：使用比自然图像默认值更小的尺度，更适合细胞核检测
	"""

	if config is None:
		config = FasterRCNNConfig(num_classes=5)

	weights_backbone, norm_layer = _resolve_backbone_setup(config)
	anchor_sizes, aspect_ratios = _canonicalize_anchor_config(config)
	anchor_generator = anchor_utils.AnchorGenerator(
		sizes=anchor_sizes,
		aspect_ratios=aspect_ratios,
	)
	backbone_model = resnet101(weights=weights_backbone, norm_layer=norm_layer)
	_freeze_backbone_layers(backbone_model, config.trainable_backbone_layers)

	backbone = ResNetC4Backbone(
		conv1=backbone_model.conv1,
		bn1=backbone_model.bn1,
		relu=backbone_model.relu,
		maxpool=backbone_model.maxpool,
		layer1=backbone_model.layer1,
		layer2=backbone_model.layer2,
		layer3=backbone_model.layer3,
	)
	box_head = ResNetC4BoxHead(backbone_model.layer4)
	box_predictor = faster_rcnn.FastRCNNPredictor(2048, config.num_classes)

	# C4 仅使用单尺度 conv4_x 特征图，因此 RoIAlign 只对一个特征层取样。
	box_roi_pool = MultiScaleRoIAlign(
		featmap_names=["0"],
		output_size=14,
		sampling_ratio=2,
	)

	model = FasterRCNN(
		backbone=backbone,
		num_classes=None,
		min_size=config.min_size,
		max_size=config.max_size,
		image_mean=list(config.image_mean),
		image_std=list(config.image_std),
		rpn_anchor_generator=anchor_generator,
		box_roi_pool=box_roi_pool,
		box_head=box_head,
		box_predictor=box_predictor,
		box_score_thresh=config.box_score_thresh,
		box_nms_thresh=config.box_nms_thresh,
		box_detections_per_img=config.box_detections_per_img,
	)
	return model


def replace_box_predictor(model: FasterRCNN, num_classes: int) -> FasterRCNN:
	"""在复用已有 checkpoint 时，替换检测头以适配新的类别集合。"""

	model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(2048, num_classes)
	return model


def _canonicalize_anchor_config(config: FasterRCNNConfig):
	"""把旧的 FPN 风格 anchor 配置折叠为单特征层 C4 配置。"""

	anchor_sizes = config.rpn_anchor_sizes
	if len(anchor_sizes) > 1 and all(len(sizes) == 1 for sizes in anchor_sizes):
		anchor_sizes = (tuple(size[0] for size in anchor_sizes),)

	aspect_ratios = config.rpn_aspect_ratios
	if len(aspect_ratios) > 1 and all(ratios == aspect_ratios[0] for ratios in aspect_ratios[1:]):
		aspect_ratios = (tuple(aspect_ratios[0]),)

	return anchor_sizes, aspect_ratios


def _freeze_backbone_layers(backbone_model, trainable_backbone_layers: int):
	if not 0 <= trainable_backbone_layers <= 5:
		raise ValueError("trainable_backbone_layers 必须在 0 到 5 之间")

	layers_in_order = ["layer4", "layer3", "layer2", "layer1", "conv1"]
	trainable_layers = set(layers_in_order[:trainable_backbone_layers])
	train_stem_bn = "conv1" in trainable_layers

	for name, parameter in backbone_model.named_parameters():
		is_trainable = any(name.startswith(layer_name) for layer_name in trainable_layers)
		if name.startswith("bn1"):
			is_trainable = train_stem_bn
		parameter.requires_grad_(is_trainable)


def _resolve_backbone_setup(config: FasterRCNNConfig):
	"""根据配置决定 backbone 权重和归一化层。

	说明：
	- 当前实现固定为自定义的 C4 + ResNet-101，没有直接可复用的 torchvision detector 权重。
	- 因此 pretrained_detector 与 pretrained_backbone 都退化为加载 ResNet-101 的 backbone 权重。
	"""

	weights_backbone = None
	norm_layer = nn.BatchNorm2d

	if config.pretrained_detector or config.pretrained_backbone:
		weights_backbone = ResNet101_Weights.DEFAULT
		norm_layer = FrozenBatchNorm2d

	return weights_backbone, norm_layer
