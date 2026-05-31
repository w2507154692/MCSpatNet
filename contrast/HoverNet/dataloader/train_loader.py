"""
train_loader.py

HoVer-Net 训练/验证阶段的数据加载器。
从预处理的 .npy 文件列表读取堆叠的「图像 + 标注」，经 imgaug 增强后，
再调用 target_gen 生成 HoVer-Net 所需的 HV 图、实例图等监督信号。
"""

import csv
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch.utils.data

import imgaug as ia
from imgaug import augmenters as iaa
from misc.utils import cropping_center

from .augs import (
    add_to_brightness,
    add_to_contrast,
    add_to_hue,
    add_to_saturation,
    gaussian_blur,
    median_blur,
)


####
class FileLoader(torch.utils.data.Dataset):
    """从 .npy 文件列表加载样本，并执行数据增强与 target 生成。

    每个 .npy 文件为通道堆叠格式：前 3 通道为 RGB 图像，后续通道为标注
    （实例 ID 图，以及可选的类型图）。

    增强库使用 imgaug（docstring 中提到的 albumentation 为历史表述）。
    增强完成后会裁剪到 input_shape / mask_shape，并生成水平/垂直距离图等 target。

    Args:
        file_list: 待加载的 .npy 文件路径列表
        with_type: 是否读取并返回细胞类型图 tp_map
        input_shape: 网络输入尺寸 [h, w]，在 config.py 中定义
        mask_shape: 监督图输出尺寸 [h, w]，在 config.py 中定义
        mode: 'train' 或 'valid'，决定增强策略
        setup_augmentor: 是否在 __init__ 中立即初始化增强器（多进程时由 worker_init_fn 延迟初始化）
        target_gen: (target_gen_func, target_gen_kwargs) 元组，用于生成 HoVer-Net 训练 target
    """

    # TODO: 补充更完整的 docstring

    def __init__(
        self,
        file_list,
        with_type=False,
        input_shape=None,
        mask_shape=None,
        mode="train",
        setup_augmentor=True,
        target_gen=None,
    ):
        assert input_shape is not None and mask_shape is not None
        self.mode = mode
        self.info_list = file_list
        self.with_type = with_type
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = target_gen[0]
        self.target_gen_kwargs = target_gen[1]
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        """为当前 DataLoader worker 构建增强流水线（由 worker_init_fn 或 __init__ 调用）。"""
        self.augmentor = self.__get_augmentation(self.mode, seed)
        # shape_augs：同时作用于图像与标注（几何变换，保持对齐）
        self.shape_augs = iaa.Sequential(self.augmentor[0])
        # input_augs：仅作用于 RGB 图像（颜色/模糊等，不改变标注）
        self.input_augs = iaa.Sequential(self.augmentor[1])
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        path = self.info_list[idx]
        data = np.load(path)

        # 将堆叠通道拆分为图像与标注
        img = (data[..., :3]).astype("uint8")  # RGB 图像
        ann = (data[..., 3:]).astype("int32")  # 实例 ID 图 + 可选类型图

        # 几何增强：图像与标注必须使用同一份 deterministic 变换，保证像素对齐
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            img = shape_augs.augment_image(img)
            ann = shape_augs.augment_image(ann)

        # 仅对图像做颜色/噪声等增强
        if self.input_augs is not None:
            input_augs = self.input_augs.to_deterministic()
            img = input_augs.augment_image(img)

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW×C -> HW，实例 ID 图
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            #type_map[type_map == 5] = 1  # merge neoplastic and non-neoplastic
            feed_dict["tp_map"] = type_map

        # TODO: 文档化此处关于输入通道数的硬编码假设
        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)

        return feed_dict

    def __get_augmentation(self, mode, rng):
        """按 train / valid 模式返回 (shape_augs, input_augs) 增强器列表。"""
        if mode == "train":
            shape_augs = [
                # * order = ``0`` -> ``cv2.INTER_NEAREST``（最近邻，适合标注图）
                # * order = ``1`` -> ``cv2.INTER_LINEAR``
                # * order = ``2`` -> ``cv2.INTER_CUBIC``
                # * order = ``3`` -> ``cv2.INTER_CUBIC``
                # * order = ``4`` -> ``cv2.INTER_CUBIC``
                # ! 对 PanNuke v0 可关闭旋转/平移，仅翻转以避免 mirror padding；此处为通用配置
                iaa.Affine(
                    # 各轴独立缩放至原尺寸的 80%–120%
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # 各轴平移 ±1%（相对图像尺寸）
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),  # 剪切角度 -5° 至 +5°
                    rotate=(-179, 179),  # 旋转 -179° 至 +179°
                    order=0,  # 使用最近邻插值，避免标注 ID 被插值污染
                    backend="cv2",  # 使用 OpenCV 后端以加速
                    seed=rng,
                ),
                # position='center'：中心裁剪；'uniform'：随机位置裁剪
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                ),
                iaa.Fliplr(0.5, seed=rng),  # 50% 概率水平翻转
                iaa.Flipud(0.5, seed=rng),  # 50% 概率垂直翻转
            ]

            input_augs = [
                # 以下三种增强随机选一种
                iaa.OneOf(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: median_blur(*args, max_ksize=3),
                        ),
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                        ),
                    ]
                ),
                # 色调 / 饱和度 / 亮度 / 对比度，顺序随机
                iaa.Sequential(
                    [
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_saturation(
                                *args, range=(-0.2, 0.2)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_brightness(
                                *args, range=(-26, 26)
                            ),
                        ),
                        iaa.Lambda(
                            seed=rng,
                            func_images=lambda *args: add_to_contrast(
                                *args, range=(0.75, 1.25)
                            ),
                        ),
                    ],
                    random_order=True,
                ),
            ]
        elif mode == "valid":
            shape_augs = [
                # position='center'：中心裁剪；'uniform'：随机位置裁剪
                iaa.CropToFixedSize(
                    self.input_shape[0], self.input_shape[1], position="center"
                )
            ]
            input_augs = []

        return shape_augs, input_augs
