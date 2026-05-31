"""
train_loader.py

HoVer-Net 训练/验证阶段的数据加载器。
从预处理的 .npy 文件列表读取堆叠的「图像 + 标注」，经 albumentations 增强后，
再调用 target_gen 生成 HoVer-Net 所需的 HV 图、实例图等监督信号。
"""

import cv2
import numpy as np
import torch.utils.data

import albumentations as A

from misc.utils import cropping_center

from .augs import (
    RandomBrightness,
    RandomContrast,
    RandomGaussianBlur,
    RandomHue,
    RandomMedianBlur,
    RandomOrder,
    RandomSaturation,
)


####
class FileLoader(torch.utils.data.Dataset):
    """从 .npy 文件列表加载样本，并执行数据增强与 target 生成。

    每个 .npy 文件为通道堆叠格式：前 3 通道为 RGB 图像，后续通道为标注
    （实例 ID 图，以及可选的类型图）。

    增强库使用 albumentations；几何变换与标注同步，颜色/模糊等仅作用于 RGB。
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
        self.shape_augs = None
        self.input_augs = None
        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        """为当前 DataLoader worker 构建增强流水线（由 worker_init_fn 或 __init__ 调用）。"""
        if seed:
            np.random.seed(seed)
        self.shape_augs = self.__get_shape_augmentation(self.mode)
        self.input_augs = self.__get_input_augmentation(self.mode)
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

        # 几何增强：图像与标注同一次 Compose，保证像素对齐
        if self.shape_augs is not None:
            aug_kwargs = {"image": img, "mask": ann[..., 0]}
            if self.with_type:
                aug_kwargs["type_map"] = ann[..., 1]
            augmented = self.shape_augs(**aug_kwargs)
            img = augmented["image"]
            if self.with_type:
                ann = np.stack(
                    [
                        augmented["mask"].astype("int32"),
                        augmented["type_map"].astype("int32"),
                    ],
                    axis=-1,
                )
            else:
                ann = np.expand_dims(augmented["mask"].astype("int32"), -1)

        # 仅对图像做颜色/噪声等增强
        if self.input_augs is not None:
            img = self.input_augs(image=img)["image"]

        img = cropping_center(img, self.input_shape)
        feed_dict = {"img": img}

        inst_map = ann[..., 0]  # HW×C -> HW，实例 ID 图
        if self.with_type:
            type_map = (ann[..., 1]).copy()
            type_map = cropping_center(type_map, self.mask_shape)
            feed_dict["tp_map"] = type_map

        target_dict = self.target_gen_func(
            inst_map, self.mask_shape, **self.target_gen_kwargs
        )
        feed_dict.update(target_dict)

        return feed_dict

    def __get_shape_augmentation(self, mode):
        """构建同时作用于图像与标注的几何增强流水线。"""
        h, w = self.input_shape
        additional_targets = {"type_map": "mask"} if self.with_type else None

        if mode == "train":
            transforms = [
                A.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    shear=(-5, 5),
                    rotate=(-179, 179),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.PadIfNeeded(
                    min_height=h,
                    min_width=w,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                    position="center",
                ),
                A.CenterCrop(height=h, width=w),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        elif mode == "valid":
            transforms = [
                A.PadIfNeeded(
                    min_height=h,
                    min_width=w,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                    position="center",
                ),
                A.CenterCrop(height=h, width=w),
            ]
        else:
            raise ValueError("Unknown mode `%s`" % mode)

        return A.Compose(transforms, additional_targets=additional_targets)

    def __get_input_augmentation(self, mode):
        """构建仅作用于 RGB 图像的颜色/模糊增强流水线。"""
        if mode != "train":
            return None

        return A.Compose(
            [
                A.OneOf(
                    [
                        RandomGaussianBlur(max_ksize=3, p=1.0),
                        RandomMedianBlur(max_ksize=3, p=1.0),
                        A.GaussNoise(
                            std_range=(0.0, 0.05),
                            mean_range=(0.0, 0.0),
                            per_channel=True,
                            p=1.0,
                        ),
                    ],
                    p=1.0,
                ),
                RandomOrder(
                    [
                        RandomHue(range=(-8, 8), p=1.0),
                        RandomSaturation(range=(-0.2, 0.2), p=1.0),
                        RandomBrightness(range=(-26, 26), p=1.0),
                        RandomContrast(range=(0.75, 1.25), p=1.0),
                    ],
                    p=1.0,
                ),
            ]
        )
