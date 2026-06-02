"""
config.py

HoVer-Net 全局训练配置。
集中定义随机种子、数据集、patch 尺寸、模型模式及 train/valid 路径等，
并动态加载 models/hovernet/opt.py 中的 model_config。
"""

import importlib
import random

import cv2
import numpy as np

from dataset import get_dataset


class Config(object):
    """HoVer-Net 训练/验证的全局配置类。"""

    def __init__(self):
        # 随机种子，用于复现实验（DataLoader、增强、网络初始化等）
        self.seed = 42

        # 是否启用 TensorBoard / stats.json 等日志
        self.logging = True

        # 开启 debug 可关闭 DataLoader 多进程，便于追踪并行加载问题
        self.debug = False

        model_name = "hovernet"
        model_mode = "original"  # 可选 `original` 或 `fast`

        if model_mode not in ["original", "fast"]:
            raise Exception("Must use either `original` or `fast` as model mode")

        # 细胞核类型数（含背景）。CoNSeP 合并 3 类细胞 + 背景 → 0~3，共 4 类
        nr_type = 3
        self.nr_type = nr_type

        # 是否预测细胞核类型；能否开启取决于数据集是否提供 type_map
        self.type_classification = True

        # ---------- patch 尺寸说明 ----------
        # 以下默认配置对应 original 模式：
        #   original：act_shape=[270,270]，out_shape=[80,80]
        #   fast    ：act_shape=[256,256]，out_shape=[164,164]
        # aug_shape：增强阶段使用的较大 patch，可减轻边界伪影
        aug_shape = [540, 540]
        # act_shape：送入网络的输入 patch（增强后中心裁剪）
        act_shape = [270, 270]
        # out_shape：网络输出 / 监督图对应的 patch 尺寸
        out_shape = [80, 80]

        if model_mode == "original":
            if act_shape != [270, 270] or out_shape != [80, 80]:
                raise Exception(
                    "If using `original` mode, input shape must be [270,270] and output shape must be [80,80]"
                )
        if model_mode == "fast":
            if act_shape != [256, 256] or out_shape != [164, 164]:
                raise Exception(
                    "If using `fast` mode, input shape must be [256,256] and output shape must be [164,164]"
                )

        # 数据集名称，对应 dataset.py 中 get_dataset 的键（kumar / cpm17 / consep）
        self.dataset_name = "monusac"
        # checkpoint 与训练日志保存目录
        self.log_dir = "../exp/exp28_HoverNet_monusac"

        # 训练 / 验证 patch 目录（各目录下应为预处理好的 .npy 文件）
        self.train_dir_list = [
            "../data/MoNuSAC_HoverNet/monusac/train/540x540_164x164"
        ]
        self.valid_dir_list = [
            "../data/MoNuSAC_HoverNet/monusac/valid/540x540_164x164"
        ]

        # 各阶段 input_shape（网络输入）与 mask_shape（监督图尺寸）
        self.shape_info = {
            "train": {"input_shape": act_shape, "mask_shape": out_shape,},
            "valid": {"input_shape": act_shape, "mask_shape": out_shape,},
        }

        # * 解析配置到运行态：加载数据集解析器，并导入对应模型的 opt 模块
        self.dataset = get_dataset(self.dataset_name)

        module = importlib.import_module(
            "models.%s.opt" % model_name
        )
        self.model_config = module.get_config(nr_type, model_mode)
