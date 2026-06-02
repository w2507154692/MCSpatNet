"""extract_patches.py

HoVer-Net 训练用 patch 提取脚本。

将整张 WSI/patch 图像与 .mat 标注切分为固定窗口大小的 .npy 文件，
供 train_loader.FileLoader 直接加载。输出为 RGB + 标注通道堆叠格式。
"""

import re
import glob
import os
import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # 是否在标注中同时提取类型图（仅对带类别标签的数据集有效，如 CoNSeP）
    type_classification = True

    # 滑窗 patch 尺寸 [高, 宽]（与 config.py 中 aug_shape 对应，通常 540×540）
    win_size = [540, 540]
    # 滑窗步长 [高, 宽]（决定 patch 重叠程度）
    step_size = [164, 164]
    # 边界处理方式：'mirror' 对边界镜像 padding；'valid' 仅提取完全在图像内的区域
    extract_type = "mirror"

    # 数据集名称，可选 kumar / cpm17 / consep / monusac
    # 用于从 dataset.py 获取对应的 load_img / load_ann 实现
    dataset_name = "monusac"
    save_root = "../data/MoNuSAC_HoverNet/"

    # MoNuSAC 示例（需先运行 MoNuSAC数据集处理.ipynb 生成 ../data/MoNuSAC_seg）:
    # dataset_name = "monusac"
    # save_root = "../data/MoNuSAC_HoverNet/"
    # dataset_info = {
    #     "train": {"img": (".png", "../data/MoNuSAC_seg/train/Images/"), "ann": (".mat", "../data/MoNuSAC_seg/train/Labels/")},
    #     "valid": {"img": (".png", "../data/MoNuSAC_seg/val/Images/"), "ann": (".mat", "../data/MoNuSAC_seg/val/Labels/")},
    # }

    # 各 split 的图像与标注路径配置
    # 键：img / ann -> (文件后缀, 目录路径)
    dataset_info = {
        "train": {
            "img": (".png", "../data/MoNuSAC_seg/Train/Images/"),
            "ann": (".mat", "../data/MoNuSAC_seg/Train/Labels/"),
        },
        "valid": {
            "img": (".png", "../data/MoNuSAC_seg/Val/Images/"),
            "ann": (".mat", "../data/MoNuSAC_seg/Val/Labels/"),
        },
    }

    # 转义 glob 路径中的方括号，避免被当作字符类
    patterning = lambda x: re.sub(r"([\[\]])", r"[\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        # 输出目录命名含 patch 尺寸与步长，便于区分不同提取配置
        out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        # 以标注文件列表为驱动，保证每张图都有对应 .mat
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # 固定排序，保证跨平台顺序一致

        rm_n_mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # * 将 RGB 与标注在通道维堆叠，再按滑窗切 patch
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()
