"""
dataset.py

HoVer-Net 各公开数据集的图像与标注加载接口。
每个数据集类封装「如何读取原图」和「如何解析 .mat 标注」的差异。
"""

import glob
import cv2
import numpy as np
import scipy.io as sio


class __AbstractDataset(object):
    """数据集抽象基类，定义后续各数据集必须实现的接口。

    设计目的：将不同数据集的图像读取与标注解析逻辑封装在各自子类中，
    上层 pipeline 只需调用统一的 load_img / load_ann，无需关心具体格式。
    """

    def load_img(self, path):
        """读取并返回 RGB 图像，形状 (H, W, 3)。"""
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        """读取标注；with_type=True 时同时返回实例图与类型图。"""
        raise NotImplementedError


####
class __Kumar(__AbstractDataset):
    """Kumar 数据集加载器。

    原始论文：
    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane,
    and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for
    computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.

    标注仅含实例分割图 inst_map，不含细胞类型。
    """

    def load_img(self, path):
        # OpenCV 默认 BGR，转为 RGB 供后续网络使用
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # 假设标注为 H×W 的实例 ID 图
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)  # (H, W) -> (H, W, 1)
        return ann


####
class __CPM17(__AbstractDataset):
    """CPM 2017 数据集加载器。

    原始论文：
    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban,
    Talha Qaiser, Navid Alemi Koohbanani et al. "Methods for segmentation and classification
    of digital microscopy tissue images." Frontiers in bioengineering and biotechnology 7 (2019).

    标注格式与 Kumar 相同：仅 inst_map，无类型图。
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # 假设标注为 H×W 的实例 ID 图
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CoNSeP(__AbstractDataset):
    """CoNSeP 数据集加载器。

    原始论文：
    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak,
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563

    支持 with_type：除 inst_map 外还可读取 type_map，并按论文规则合并为 4 类（含背景）。
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # 假设 inst_map 为 H×W；with_type 时 type_map 同为 H×W
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]
            ann_type_origin = ann_type.copy()

            # CoNSeP 原始 7 类合并为论文使用的 3 类细胞 + 背景（见 Hover-Net 原文）
            # 若使用自有数据集，需按实际类别定义修改下列映射
            ann_type[(ann_type_origin == 2)] = 1
            ann_type[(ann_type_origin == 3) | (ann_type_origin == 4)] = 2
            ann_type[(ann_type_origin == 1) | (ann_type_origin == 5) | (ann_type_origin == 6) | (ann_type_origin == 7)] = 3
            

            # 通道 0：实例 ID；通道 1：合并后的类型 ID
            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann


####
class __MoNuSAC(__AbstractDataset):
    """MoNuSAC 分割 patch 加载器（inst_map + type_map，2 类细胞 + 背景）。

    原始 label_id：1=Epithelial，2=Lymphocyte，3=Macrophage，4=Neutrophil。
    与 MCSpatNet 一致，仅保留前两类；3/4 类像素与实例置为背景。
    训练时 config.nr_type 应设为 3（背景 + 2 类细胞）。
    """

    _ACTIVE_TYPE_IDS = (1, 2)

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        mat = sio.loadmat(path)
        ann_inst = mat["inst_map"].astype("int32")
        if with_type:
            ann_type = mat["type_map"].astype("int32")
            keep_mask = np.isin(ann_type, self._ACTIVE_TYPE_IDS)
            ann_inst = np.where(keep_mask, ann_inst, 0).astype("int32")
            ann_type = np.where(keep_mask, ann_type, 0).astype("int32")
            ann = np.dstack([ann_inst, ann_type])
        else:
            ann_type = mat["type_map"].astype("int32")
            keep_mask = np.isin(ann_type, self._ACTIVE_TYPE_IDS)
            ann_inst = np.where(keep_mask, ann_inst, 0).astype("int32")
            ann = np.expand_dims(ann_inst, -1)
        return ann


####
def get_dataset(name):
    """根据名称返回预定义的数据集加载器实例。

    Args:
        name: 数据集名称，不区分大小写。可选 'kumar'、'cpm17'、'consep'、'monusac'。

    Returns:
        对应数据集的 __AbstractDataset 子类实例。
    """
    name_dict = {
        "kumar": lambda: __Kumar(),
        "cpm17": lambda: __CPM17(),
        "consep": lambda: __CoNSeP(),
        "monusac": lambda: __MoNuSAC(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name
