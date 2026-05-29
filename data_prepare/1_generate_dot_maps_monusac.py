"""
对 MoNuSAC_point_MCSpatNet 数据集进行处理。
"""

import csv
import glob
import os
from collections import defaultdict

import cv2
import numpy as np
import scipy
import skimage.io as io
from scipy import ndimage

# 配置变量。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
in_root_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../data/MoNuSAC_point_MCSpatNet"))
annotations_csv = os.path.join(in_root_dir, "annotations", "points.csv")
DATA_SPLITS = ("train", "val", "test")

# 原始类别索引的最大值。类别编号从 1 开始，0 作为背景保留。
classes_max_indx = 4
# 可视化颜色（与 data/MoNuSAC_point/可视化.ipynb 一致）
color_set = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
}
# 四类细胞 1:1 映射，不做合并。
class_group_mapping_dict = {
    1: [1],
    2: [2],
    3: [3],
    4: [4],
}
n_grouped_class_channels = 5  # 4 个类别加上背景
# 图像缩放比例。原始 patch 与细胞中心坐标会按相同比例缩小。
img_scale = 1.0
remove_duplicates = False  # 若为 True，则去除 5 像素邻域内重复标注的细胞点。
"""
原始细胞类别：
          1 = epithelial
	      2 = lymphocyte
	      3 = macrophage
	      4 = neutrophil
分组后的细胞类别：
	      1 = epithelial
	      2 = lymphocyte
	      3 = macrophage
	      4 = neutrophil
"""

"""
    本脚本假设输入数据满足以下目录结构和标注格式（仿照 CoNSeP_train，按 split 组织）：
        - 在 <in_root_dir> 下：
            {train,val,test}/images/：存放 patch 图像。
            {train,val,test}/gt_custom/：存放本脚本生成的标注与密度图。
            annotations/points.csv：点标注，字段含 image_path, x, y, label_id, is_negative。
        - CSV 中 image_path 形如 images/train/xxx.png，对应磁盘路径 train/images/xxx.png。
        - 坐标按 (x, y) 存储，label_id 从 1 开始。
        - k_func_maps 由后续 2_calc_kmaps.py 生成，不在本脚本中输出。
"""

IMAGE_GLOB_PATTERNS = ("*.png", "*.jpg", "*.jpeg")


def csv_image_path_to_fs_path(csv_image_path):
    """将 CSV 中的 image_path 转为相对 in_root_dir 的磁盘路径。"""
    csv_image_path = csv_image_path.replace("\\", "/")
    if csv_image_path.startswith("images/"):
        split, filename = csv_image_path[len("images/") :].split("/", 1)
        return f"{split}/images/{filename}"
    return csv_image_path


def load_points_by_image(csv_path):
    """按磁盘相对路径聚合点标注，跳过 is_negative=1 的样本。"""
    grouped = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["is_negative"]):
                continue
            fs_image_path = csv_image_path_to_fs_path(row["image_path"])
            grouped[fs_image_path].append(
                (float(row["x"]), float(row["y"]), int(row["label_id"]))
            )
    return grouped


def gaussian_filter_density(
    img, points, point_class_map, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1
):
    """
    根据细胞中心点生成高斯密度图/二值掩码图。

    img：原始图像的缩放图
    points：[N, 2]，存储每个细胞像素的坐标
    point_class_map：[H, W, C]，存储每个像素的类别
    out_filepath：输出路径，图像名.npy

    处理流程如下：
    1. 先对所有点构建 KD-tree，并为每个点查找最近邻距离。
    2. 根据最近邻距离自适应设置该点对应高斯核的 sigma，避免相邻细胞的高斯区域过度重叠。
    3. 对每个点单独生成一个高斯分布，并归一化后累加到最终密度图中。
    4. 同时按类别分别累加，得到分类密度图。
    5. 最终不直接保存连续值密度图，而是通过 >0 的方式转成二值图后保存。

    说明：
    - 默认高斯核最大宽度对应 kernel_width=9。
    - sigma 采用自适应方式：min(最近邻距离 * 0.125, 2)，并在 truncate=2 下截断。
    - 会额外保存二值图的可视化结果，例如 <img_name>_binary.png。
    """
    img_shape = [img.shape[0], img.shape[1]]
    print(
        "Shape of current image: ",
        img_shape,
        ". Totally need generate ",
        len(points),
        "gaussian kernels.",
    )
    density = np.zeros(img_shape, dtype=np.float32)  # density: [H, W]
    density_class = np.zeros(
        (img.shape[0], img.shape[1], point_class_map.shape[2]), dtype=np.float32
    )  # density_class: [H, W, C]
    if end_y <= 0:
        end_y = img.shape[0]
    if end_x <= 0:
        end_x = img.shape[1]
    gt_count = len(points)
    if gt_count == 0:
        return density
    leafsize = 2048
    # 构建 KD-tree，用于快速查询每个点的最近邻距离。
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # 查询每个点最近的两个邻居。第一个通常是其自身，第二个是真正的最近邻。
    distances, locations = tree.query(points, k=2)
    print("generate density...")

    max_sigma = 2
    # kernel size = 4, kernel_width=9

    for i, pt in enumerate(points):  # pt是坐标
        pt2d = np.zeros(
            img_shape, dtype=np.float32
        )  # pt2d: [H, W, 1]，将坐标在图上标记出来
        if pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x:
            continue
        pt[1] -= start_y
        pt[0] -= start_x
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.0  # 将坐标在图上标记出来
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]) * 0.125  # 第 i 个细胞的最近邻
            sigma = min(max_sigma, sigma)
        else:
            sigma = max_sigma

        # 根据 sigma 反推使用的离散高斯核尺寸，并统一让 sigma 与核尺寸对应，保证滤波稳定。
        kernel_size = min(max_sigma * 2, int(2 * sigma + 0.5))
        sigma = kernel_size / 2
        kernel_width = kernel_size * 2 + 1
        # if(kernel_width < 9):
        #     print('i',i)
        #     print('distances',distances.shape)
        #     print('kernel_width',kernel_width)
        pnt_density = scipy.ndimage.filters.gaussian_filter(
            pt2d, sigma, mode="constant", truncate=2
        )
        # 归一化后再累加，使每个细胞点对总密度图的贡献一致。
        pnt_density /= pnt_density.sum()
        density += pnt_density
        class_indx = point_class_map[
            int(pt[1]), int(pt[0])
        ].argmax()  # 取出当前点对应的类别
        density_class[:, :, class_indx] = density_class[:, :, class_indx] + pnt_density

    # density_class.astype(np.float16).dump(out_filepath)
    # density.astype(np.float16).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
    # KEY：保存的是每个类别的高斯密度图（二值化）（可以说是膨胀点图），名称如train_1.npy
    (density_class > 0).astype(np.uint8).dump(out_filepath)
    # KEY：保存的是所有类别总的高斯密度图（二值化），名称如train_1_all.npy
    (density > 0).astype(np.uint8).dump(
        os.path.splitext(out_filepath)[0] + "_all.npy"
    )  # 图像名_all.npy，保存的是所有类别总的高斯密度图
    # io.imsave(out_filepath.replace('.npy', '.png'), (density / density.max() * 255).astype(np.uint8))
    # KEY：保存所有类别总的高斯密度图（二值化），不过是png格式（供可视化），名称如train_1_binary.png
    io.imsave(
        out_filepath.replace(".npy", "_binary.png"),
        ((density > 0) * 255).astype(np.uint8),
    )
    # KEY：保存每个类别的高斯密度图（二值化），png格式，名称如train_1_s0_binary.png
    for s in range(1, density_class.shape[-1]):
        io.imsave(
            out_filepath.replace(".npy", "_s" + str(s) + "_binary.png"),
            ((density_class[:, :, s] > 0) * 255).astype(np.uint8),
        )
    print("done.")
    return density.astype(np.float16), density_class.astype(np.float16)


if __name__ == "__main__":
    """
    对每张图像，脚本会执行以下处理：
        1. 对 patch 图像和标注的细胞中心坐标做统一缩放。
            同时会生成一个彩色可视化图，将不同类别的细胞点叠加到图像上，
            保存为 <split>/gt_custom/<img_name>_img_with_dots.jpg。
        2. 生成分类点标注图，保存为 <split>/gt_custom/<img_name>_gt_dots.npy。
        3. 生成检测点标注图，保存为 <split>/gt_custom/<img_name>_gt_dots_all.npy。
        4. 以每个细胞中心为中心生成高斯图，并自适应设置高斯宽度，以尽量避免相邻细胞区域过度相交。
        5. 最终将高斯图转为二值掩码保存，即所有 >0 的像素置为 1，其余位置为 0。
            分类图保存为 <split>/gt_custom/<img_name>.npy，
                每个类别还会输出对应的二值图可视化结果，文件名为 <split>/gt_custom/<img_name>_s<class_indx>_binary.png。
            检测图保存为 <split>/gt_custom/<img_name>_all.npy，
                对应的整体检测二值图可视化保存为 <split>/gt_custom/<img_name>_binary.png。
    """

    if not os.path.isfile(annotations_csv):
        raise FileNotFoundError(f"未找到点标注文件: {annotations_csv}")

    points_by_image = load_points_by_image(annotations_csv)

    for split in DATA_SPLITS:
        in_img_dir = os.path.join(in_root_dir, split, "images")
        out_gt_dir = os.path.join(in_root_dir, split, "gt_custom")
        if not os.path.isdir(in_img_dir):
            print(f"跳过 split={split}，图像目录不存在: {in_img_dir}")
            continue
        os.makedirs(out_gt_dir, exist_ok=True)

        img_files = []
        for pattern in IMAGE_GLOB_PATTERNS:
            img_files.extend(glob.glob(os.path.join(in_img_dir, pattern)))
        img_files = sorted(img_files)

        for img_filepath in img_files:
            print("img_filepath", img_filepath)

            rel_image_path = os.path.relpath(img_filepath, in_root_dir).replace("\\", "/")
            out_dir = out_gt_dir

            # 读取图像文件。
            img_name = os.path.splitext(os.path.basename(img_filepath))[0]
            out_gt_dmap_filepath = os.path.join(out_dir, img_name + ".npy")
            img = io.imread(img_filepath)[:, :, 0:3]

            # 从 CSV 读取细胞中心坐标与类别，并按图像缩放比例同步缩放坐标。
            point_entries = points_by_image.get(rel_image_path, [])
            if point_entries:
                centroids = (
                    np.asarray([(x, y) for x, y, _ in point_entries]) * img_scale
                ).astype(int)
                class_types = np.asarray([label_id for _, _, label_id in point_entries])
            else:
                centroids = np.zeros((0, 2), dtype=int)
                class_types = np.asarray([], dtype=int)
            class_types = np.atleast_1d(class_types)
            # centroids: [N, 2]，每一行表示一个细胞中心点坐标，顺序为 (x, y)。
            # class_types: [N]，存储每个细胞点对应的类别编号。

            # 缩放图像，并保留一份副本用于后续绘制可视化标注。img2是原始图像的缩放
            img2 = cv2.resize(
                img,
                (
                    int(img.shape[1] * img_scale + 0.5),
                    int(img.shape[0] * img_scale + 0.5),
                ),
            )
            img3 = img2.copy()  # img3是img2的复制

            # 初始化原始类别的点标注张量。
            # 形状为 H x W x C，通道索引与 label_id 一致。
            patch_label_arr_dots = np.zeros(
                (img2.shape[0], img2.shape[1], n_grouped_class_channels), dtype=np.uint8
            )  # [H, W, C]

            if len(centroids) > 0:
                centroids[(np.where(centroids[:, 1] >= img2.shape[0]), 1)] = (
                    img2.shape[0] - 1
                )
                centroids[(np.where(centroids[:, 0] >= img2.shape[1]), 0)] = (
                    img2.shape[1] - 1
                )

            # 生成原始类别层面的分类点标注图。
            # 这里每个细胞中心只在对应位置写入 1，其余位置为 0。
            for dot_class in range(1, classes_max_indx + 1):
                patch_label_arr = np.zeros((img2.shape[0], img2.shape[1]))  # [H, W]
                class_mask = class_types == dot_class
                if np.any(class_mask):
                    patch_label_arr[
                        (centroids[class_mask][:, 1], centroids[class_mask][:, 0])
                    ] = 1
                patch_label_arr_dots[:, :, dot_class] = patch_label_arr

            # 将原始类别按照 class_group_mapping_dict 合并（MoNuSAC 为 1:1，逻辑保持不变）。
            # 同时构建可视化时使用的彩色覆盖图 img3。
            patch_label_arr_dots_grouped = np.zeros(
                (img2.shape[0], img2.shape[1], n_grouped_class_channels), dtype=np.uint8
            )  # [H, W, C]
            for class_id, map_class_lst in class_group_mapping_dict.items():
                patch_label_arr = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
                # 用卷积把点适度扩张，便于在可视化图中更清楚地看到标注位置。
                patch_label_arr = ndimage.convolve(
                    patch_label_arr, np.ones((9, 9)), mode="constant", cval=0.0
                )
                img3[np.where(patch_label_arr > 0)] = color_set[class_id]
                patch_label_arr_dots_grouped[:, :, class_id] = patch_label_arr_dots[
                    :, :, map_class_lst
                ].sum(axis=-1)
            patch_label_arr_dots = patch_label_arr_dots_grouped  # [H, W, C]

            # 可选步骤：移除局部邻域内的重复点标注。
            # 适用于同一细胞被重复点击标注、导致局部多个相邻点同时存在的情况。
            if remove_duplicates:
                for c in range(patch_label_arr_dots.shape[-1]):
                    tmp = ndimage.convolve(
                        patch_label_arr_dots[:, :, c],
                        np.ones((5, 5)),
                        mode="constant",
                        cval=0.0,
                    )  # 每个类别的细胞中心点，做5*5常数卷积，如果结果大于1，说明有重复
                    duplicate_points = np.where(tmp > 1)
                    while len(duplicate_points[0]) > 0:
                        y = duplicate_points[0][0]
                        x = duplicate_points[1][0]
                        patch_label_arr_dots[
                            max(0, y - 2) : min(
                                patch_label_arr_dots.shape[0] - 1, y + 3
                            ),
                            max(0, x - 2) : min(
                                patch_label_arr_dots.shape[1] - 1, x + 3
                            ),
                            c,
                        ] = 0
                        patch_label_arr_dots[y, x, c] = 1
                        tmp = ndimage.convolve(
                            patch_label_arr_dots[:, :, c],
                            np.ones((5, 5)),
                            mode="constant",
                            cval=0.0,
                        )
                        duplicate_points = np.where(tmp > 1)

            # 通过对各类别点图求和，得到整体检测任务使用的点标注图。
            patch_label_arr_dots_all = patch_label_arr_dots[:, :, :].sum(axis=-1)  # [H, W]
            # KEY：保存分类点图和检测点图。
            patch_label_arr_dots.astype(np.uint8).dump(
                os.path.join(out_dir, img_name + "_gt_dots.npy")
            )  # 每个像素位置一个类别独热向量
            patch_label_arr_dots_all.astype(np.uint8).dump(
                os.path.join(out_dir, img_name + "_gt_dots_all.npy")
            )  # 每个像素位置一个值，是各个类别掩码的和

            # 生成带有点标注叠加的图像可视化结果。
            # 为了让单点更明显，这里再做一次小范围卷积扩张后上色（仅为了可视化）
            for dot_class in range(1, patch_label_arr_dots.shape[-1]):
                print("dot_class", dot_class)
                print(
                    "patch_label_arr_dots[:,:,dot_class]",
                    patch_label_arr_dots[:, :, dot_class].sum(),
                )
                patch_label_arr = patch_label_arr_dots[:, :, dot_class].astype(int)
                patch_label_arr = ndimage.convolve(
                    patch_label_arr, np.ones((5, 5)), mode="constant", cval=0.0
                )
                if dot_class in color_set:
                    img2[np.where(patch_label_arr > 0)] = color_set[dot_class]
            # KEY：保存彩色点标注叠加图
            io.imsave(os.path.join(out_dir, img_name + "_img_with_dots.jpg"), img2)

            # 生成高斯图/二值掩码图。
            # 这里不能按类别分别独立生成后再简单相加，否则可能导致检测图中不同类别区域互相重叠不一致。
            mat_s_points = np.where(patch_label_arr_dots > 0)  # 二值掩码
            points = np.zeros(
                (len(mat_s_points[0]), 2)
            )  # points: [N, 2]，存储每个细胞像素的坐标（Y，X）
            print(points.shape)
            points[:, 0] = mat_s_points[1]
            points[:, 1] = mat_s_points[0]
            patch_label_arr_dots_custom_all, patch_label_arr_dots_custom = (
                gaussian_filter_density(
                    img2, points, patch_label_arr_dots, out_gt_dmap_filepath
                )
            )
