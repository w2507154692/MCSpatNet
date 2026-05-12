"""
对CoNSeP数据集进行处理
"""

import numpy as np
import glob
import os
import sys
import skimage.io as io
from scipy import ndimage
import scipy.io as sio
import cv2
import scipy

# 配置变量。
in_root_dir = '../../MCSpatNet_datasets/CoNSeP/Train'
out_root_dir = '../../MCSpatNet_datasets/CoNSeP_train'
# 原始类别索引的最大值。这里类别索引从 1 开始，0 通常作为背景保留。
classes_max_indx = 8
# 可视化时使用的颜色映射：炎症细胞为蓝色，上皮/肿瘤细胞为红色，基质细胞为绿色。
color_set = {1: (0, 162, 232), 2: (255, 0, 0), 3: (0, 255, 0)}
# 定义输入数据集原始类别到输出分组类别的映射关系。
# 输出只保留三大类：1=炎症，2=上皮，3=基质。
class_group_mapping_dict = {1:[2],2:[3,4],3:[1,5,6,7]}
n_grouped_class_channels = 4    # 3个类别加上背景
# 图像缩放比例。原始 patch 与细胞中心坐标会按相同比例缩小。
img_scale = 0.5
remove_duplicates = False  # 若为 True，则去除 5 像素邻域内重复标注的细胞点。
'''
原始细胞类别：
          1 = other 
	      2 = inflammatory 
	      3 = healthy epithelial 
	      4 = dysplastic/malignant epithelial 
          5 = fibroblast 
          6 = muscle  
	      7 = endothelial 
分组后的细胞类别： 
	      1 = inflammatory（天蓝色）
	      2 = All epithelial（healthy epithelial + dysplastic/malignant epithelial，红色）
          3 = All stromal（fibroblast + muscle + endothelial + other，绿色）
'''

"""
    本脚本假设输入数据满足以下目录结构和标注格式：
        - 在 <in_root_dir> 下： 
            Images 文件夹：存放该 slide 切分得到并完成标注的图像 patch。
            Labels 文件夹：存放与 Images 中每个 patch 对应的 mat 标注文件。
        - 每个 mat 文件中包含以下变量： 
            inst_centroid：形状为 n x 2 的数组，n 为细胞数量，每个坐标按 (x, y) 存储。
            inst_type：长度为 n 的数组，表示 inst_centroid 中每个细胞对应的类别编号。
                       类别编号从 1 开始连续编号，因此 color_set 的键也从 1 开始定义。

"""


def gaussian_filter_density(img, points, point_class_map, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1):
    '''
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
    '''
    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)     # density: [H, W]
    density_class = np.zeros((img.shape[0], img.shape[1], point_class_map.shape[2]), dtype=np.float32)  # density_class: [H, W, C]
    if (end_y <= 0):
        end_y = img.shape[0]
    if (end_x <= 0):
        end_x = img.shape[1]
    gt_count = len(points)
    if gt_count == 0:
        return density
    leafsize = 2048
    # 构建 KD-tree，用于快速查询每个点的最近邻距离。
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # 查询每个点最近的两个邻居。第一个通常是其自身，第二个是真正的最近邻。
    distances, locations = tree.query(points, k=2)
    print('generate density...')

    max_sigma = 2;  # kernel size = 4, kernel_width=9

    for i, pt in enumerate(points): # pt是坐标
        pt2d = np.zeros(img_shape, dtype=np.float32)    # pt2d: [H, W, 1]，将坐标在图上标记出来
        if (pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x):
            continue
        pt[1] -= start_y
        pt[0] -= start_x
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.   # 将坐标在图上标记出来
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]) * 0.125   # 第 i 个细胞的最近邻
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
        pnt_density = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant', truncate=2)
        # 归一化后再累加，使每个细胞点对总密度图的贡献一致。
        pnt_density /= pnt_density.sum()
        density += pnt_density
        class_indx = point_class_map[int(pt[1]), int(pt[0])].argmax()   # 取出当前点对应的类别
        density_class[:, :, class_indx] = density_class[:, :, class_indx] + pnt_density

    #density_class.astype(np.float16).dump(out_filepath)
    #density.astype(np.float16).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
    # KEY：保存的是每个类别的高斯密度图（二值化）（可以说是膨胀点图），名称如train_1.npy
    (density_class > 0).astype(np.uint8).dump(out_filepath)
    # KEY：保存的是所有类别总的高斯密度图（二值化），名称如train_1_all.npy
    (density > 0).astype(np.uint8).dump(os.path.splitext(out_filepath)[0] + '_all.npy')     # 图像名_all.npy，保存的是所有类别总的高斯密度图
    #io.imsave(out_filepath.replace('.npy', '.png'), (density / density.max() * 255).astype(np.uint8))
    # KEY：保存所有类别总的高斯密度图（二值化），不过是png格式（供可视化），名称如train_1_binary.png
    io.imsave(out_filepath.replace('.npy', '_binary.png'), ((density > 0) * 255).astype(np.uint8))
    # KEY：保存每个类别的高斯密度图（二值化），png格式，名称如train_1_s0_binary.png
    for s in range(1, density_class.shape[-1]):
        io.imsave(out_filepath.replace('.npy', '_s' + str(s) + '_binary.png'),
                  ((density_class[:, :, s] > 0) * 255).astype(np.uint8))
    print('done.')
    return density.astype(np.float16), density_class.astype(np.float16)


if __name__ == "__main__":
    '''
        对每张图像，脚本会执行以下处理： 
            1. 对 patch 图像和标注的细胞中心坐标做统一缩放。
                缩放后的图像保存为 (<out_img_dir>/<img_name>.png)。
                同时会生成一个彩色可视化图，将不同类别的细胞点叠加到图像上，保存为 <out_gt_dir>/<img_name>_img_with_dots.jpg。
            2. 生成分类点标注图，保存为 <out_gt_dir>/<img_name>_gt_dots.npy。
            3. 生成检测点标注图，保存为 <out_gt_dir>/<img_name>_gt_dots_all.npy。
            4. 以每个细胞中心为中心生成高斯图，并自适应设置高斯宽度，以尽量避免相邻细胞区域过度相交。
            5. 最终将高斯图转为二值掩码保存，即所有 >0 的像素置为 1，其余位置为 0。
                分类图保存为 <out_gt_dir>/<img_name>.npy，
                    每个类别还会输出对应的二值图可视化结果，文件名为 <out_gt_dir>/<img_name>_s<class_indx>_binary.png。
                检测图保存为 <out_gt_dir>/<img_name>_all.npy，
                    对应的整体检测二值图可视化保存为 <out_gt_dir>/<img_name>_binary.png。

    '''
    '''
        每个 .mat 标注文件中包含以下键：
        'inst_type'
        'inst_centroid'
    '''

    # 输入图像和标签
    in_img_dir = os.path.join(in_root_dir, 'Images')
    in_label_dir = os.path.join(in_root_dir, 'Labels')

    # 输出的图像和标签路径
    out_img_dir = os.path.join(out_root_dir, 'images')
    out_gt_dir = os.path.join(out_root_dir, 'gt_custom')
    if not os.path.exists(out_root_dir):
        os.mkdir(out_root_dir)
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    if not os.path.exists(out_gt_dir):
        os.mkdir(out_gt_dir)

    img_files = glob.glob(os.path.join(in_img_dir, '*.png'))

    for img_filepath in img_files:
        print('img_filepath', img_filepath)

        # 读取图像文件。
        img_name = os.path.splitext(os.path.basename(img_filepath))[0]  # 获取图像名（无后缀，无前面的目录）
        out_gt_dmap_filepath = os.path.join(out_gt_dir, img_name  + '.npy')
        img = io.imread(img_filepath)[:, :, 0:3]

        # 读取与当前图像对应的 mat 标注文件。
        mat_filepath = os.path.join(in_label_dir, img_name + '.mat')
        mat = sio.loadmat(mat_filepath)

        # 读取细胞中心坐标与类别，并按图像缩放比例同步缩放坐标。
        centroids = (mat["inst_centroid"] * img_scale).astype(int)
        class_types = mat["inst_type"].squeeze()
        # centroids: [N, 2]，每一行表示一个细胞中心点坐标，顺序为 (x, y)。
        # class_types: 通常为 [N]，存储每个细胞点对应的原始类别编号；若图中只有一个细胞，squeeze 后可能退化为标量。

        # 缩放图像，并保留一份副本用于后续绘制可视化标注。img2是原始图像的缩放
        img2 = cv2.resize(img, (int(img.shape[1] * img_scale + 0.5), int(img.shape[0] * img_scale + 0.5)))
        img3 = img2.copy()  # img3是img2的复制
        io.imsave(os.path.join(out_img_dir, img_name+'.png'), img2)     # 保存原始图像的缩放图，文件名.png

        # 初始化原始类别的点标注张量。
        # 形状为 H x W x C，其中每个通道对应一个原始类别的点图。
        patch_label_arr_dots = np.zeros((img2.shape[0], img2.shape[1], classes_max_indx), dtype=np.uint8)   # [H, W, C]

        # 缩放后再次约束坐标，防止坐标落到图像边界之外。
        # print('centroids',centroids.shape)
        # print('class_types',class_types.shape)
        centroids[(np.where(centroids[:, 1] >= img2.shape[0]), 1)] = img2.shape[0] - 1
        centroids[(np.where(centroids[:, 0] >= img2.shape[1]), 0)] = img2.shape[1] - 1

        # 生成原始类别层面的分类点标注图。
        # 这里每个细胞中心只在对应位置写入 1，其余位置为 0。
        for dot_class in range(1, classes_max_indx):    # 注意这里是按所有类别算的
            patch_label_arr = np.zeros((img2.shape[0], img2.shape[1]))  # [H, W]
            patch_label_arr[(centroids[np.where(class_types == dot_class)][:, 1],
                                centroids[np.where(class_types == dot_class)][:, 0])] = 1
            patch_label_arr_dots[:, :, dot_class] = patch_label_arr
            #patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((5, 5)), mode='constant', cval=0.0)
            # img2[np.where(patch_label_arr > 0)] = color_set[dot_class]

        # 将原始类别按照 class_group_mapping_dict 合并为三大类。
        # 同时构建可视化时使用的彩色覆盖图 img3。
        patch_label_arr_dots_grouped = np.zeros((img2.shape[0], img2.shape[1], n_grouped_class_channels), dtype=np.uint8)   # [H, W, C]
        for class_id, map_class_lst in class_group_mapping_dict.items():
            patch_label_arr = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
            # 用卷积把点适度扩张，便于在可视化图中更清楚地看到标注位置。
            patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((9, 9)), mode='constant', cval=0.0)
            img3[np.where(patch_label_arr > 0)] = color_set[class_id]
            patch_label_arr_dots_grouped[:, :, class_id] = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
        patch_label_arr_dots = patch_label_arr_dots_grouped     # [H, W, C]

        # 可选步骤：移除局部邻域内的重复点标注。
        # 适用于同一细胞被重复点击标注、导致局部多个相邻点同时存在的情况。
        if (remove_duplicates):
            for c in range(patch_label_arr_dots.shape[-1]):
                tmp = ndimage.convolve(patch_label_arr_dots[:, :, c], np.ones((5, 5)), mode='constant', cval=0.0)   # 每个类别的细胞中心点，做5*5常数卷积，如果结果大于1，说明有重复
                duplicate_points = np.where(tmp > 1)
                while (len(duplicate_points[0]) > 0):
                    y = duplicate_points[0][0]
                    x = duplicate_points[1][0]
                    patch_label_arr_dots[max(0, y - 2):min(patch_label_arr_dots.shape[0] - 1, y + 3),
                    max(0, x - 2):min(patch_label_arr_dots.shape[1] - 1, x + 3), c] = 0
                    patch_label_arr_dots[y, x, c] = 1
                    tmp = ndimage.convolve(patch_label_arr_dots[:, :, c], np.ones((5, 5)), mode='constant',
                                            cval=0.0)
                    duplicate_points = np.where(tmp > 1)

        # 通过对各类别点图求和，得到整体检测任务使用的点标注图。
        patch_label_arr_dots_all = patch_label_arr_dots[:, :, :].sum(axis=-1)   # [H, W]
        # KEY：保存分类点图和检测点图。
        patch_label_arr_dots.astype(np.uint8).dump(
            os.path.join(out_gt_dir, img_name + '_gt_dots.npy'))    # 每个像素位置一个类别独热向量
        patch_label_arr_dots_all.astype(np.uint8).dump(
            os.path.join(out_gt_dir, img_name + '_gt_dots_all.npy'))    # 每个像素位置一个值，是各个类别掩码的和

        # 生成带有点标注叠加的图像可视化结果。
        # 为了让单点更明显，这里再做一次小范围卷积扩张后上色（仅为了可视化）
        for dot_class in range(1, patch_label_arr_dots.shape[-1]):
            print('dot_class', dot_class)
            print('patch_label_arr_dots[:,:,dot_class]', patch_label_arr_dots[:, :, dot_class].sum())
            patch_label_arr = patch_label_arr_dots[:, :, dot_class].astype(int)
            patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((5, 5)), mode='constant', cval=0.0)
            img2[np.where(patch_label_arr > 0)] = color_set[dot_class]
        # KEY：保存彩色三类图
        io.imsave(os.path.join(out_gt_dir, img_name + '_img_with_dots.jpg'), img2)  # 彩色三类图

        # 生成高斯图/二值掩码图。
        # 这里不能按类别分别独立生成后再简单相加，否则可能导致检测图中不同类别区域互相重叠不一致。
        mat_s_points = np.where(patch_label_arr_dots > 0)   # 二值掩码
        points = np.zeros((len(mat_s_points[0]), 2))    # points: [N, 2]，存储每个细胞像素的坐标（Y，X）
        print(points.shape)
        points[:, 0] = mat_s_points[1]
        points[:, 1] = mat_s_points[0]
        patch_label_arr_dots_custom_all, patch_label_arr_dots_custom = gaussian_filter_density(img2, points, patch_label_arr_dots, out_gt_dmap_filepath)

