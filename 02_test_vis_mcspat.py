import os
import numpy as np
from skimage import io
import cv2
import sys
from skimage.measure import label, moments
from skimage import filters
from tqdm import tqdm as tqdm
import torch
import torch.nn as nn
import glob

from model_arch import UnetVggMultihead
from my_dataloader import CellsDataset

# # 所有训练输出的根目录。
# checkpoints_root_dir = './exp'
# # 当前实验对应的子目录名称，最终模型权重会从 <checkpoints_root_dir>/<checkpoints_folder_name> 下读取。
# checkpoints_folder_name = 'exp7_brcam2c'
# eval_root_dir = './exp'
# epoch=5 # 需要测试的 epoch。
# visualize=True # 是否输出预测结果的可视化图像。
# test_data_root = './data/BRCA-M2C'
# test_split_filepath = './data_splits/brca-m2c/test_split.txt'
# ----------------------------------------------------------------
# 所有训练输出的根目录。
checkpoints_root_dir = "./exp"
# 当前实验对应的子目录名称，最终模型权重会从 <checkpoints_root_dir>/<checkpoints_folder_name> 下读取。
checkpoints_folder_name = "exp9_monusac"
eval_root_dir = "./exp"
epoch = 282  # 需要测试的 epoch。
visualize = True  # 是否输出预测结果的可视化图像。
test_data_root = "./data/MoNuSAC_point_MCSpatNet"
test_split_filepath = None

if __name__ == "__main__":

    # ------------------------------
    # 初始化基础配置
    # ------------------------------

    # 类别颜色定义，用于可视化分类结果。
    # 0: 淋巴细胞，对应蓝色
    # 1: 肿瘤细胞，对应红色
    # 2: 其他细胞，对应绿色
    color_set = {
        0: (0, 162, 232),
        1: (255, 0, 0),
        2: (0, 255, 0),
    }

    # 模型权重目录与测试结果输出目录。
    models_root_dir = os.path.join(checkpoints_root_dir, checkpoints_folder_name)
    out_dir = os.path.join(eval_root_dir, checkpoints_folder_name + f"_e{epoch}")

    if not os.path.exists(eval_root_dir):
        os.mkdir(eval_root_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 测试数据路径配置。
    # images: 原始图像
    # gt_custom: 检测/分类标签
    test_image_root = os.path.join(test_data_root, "images")
    test_dmap_root = os.path.join(test_data_root, "gt_custom")
    test_dots_root = os.path.join(test_data_root, "gt_custom")

    # ------------------------------
    # 模型相关参数
    # ------------------------------
    gt_multiplier = 1
    gpu_or_cpu = "cuda"  # 指定运行设备，原脚本默认使用 cuda。
    dropout_prob = 0
    initial_pad = 126  # 为保证 U-Net 边界对齐所做的初始 padding。
    interpolate = "False"  # 解码阶段是否使用插值。
    conv_init = "he"  # 卷积层初始化方式。
    n_classes = 3  # 主分类头中的细胞类别数。
    n_classes_out = n_classes + 1  # 额外包含背景时的类别总数。
    class_indx = "1,2,3"  # 参与测试的类别编号。
    class_weights = np.array([1, 1, 1])
    n_clusters = 5  # 每个主类别进一步细分的聚类数。
    n_classes2 = n_clusters * (n_classes)  # 子类别总数。

    # K-function 回归头的半径采样配置。
    r_step = 15
    r_range = range(0, 100, r_step)
    r_arr = np.array([*r_range])
    r_classes = len(r_range)
    r_classes_all = r_classes * (n_classes)

    # 检测后处理阈值。
    # 这里使用滞后阈值分割预测热图，再过滤掉过小连通域。
    thresh_low = 0.5
    thresh_high = 0.5
    size_thresh = 5

    # 构建模型、激活函数和测试数据加载器。
    device = torch.device(gpu_or_cpu)
    model = UnetVggMultihead(
        kwargs={
            "dropout_prob": dropout_prob,
            "initial_pad": initial_pad,
            "interpolate": interpolate,
            "conv_init": conv_init,
            "n_classes": n_classes,
            "n_channels": 3,
            "n_heads": 4,
            "head_classes": [1, n_classes, n_classes2, r_classes_all],
        }
    )
    model.to(device)
    criterion_sig = nn.Sigmoid()  # 将检测头输出映射到 0~1。
    criterion_softmax = nn.Softmax(dim=1)  # 将分类头输出转成各类别概率。
    test_dataset = CellsDataset(
        test_image_root,
        test_dmap_root,
        test_dots_root,
        class_indx,
        split_filepath=test_split_filepath,
        phase="test",
        fixed_size=-1,
        max_scale=16,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("thresh", thresh_low, thresh_high)

    # ------------------------------
    # KEY：加载模型权重
    # ------------------------------
    print("test epoch " + str(epoch))
    model_files = glob.glob(
        os.path.join(models_root_dir, "mcspat_epoch_" + str(epoch) + "_*.pth")
    )
    model_files2 = glob.glob(
        os.path.join(models_root_dir, "*epoch_" + str(epoch) + "_*.pth")
    )
    # 兼容两种权重命名方式，优先使用 mcspat_epoch_*.pth。
    if (model_files == None) or (len(model_files) == 0):
        if (model_files2 == None) or (len(model_files2) == 0):
            print("not found ", "mcspat_epoch_" + str(epoch))
            exit()
        else:
            model_param_path = model_files2[0]
    else:
        model_param_path = model_files[0]

    sys.stdout.flush()
    model.load_state_dict(torch.load(model_param_path), strict=True)
    model.to(device)
    model.eval()

    # 整个测试阶段不需要梯度，关闭 autograd 以减少显存与计算开销。
    with torch.no_grad():

        for i, (img, gt_dmap, gt_dots, img_name) in enumerate(
            tqdm(test_loader, disable=True)
        ):
            img_name = img_name[0]
            sys.stdout.flush()

            # ------------------------------
            # 前向推理
            # ------------------------------
            img = img.to(device)
            et_dmap_lst = model(img)
            # 4 个输出头分别对应：
            # 0: 全部细胞的检测图
            # 1: 主类别分类图
            # 2: 子类别分类图
            # 3: K-function 回归图
            # [2:-2, 2:-2] 用于去掉边界 padding 对输出的影响。
            et_dmap_all = et_dmap_lst[0][:, :, 2:-2, 2:-2]
            et_dmap_class = et_dmap_lst[1][:, :, 2:-2, 2:-2]
            et_dmap_subclasses = et_dmap_lst[2][:, :, 2:-2, 2:-2]
            et_kmap = et_dmap_lst[3][:, :, 2:-2, 2:-2] ** 2

            # 将 GT 和预测结果搬回 CPU，并转换成后续 numpy 后处理所需格式。
            gt_dmap = gt_dmap > 0
            gt_dmap_all = gt_dmap.max(1)[0].detach().cpu().numpy()
            gt_dots_all = gt_dots.max(1)[0].detach().cpu().numpy().squeeze()
            gt_dots = gt_dots.detach().cpu().numpy()

            et_all_sig = criterion_sig(et_dmap_all).detach().cpu().numpy()
            et_class_sig = criterion_softmax(et_dmap_class).detach().cpu().numpy()

            # 将输入图像恢复成 HWC 格式并放缩回 0~255，便于直接可视化保存。
            img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0) * 255
            img_centers_all = img.copy()
            img_centers_all_gt = img.copy()

            img_centers_all_all = img.copy()
            img_centers_all_all_gt = img.copy()

            # ------------------------------
            # 检测评估：所有细胞合并检测
            # ------------------------------
            g_count = gt_dots_all.sum()

            # 对检测头概率图做滞后阈值分割，得到二值预测图。
            # 随后提取连通域，并过滤掉面积过小的噪声区域。
            e_hard = filters.apply_hysteresis_threshold(
                et_all_sig.squeeze(), thresh_low, thresh_high
            )
            e_hard2 = (e_hard > 0).astype(np.uint8)
            comp_mask = label(e_hard2)
            e_count = comp_mask.max()
            s_count = 0
            if size_thresh > 0:
                for c in range(1, comp_mask.max() + 1):
                    s = (comp_mask == c).sum()
                    if s < size_thresh:
                        e_count -= 1
                        s_count += 1
                        e_hard2[comp_mask == c] = 0
            e_hard2_all = e_hard2.copy()

            # 从每个预测连通域中提取质心，作为最终的检测点。
            e_dot = np.zeros((img.shape[0], img.shape[1]))
            e_dot_vis = np.zeros((img.shape[0], img.shape[1]))
            contours, hierarchy = cv2.findContours(
                e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for idx in range(len(contours)):
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                if M["m00"] == 0:
                    continue
                cx = round(M["m10"] / M["m00"])
                cy = round(M["m01"] / M["m00"])
                # e_dot_vis 用于显示一个小块区域，e_dot 只保留单个中心点。
                e_dot_vis[cy - 1 : cy + 1, cx - 1 : cx + 1] = 1
                e_dot[min(cy, e_dot.shape[0] - 1), min(cx, e_dot.shape[1] - 1)] = 1
                # 在可视化图像上把预测中心附近涂黑，方便肉眼观察。
                img_centers_all_all[cy - 3 : cy + 3, cx - 3 : cx + 3, :] = (0, 0, 0)
            e_dot_all = e_dot.copy()

            # 把 GT 中的细胞中心也画到另一张图上，便于与预测结果对照。
            gt_centers = np.where(gt_dots_all > 0)
            for idx in range(len(gt_centers[0])):
                cx = gt_centers[1][idx]
                cy = gt_centers[0][idx]
                img_centers_all_all_gt[cy - 3 : cy + 3, cx - 3 : cx + 3, :] = (0, 0, 0)

            # 保存所有细胞的检测中心点图。
            e_dot.astype(np.uint8).dump(
                os.path.join(
                    out_dir, img_name.replace(".png", "_centers" + "_all" + ".npy")
                )
            )
            if visualize:
                # 这里保留原有的注释代码，不改变其存在状态。
                # io.imsave(os.path.join(out_dir, img_name.replace('.png','_centers'+'_allcells' +'.png')), (e_dot_vis*255).astype(np.uint8))
                io.imsave(
                    os.path.join(
                        out_dir,
                        img_name.replace(".png", "_centers" + "_det" + "_overlay.png"),
                    ),
                    (img_centers_all_all).astype(np.uint8),
                )
                # io.imsave(os.path.join(out_dir, img_name.replace('.png','_allcells' +'_hard.png')), (e_hard2*255).astype(np.uint8))

            # 所有细胞的检测评估结束。

            # ------------------------------
            # 分类评估：在检测点基础上做类别划分
            # ------------------------------
            et_class_argmax = et_class_sig.squeeze().argmax(axis=0)
            e_hard2_all = e_hard2.copy()

            for s in range(n_classes):
                g_count = gt_dots[0, s, :, :].sum()

                e_hard2 = et_class_argmax == s

                # 用当前类别的 argmax 掩码去筛选总检测点图，得到该类别的预测中心点。
                e_dot = e_hard2 * e_dot_all
                e_count = e_dot.sum()

                g_dot = gt_dots[0, s, :, :].squeeze()
                e_dot_vis = np.zeros(g_dot.shape)
                e_dots_tuple = np.where(e_dot > 0)
                for idx in range(len(e_dots_tuple[0])):
                    cy = e_dots_tuple[0][idx]
                    cx = e_dots_tuple[1][idx]
                    # 在分类可视化图上按类别颜色绘制预测中心。
                    img_centers_all[cy - 3 : cy + 3, cx - 3 : cx + 3, :] = color_set[s]

                gt_centers = np.where(g_dot > 0)
                for idx in range(len(gt_centers[0])):
                    cx = gt_centers[1][idx]
                    cy = gt_centers[0][idx]
                    # 在 GT 可视化图上按类别颜色绘制真实中心。
                    img_centers_all_gt[cy - 3 : cy + 3, cx - 3 : cx + 3, :] = color_set[
                        s
                    ]

                # 分类别保存检测中心点。
                e_dot.astype(np.uint8).dump(
                    os.path.join(
                        out_dir,
                        img_name.replace(".png", "_centers" + "_s" + str(s) + ".npy"),
                    )
                )
                # if(visualize):
                #    io.imsave(os.path.join(out_dir, img_name.replace('.png','_likelihood_s'+ str(s)+'.png')), (et_class_sig.squeeze()[s]*255).astype(np.uint8));
            # 分类评估结束。

            # 保存原始概率图与 GT 点图，便于后续评估和复现实验。
            et_class_sig.squeeze().astype(np.float16).dump(
                os.path.join(
                    out_dir, img_name.replace(".png", "_likelihood_class" + ".npy")
                )
            )
            et_all_sig.squeeze().astype(np.float16).dump(
                os.path.join(
                    out_dir, img_name.replace(".png", "_likelihood_all" + ".npy")
                )
            )
            gt_dots.squeeze().astype(np.uint8).dump(
                os.path.join(
                    out_dir, img_name.replace(".png", "_gt_dots_class" + ".npy")
                )
            )
            gt_dots_all.squeeze().astype(np.uint8).dump(
                os.path.join(out_dir, img_name.replace(".png", "_gt_dots_all" + ".npy"))
            )
            if visualize:
                # 保存分类预测叠加图、GT 叠加图以及原图。
                io.imsave(
                    os.path.join(
                        out_dir,
                        img_name.replace(
                            ".png", "_centers" + "_class_overlay" + ".png"
                        ),
                    ),
                    (img_centers_all).astype(np.uint8),
                )
                io.imsave(
                    os.path.join(
                        out_dir,
                        img_name.replace(
                            ".png", "_gt_centers" + "_class_overlay" + ".png"
                        ),
                    ),
                    (img_centers_all_gt).astype(np.uint8),
                )
                io.imsave(os.path.join(out_dir, img_name), (img).astype(np.uint8))
                # io.imsave(os.path.join(out_dir, img_name.replace('.png','_likelihood_all'+'.png')), (et_all_sig.squeeze()*255).astype(np.uint8));

            # 主循环末尾释放大数组，减小长时间测试时的内存占用。
            del img, gt_dots
