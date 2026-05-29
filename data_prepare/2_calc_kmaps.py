import numpy as np
import os
import sys;
import skimage.io as io
from skimage.measure import label
import glob
sys.path.append("..")
from spatial_analysis_utils_v2_sh import *

# 为每个细胞计算 cross K-function，
# 并将该细胞对应的 K-function 值传播到真值膨胀点图/二值掩码中该细胞所属连通域的所有像素上。

# 配置变量。

# 配置数据输入/输出路径。
root_dir = '../data/MoNuSAC_point_MCSpatNet'
image_dir= os.path.join(root_dir, 'images')
gt_dir= os.path.join(root_dir, 'gt_custom')
out_dir = os.path.join(root_dir, 'k_func_maps')

# 配置 K function 计算参数。
do_k_correction=True
n_classes = 4
r_step = 15 # r 表示半径采样步长。
r_range = range(0, 100, r_step)
r_list = [*r_range]
r_classes = len(r_range)
r_classes_all = r_classes * (n_classes) # 所有类别拼接后的 K-function 通道总数。

if __name__ == "__main__":
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    img_path_list = glob.glob(os.path.join(image_dir, '*.png'))

    # 对每张图像中的每个细胞计算 cross K-function，
    # 并将该值写入该细胞在真值掩码中所属连通域的所有像素。
    for img_path in img_path_list:
        # 读取真值点图和真值二值掩码。
        print('img', img_path )
        img_name = os.path.basename(img_path)
        gt_path = os.path.join(gt_dir,img_name.replace('.png','_gt_dots.npy'))  
        gt_dots=np.load(gt_path, allow_pickle=True)[:,:,1:].squeeze()   # [H, W, C]，细胞中心点图
        gt_dmap_path = os.path.join(gt_dir,img_name.replace('.png','.npy'))
        gt_dmap=np.load(gt_dmap_path, allow_pickle=True)[:,:,1:].squeeze()  # [H, W, C]，细胞中心高斯密度点图（膨胀）
        gt_dots_all = gt_dots.max(-1) # 各类别点图按通道取最大，得到整体检测点图。
        gt_dmap = gt_dmap > 0   # 为防止传入的是密度图（非二值），对其进行二值化
        gt_dmap_all = gt_dmap.max(-1) # 合并所有类别后的整体检测掩码。
        gt_dmap_all_comp = label(gt_dmap_all) # 对整体掩码做连通域标记，便于后续把 K-function 传播给该细胞区域内所有像素。
        gt_kmap_out_path = os.path.join(out_dir,img_name.replace('.png','_gt_kmap.npy')); # 输出文件路径。
        k_area = gt_dots.shape[0]*gt_dots.shape[1] # 当前图像面积，用于对 K 值做面积归一化。

        # cells_y、cells_x 用于保存所有细胞点坐标，cells_mark 用于保存每个点的类别标记。
        # 第 0 个位置预留给“当前中心细胞”，计算每个细胞的 K-function 时都会动态替换这个位置。
        # 这里将中心细胞标记成特殊类别 '1000'，这样在调用 R 中的 Kcross 时可以把它当作 i 类。
        cells_y=[0]
        cells_x=[0]
        cells_mark=['1000']

        # KEY：将所有真实细胞的坐标和类别依次写入 cells_y、cells_x、cells_mark。
        for c in range(n_classes):
            c_points = np.where(gt_dots[:,:, c] > 0)
            if(len(c_points[0])>0):
                cells_y = np.concatenate((cells_y, c_points[0]))
                cells_x = np.concatenate((cells_x, c_points[1]))
                cells_mark =  cells_mark + [str(c+1)]*len(c_points[0])

        # 初始化输出的 kmap，形状为 [H, W, Ck]。k是取得不同的半径得数量
        # 对于图中每个细胞，其所在连通域区域内的像素都会被赋同一个 K-function 向量。
        gt_kmap = np.zeros((gt_dots.shape[0],gt_dots.shape[1],r_classes_all))

        # c_points 表示所有细胞中心点，也就是之后逐个作为中心细胞计算 K-function 的目标点。
        c_points = np.where(gt_dots_all > 0)
        if(len(c_points[0]) == 0):
            continue

        '''
            KEY：
            遍历 c_points 中的每一个细胞：
                将 cells_y 和 cells_x 的第一个位置替换为当前中心细胞坐标。
                分别计算该中心细胞相对于各个类别的 K-function。
                将得到的 K-function 向量写入当前细胞对应连通域中的所有像素。
        '''
        for ci in range(len(c_points[0])):
            cy = c_points[0][ci]
            cx = c_points[1][ci]
            comp_indx = gt_dmap_all_comp[cy,cx]
            cells_y[0] = cy
            cells_x[0] = cx
            cells_ppp = ppp(cells_x, cells_y, cells_mark) # 构造传给 R/spatstat 的点模式对象。
            k_indx = 0
            c_k_func = np.zeros(r_classes_all)
            for s2 in range(n_classes):
                if(gt_dots[:,:, s2].sum() > 0):
                    if(do_k_correction):
                        r_Kcross, K_val_samp = Kcross(cells_ppp, i='1000', j=str(s2+1), correction='iso', plot=False, r=r_range)
                    else:
                        r_Kcross, K_val_samp = Kcross(cells_ppp, i='1000', j=str(s2+1), correction='none', plot=False, r=r_range)
                    c_k_func[k_indx:k_indx+r_classes] = K_val_samp/k_area * gt_dots[:,:, s2].sum()
                else:
                    c_k_func[k_indx:k_indx+r_classes] = 0
                k_indx += r_classes
            # 将当前中心细胞的 K-function 向量复制到其连通域区域中的所有像素。
            # 这样后续训练时，网络在该细胞区域任一点都能看到相同的空间上下文监督。
            gt_kmap[gt_dmap_all_comp == comp_indx] = c_k_func
        gt_kmap.dump(gt_kmap_out_path)

