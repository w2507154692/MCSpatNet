import os
import numpy as np
import glob
import sys
import scipy.spatial
from skimage import io;
from scipy.ndimage.filters import convolve   
from scipy import ndimage

# 配置参数
# data_dir 目录中同时包含真实标注和预测结果，默认已经先运行过 02_test_vis_mcspat.py 生成预测文件
data_dir = './exp/exp2_brcam2c_e181/'

max_dist_thresh = 6 # 在 1 到 max_dist_thresh 的像素距离阈值范围内计算 F-score；40x 下 mpp=0.254，20x 下 mpp=0.508，因此 6 px 约等于 3.048 微米，30 px 约等于 15.24 微米
color_set = {'tp':(0,162,232),'fp':(0,255,0),'fn':(255,255,0)} 

def calc(g_dot, e_dot, class_indx, img_indx, img_name):
    '''
        统计指定类别在不同距离阈值下的 TP、FP、FN 数量。
        对于阈值 t，只要某个预测点与一个尚未匹配过的真实点距离不超过 t 像素，
        就将该预测点记为一个 TP。
    '''
    leafsize = 2048
    k = 50
    e_coords = np.where(e_dot > 0)
    # 根据预测得到的细胞中心点构建 KDTree，用于快速查找阈值范围内的最近邻
    z = np.zeros((len(e_coords[0]),2))
    z[:,0] = e_coords[0]
    z[:,1] = e_coords[1]
    if(len(e_coords[0]) > 0):
        tree = scipy.spatial.KDTree(z, leafsize=leafsize)
        print('tree.data.shape', tree.data.shape)

    for dist_thresh in range(1,max_dist_thresh+1):
        img_f = np.zeros((e_dot.shape[0],e_dot.shape[1],3))
        print('class_indx', class_indx, 'thresh', dist_thresh, 'len(e_coords[0])',len(e_coords[0]))
        if(len(e_coords[0]) == 0): # 当前图像没有任何预测点，所有真实点都将计为漏检
            for dist_thresh in range(1,max_dist_thresh+1):
                tp_img = 0
                fn_img = (g_dot > 0).sum()
                fp_img = 0
                fn[class_indx, dist_thresh] += fn_img
        else:
            tp_img = 0
            fn_img = 0
            fp_img = 0

            e_dot_processing = np.copy(e_dot)

            gt_points = np.where(g_dot > 0)
            ''' 
                依次遍历每个真实点，并在当前阈值内查找最近的预测点。
                如果找到的预测点尚未被使用，则记为 TP，
                同时从 e_dot_processing 中移除，保证每个预测点最多只匹配一次。
                如果没有找到有效匹配，则当前真实点记为 FN。
                最终 e_dot_processing 中剩余的预测点都记为 FP。
            '''
            for pi in range(len(gt_points[0])):
                p = [[gt_points[0][pi], gt_points[1][pi]]]
                distances, locations = tree.query(p, k=k,distance_upper_bound =dist_thresh)
                match = False
                for nn in range(min(k,len(locations[0]))):
                    if((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) and (e_dot_processing[int(tree.data[locations[0][nn]][0]),int(tree.data[locations[0][nn]][1])] > 0)):
                    #if((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) ):
                        tp[class_indx, dist_thresh] += 1
                        tp_img +=1
                        e_dot_processing[int(tree.data[locations[0][nn]][0]),int(tree.data[locations[0][nn]][1])] = 0
                        match = True
                        py = int(tree.data[locations[0][nn]][0])
                        px = int(tree.data[locations[0][nn]][1])
                        img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['tp']
                        break
                if(not match):
                    fn[class_indx, dist_thresh] += 1
                    fn_img +=1
                    py = gt_points[0][pi]
                    px = gt_points[1][pi]
                    img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['fn']

            fp[class_indx, dist_thresh] += e_dot_processing.sum()
            fp_img +=e_dot_processing.sum()
            fp_points = np.where(e_dot_processing > 0)
            for pi in range(len(fp_points[0])):
                py = fp_points[0][pi]
                px = fp_points[1][pi]
                img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['fp']
            #io.imsave(os.path.join(out_dir, img_name+'_s'+str(class_indx)+'_f'+'_th'+str(dist_thresh)+'.png'), img_f.astype(np.uint8))
            print(img_name, 's',class_indx, 'thresh', dist_thresh, 'tp', tp_img, 'fp', fp_img, 'fn', fn_img)
            sys.stdout.flush();

        # 计算当前图像在当前距离阈值下的精确率、召回率和 F1 分数
        if(tp_img + fp_img == 0):
            precision_img[class_indx, dist_thresh, img_indx] = 1
        else:
            precision_img[class_indx, dist_thresh, img_indx] = tp_img/(tp_img + fp_img)
        if(tp_img + fn_img == 0):
            recall_img[class_indx, dist_thresh, img_indx] = 1
        else:
            recall_img[class_indx, dist_thresh, img_indx] = tp_img/(tp_img + fn_img) # 召回率，即真正例率
        if(precision_img[class_indx, dist_thresh, img_indx] + recall_img[class_indx, dist_thresh, img_indx] == 0):
            f1_img[class_indx, dist_thresh, img_indx] = 0
        else:
            f1_img[class_indx, dist_thresh, img_indx] = 2*(( precision_img[class_indx, dist_thresh, img_indx]*recall_img[class_indx, dist_thresh, img_indx] )/( precision_img[class_indx, dist_thresh, img_indx]+recall_img[class_indx, dist_thresh, img_indx] ))


def eval(data_dir, out_dir):
    '''
        默认真实点图和预测结果文件位于同一目录。
        真实点图文件命名格式为 <img name>_gt_dots_class.npy。
        预测点图文件命名格式为：
        分类结果使用 <img name>_centers_s<class indx>.npy，
        检测结果使用 <img name>_centers_allcells.npy。
    '''

    img_indx=-1
    with open(os.path.join(out_dir, 'out_distance_scores.txt'), 'w+') as log_file:
        for gt_filepath in gt_files:
            img_indx += 1
            print('gt_filepath',gt_filepath)
            sys.stdout.flush()
            img_name = os.path.basename(gt_filepath)[:-len('_gt_dots_class.npy')]
            g_dot_arr=np.load(gt_filepath, allow_pickle=True)

            # 逐个细胞类别评估分类定位效果
            for s in range(n_classes):
                e_soft_filepath = glob.glob(os.path.join(data_dir, img_name + '_*'+'centers_s'+str(s)+'.npy'))[0]
                print('e_soft_filepath', e_soft_filepath)
                class_indx = s
                g_dot = g_dot_arr[class_indx]
                #print('e_soft_filepath',e_soft_filepath)
                sys.stdout.flush()
                e_dot = np.load(e_soft_filepath, allow_pickle=True)
                e_dot_vis = ndimage.convolve(e_dot, np.ones((5,5)), mode='constant', cval=0.0)            
                #io.imsave(os.path.join(data_dir,img_name + '_centers_s0_et.png'),(e_dot_vis*255).astype(np.uint8))
                calc(g_dot, e_dot, class_indx, img_indx, img_name)


            # 将所有类别合并后，再评估整体细胞检测效果
            e_soft_filepath = glob.glob(os.path.join(data_dir, img_name + '_*'+'centers_all*.npy'))[0]
            class_indx += 1
            g_dot = g_dot_arr.max(axis=0)
            #print('e_soft_filepath',e_soft_filepath)
            sys.stdout.flush()
            e_dot = np.load(e_soft_filepath, allow_pickle=True)
            calc(g_dot, e_dot, class_indx, img_indx, img_name)



        # tp.astype(np.int).dump(os.path.join(out_dir, 'tp.npy'))
        # fp.astype(np.int).dump(os.path.join(out_dir, 'fp.npy'))
        # fn.astype(np.int).dump(os.path.join(out_dir, 'fn.npy'))

        # 汇总每个类别在各个距离阈值下的精确率、召回率和 F1 分数；
        # class_indx 取值 0 到 n_classes-1 表示分类任务中的各个类别，
        # class_indx = n_classes 表示所有细胞合并后的检测任务
        for class_indx in range(n_classes_out):
            for dist_thresh in range(1,max_dist_thresh+1):
                if(tp[class_indx, dist_thresh] + fp[class_indx, dist_thresh] == 0):
                    precision[class_indx, dist_thresh] = 1
                else:
                    precision[class_indx, dist_thresh] = tp[class_indx, dist_thresh]/(tp[class_indx, dist_thresh] + fp[class_indx, dist_thresh])
                if(tp[class_indx, dist_thresh] + fn[class_indx, dist_thresh] == 0):
                    recall[class_indx, dist_thresh] = 1
                else:
                    recall[class_indx, dist_thresh] = tp[class_indx, dist_thresh]/(tp[class_indx, dist_thresh] + fn[class_indx, dist_thresh]) # 召回率，即真正例率
                if(precision[class_indx, dist_thresh] + recall[class_indx, dist_thresh] == 0):
                    f1[class_indx, dist_thresh] = 0
                else:
                    f1[class_indx, dist_thresh] = 2*((precision[class_indx, dist_thresh]*recall[class_indx, dist_thresh])/(precision[class_indx, dist_thresh]+recall[class_indx, dist_thresh]))

                print('class', class_indx, 'thresh', dist_thresh, 'prec', precision[class_indx, dist_thresh], 'recall', recall[class_indx, dist_thresh], 'fscore',f1[class_indx, dist_thresh])
                log_file.write("class {} thresh {} prec {} recall {} fscore {}\n".format(class_indx, dist_thresh, precision[class_indx, dist_thresh], recall[class_indx, dist_thresh], f1[class_indx, dist_thresh]))


            log_file.flush()

        # 在最大距离阈值下，额外记录三类分类 F1 的平均值，以及三类合并后的整体检测 F1
        mean_f1_max_thresh = np.mean(f1[:n_classes, max_dist_thresh])
        combined_f1_max_thresh = f1[n_classes, max_dist_thresh]
        print('thresh', max_dist_thresh, 'mean_f1_three_classes', mean_f1_max_thresh, 'combined_f1_all_classes', combined_f1_max_thresh)
        log_file.write("thresh {} mean_f1_three_classes {} combined_f1_all_classes {}\n".format(max_dist_thresh, mean_f1_max_thresh, combined_f1_max_thresh))
        log_file.flush()

if __name__ == "__main__":
    out_dir= data_dir # 可按需修改输出目录
    n_classes=3 # 细胞类别数
    n_classes_out = n_classes + 1 # 输出统计同时包含分类结果和整体检测结果

    # 初始化全局统计量
    tp = np.zeros((n_classes_out, max_dist_thresh + 1))
    fp = np.zeros((n_classes_out, max_dist_thresh + 1))
    fn = np.zeros((n_classes_out, max_dist_thresh + 1))
    precision = np.zeros((n_classes_out, max_dist_thresh + 1))
    recall = np.zeros((n_classes_out, max_dist_thresh + 1))
    f1 = np.zeros((n_classes_out, max_dist_thresh + 1))

    gt_files = glob.glob(os.path.join(data_dir, '*_gt_dots_class'+'.npy'))
    #gt_files = glob.glob(os.path.join(data_dir, '*test_1_gt_dots_class'+'.npy'))
    print('len(gt_files)',len(gt_files))

    precision_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))
    recall_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))
    f1_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))

    eval(data_dir, out_dir)



