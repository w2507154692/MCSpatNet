import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm as tqdm
import skimage.io as io
from skimage.measure import label
from sklearn.cluster import KMeans


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collect_features(model, simple_train_loader, feature_indx_list):

    n_samples = len(simple_train_loader.dataset)
    features_list = [[0]*n_samples] # 保存每张输入图像中每个细胞提取到的特征
    coord_list = [[0]*n_samples] # 保存每张输入图像中细胞中心点的坐标
    img_name_list = [0]*n_samples # 保存每张输入图像对应的文件名
    model.eval()
    with torch.no_grad():
        sample_index = 0
        for img,gt_dmap,gt_dots,img_name, padding in tqdm(simple_train_loader, disable=True):
            # 先对整个 batch 做一次前向，避免 batch_size>1 时仍退化成逐张推理
            img=img.to(device)
            et_dmap_lst, img_feat=model(img, feature_indx_list)
            img_feat = img_feat[:,:,2:-2,2:-2]

            for b in range(img.shape[0]):
                # padding 表示为适配网络下采样结构而补到图像四周的像素，
                # 这样输入尺寸可以被 16 整除，对应网络中的 4 次最大池化
                pad_y1 = int(padding[0][b].item())
                pad_y2 = int(padding[1][b].item())
                pad_x1 = int(padding[2][b].item())
                pad_x2 = int(padding[3][b].item())

                # 记录当前图像文件名
                img_name_list[sample_index] = img_name[b]

                # 去掉 padding 后，得到当前样本所有细胞类别合并的真实中心点图
                gt_dots_sample = gt_dots[b,:,pad_y1:gt_dots.shape[-2]-pad_y2,pad_x1:gt_dots.shape[-1]-pad_x2]
                gt_dots_all = gt_dots_sample.max(0)[0]
                gt_dots_all = gt_dots_all.detach().cpu().numpy()

                # 对当前样本单独去掉 padding，并显式保留通道维，避免 batch_size=1 以外使用 squeeze 误删维度
                img_feat_sample = img_feat[b,:,pad_y1:img_feat.shape[-2]-pad_y2,pad_x1:img_feat.shape[-1]-pad_x2]
                if isinstance(img_feat_sample, torch.Tensor):
                    img_feat_sample = img_feat_sample.permute(1,2,0).detach().cpu().numpy()
                else:
                    img_feat_sample = np.transpose(img_feat_sample, (1,2,0))

                # 根据真实点图确定所有细胞中心坐标
                points = np.where(gt_dots_all > 0)
                coord_list[0][sample_index] = points
                if(len(points[0])==0):
                    # 当前图像不存在细胞时，用 None 占位，方便后续聚类阶段跳过
                    features_list[0][sample_index] = None
                    sample_index += 1
                    continue

                # 直接在特征图上索引细胞中心位置，得到每个细胞对应的特征向量
                features_list[0][sample_index] = img_feat_sample[points]
                sample_index += 1

            del et_dmap_lst

    print(features_list.shape)
    return features_list, coord_list, img_name_list
        
def collect_features_by_class(model, simple_train_loader, feature_indx_list, n_classes):
    n_samples = len(simple_train_loader.dataset)
    features_list = [[0]*n_samples for i in range(n_classes)]    # 为每个类别分别创建独立列表，避免多个类别引用同一对象；n_classes * [n_data]，每个元素是一个高维特征向量
    coord_list = [[0]*n_samples for i in range(n_classes)] # 为每个类别分别保存每张图像中的细胞坐标；n_classes * [n_data]
    img_name_list = ['']*n_samples
    model.eval()
    with torch.no_grad():
        # PROBLEM：这里面的gt_dmap密度图到底干啥用的，这里面也没用到
        sample_index = 0
        for img,gt_dmap,gt_dots,img_name, padding in tqdm(simple_train_loader, disable=True):
            # 先对整个 batch 做一次特征提取
            img=img.to(device)
            et_dmap_lst, img_feat=model(img, feature_indx_list)
            img_feat = img_feat[:,:,2:-2,2:-2]

            for b in range(img.shape[0]):
                # padding 表示为适配网络结构补到输入边界的像素范围
                pad_y1 = int(padding[0][b].item())
                pad_y2 = int(padding[1][b].item())
                pad_x1 = int(padding[2][b].item())
                pad_x2 = int(padding[3][b].item())

                # 记录图像文件名
                img_name_list[sample_index] = img_name[b]

                # 去除 padding 后，保留各个细胞类别的真实点图
                gt_dots_sample = gt_dots[b,:,pad_y1:gt_dots.shape[-2]-pad_y2,pad_x1:gt_dots.shape[-1]-pad_x2]
                gt_dots_sample = gt_dots_sample.detach().cpu().numpy()

                # 从模型特征图中取出当前样本对应区域，并转换为 HWC 形式，保留特征通道维
                img_feat_sample = img_feat[b,:,pad_y1:img_feat.shape[-2]-pad_y2,pad_x1:img_feat.shape[-1]-pad_x2]
                if isinstance(img_feat_sample, torch.Tensor):
                    img_feat_sample = img_feat_sample.permute(1,2,0).detach().cpu().numpy()
                else:
                    img_feat_sample = np.transpose(img_feat_sample, (1,2,0))

                # 分类别收集细胞中心坐标以及这些中心点对应的特征向量
                for s in range(gt_dots_sample.shape[0]):
                    points = np.where(gt_dots_sample[s] > 0)   # 返回一个二元组([x1,x2,x3], [y1,y2,y3])
                    coord_list[s][sample_index] = points   # 存储坐标，[n_class, n_data]
                    if(len(points[0])==0):
                        # 当前图像中该类别没有细胞，后续聚类时直接跳过
                        features_list[s][sample_index] = None
                        continue
                    img_feat_s = img_feat_sample[points]   # 从特征图中提取坐标为 points 的特征
                    features_list[s][sample_index] = img_feat_s    # 加入到 s 类别，[n_class, n_data]

                sample_index += 1

            del et_dmap_lst            
 
    return features_list, coord_list, img_name_list
    
def cluster(features_list, coord_list, n_clusters, prev_centroids):
    '''
        features_list: [n_class, n_data]，记录某个类别某张图上的特征
        coord_list: [n_class, n_data]，记录某张图上满足该类别的若干细胞坐标
    '''
    
    # 对每个类别分别汇总所有细胞特征，执行 KMeans 聚类，
    # 再用训练好的聚类器为每个细胞生成伪子类标签
    cluster_centers_all = None
    pseudo_labels_list = [[0]*len(features_list[0]) for i in range(len(features_list))]     # 伪标签列表，[n_class, n_data]
    for s in range(len(features_list)):
        features = None
        # 拼接当前类别下所有图像中的细胞特征，形成一个总的聚类输入矩阵
        for i in range(len(features_list[s])):
            if(features_list[s][i] is None):
                continue
            if(features is None):
                features = features_list[s][i]
            else:
                features = np.concatenate((features, features_list[s][i]), axis=0)
        print(features.shape)   # [N, 128]，这里的N是属于该类别s的细胞数目，128是特征向量的维度

        # 为了让相邻轮次的聚类结果更稳定，可以使用上一轮的聚类中心作为初始化
        if(prev_centroids is None):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters, init=prev_centroids[s*n_clusters:s*n_clusters+n_clusters]).fit(features)

        # 对该类别下每张图像中的每个细胞预测其所属的子簇标签
        for i in range(len(features_list[s])):
            if(features_list[s][i] is None):
                pseudo_labels_list[s][i] = None
                continue
            pseudo_labels_list[s][i] = kmeans.predict(features_list[s][i])
        if(cluster_centers_all is None):
            cluster_centers_all = kmeans.cluster_centers_
        else:
            cluster_centers_all = np.concatenate((cluster_centers_all, kmeans.cluster_centers_), axis=0)
    print(cluster_centers_all.shape)    # [15, 128]
    # cluster_centers_all: [n_class*n_subclass, feature_dim?]

    # 返回每个细胞的伪子类标签，以及所有类别的聚类中心
    return pseudo_labels_list, cluster_centers_all

def create_pseudo_lbl_gt(simple_train_loader, pseudo_labels_list, coord_list, img_name_list, n_clusters, out_dir):
    n_subclasses = len(pseudo_labels_list) * n_clusters # 子类总数 = 细胞大类数 × 每类聚类数
    sample_index = 0
    for img,gt_dmap,gt_dots,img_name, padding in tqdm(simple_train_loader, disable=True):
        for b in range(img.shape[0]):
            ''' 
                img: 输入图像。
                gt_dmap: 细胞类别的真实区域图，通常是膨胀后的点图。
                         它可能是二值 mask，也可能是密度图；若是密度图，下面会转成二值 mask。
                gt_dots: 细胞类别的真实二值中心点图。
                img_name: 图像文件名。
                padding: 为使输入尺寸能被 16 整除而在四周补的边界像素。
            '''
            pad_y1 = int(padding[0][b].item())
            pad_y2 = int(padding[1][b].item())
            pad_x1 = int(padding[2][b].item())
            pad_x2 = int(padding[3][b].item())
            # 去掉 padding，恢复到真实有效区域上的标注图
            gt_dmap_sample = gt_dmap[b:b+1,:,pad_y1:gt_dmap.shape[-2]-pad_y2,pad_x1:gt_dmap.shape[-1]-pad_x2]
            gt_dots_sample = gt_dots[b:b+1,:,pad_y1:gt_dots.shape[-2]-pad_y2,pad_x1:gt_dots.shape[-1]-pad_x2]
            # 若真实区域图是密度图，则转换为二值 mask，便于后续连通域操作
            gt_dmap_sample = gt_dmap_sample > 0

            # 初始化聚类子类对应的真实点图和真实区域图
            gt_dmap_all =  gt_dmap_sample.max(1)[0]
            gt_dots_all =  gt_dots_sample.max(1)[0] 
            gt_dmap_all = gt_dmap_all.detach().cpu().numpy().squeeze()
            gt_dots_all = gt_dots_all.detach().cpu().numpy().squeeze()
            gt_dots_subclasses = np.zeros((gt_dots_all.shape[0], gt_dots_all.shape[1], n_subclasses+1))
            gt_dmap_subclasses = np.zeros((gt_dots_all.shape[0], gt_dots_all.shape[1], n_subclasses+1))

            # 先对全部细胞区域做连通域标记，后面会把同一细胞区域整体分配到对应子类
            label_comp = label(gt_dmap_all)
            cci = 0
            for s in range(len(pseudo_labels_list)):
                pseudo_labels = pseudo_labels_list[s][sample_index]
                if(pseudo_labels is None):
                    cci += n_clusters
                    continue
                points = coord_list[s][sample_index]
                for c in range(n_clusters):
                    cci += 1
                    # 生成当前“类别-子簇”对应的中心点图
                    gt_map_tmp = np.zeros((gt_dots_subclasses.shape[0],gt_dots_subclasses.shape[1]))
                    gt_map_tmp [(points[0][(pseudo_labels == c)], points[1][(pseudo_labels == c)])]=1
                    gt_dots_subclasses[:,:,cci] = gt_map_tmp

                    # 生成当前“类别-子簇”对应的区域图（膨胀点图或细胞 mask）
                    gt_map_tmp = np.zeros((gt_dmap_subclasses.shape[0],gt_dmap_subclasses.shape[1]))
                    # 将细胞中心点所在的整个连通域都赋给同一个子类，
                    # 从而保证一个细胞区域不会被拆到多个子类中
                    comp_in_cluster = label_comp[(points[0][(pseudo_labels == c)], points[1][(pseudo_labels == c)])]
                    for comp in comp_in_cluster:
                        gt_map_tmp[label_comp==comp] = 1
                    gt_dmap_subclasses[:,:,cci] = gt_map_tmp
                    # 同时保存可视化图像，便于调试伪标签生成是否合理
                    io.imsave(os.path.join(out_dir, img_name_list[sample_index].replace('.png','_gt_dmap_s'+str(s)+'_c'+str(c)+'.png')), (gt_map_tmp*255).astype(np.uint8))

            # 保存当前图像生成好的伪子类点图和区域图，供后续训练阶段直接读取
            gt_dots_subclasses.astype(np.uint8).dump(os.path.join(out_dir, img_name_list[sample_index].replace('.png','_gt_dots.npy')))
            gt_dmap_subclasses.astype(np.uint8).dump(os.path.join(out_dir, img_name_list[sample_index].replace('.png','.npy')))
            sample_index += 1
        

def perform_clustering(model, simple_train_loader, n_clusters, n_classes, feature_indx_list, out_dir, prev_centroids):
    '''
        model: 当前训练中的 MCSpatNet 模型。
        simple_train_loader: 训练数据加载器，用于遍历输入图像并抽取特征。
        n_clusters: 每个细胞类别内部要划分的聚类数。
        n_classes: 细胞大类数量。
        feature_indx_list: 聚类时使用的特征编号列表，
                           feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4}。
        out_dir: 输出生成伪标签文件的目录。
        prev_centroids: 上一轮训练/聚类得到的聚类中心，可用于稳定当前轮初始化。
    '''

    # 先按照任务类型收集聚类所需的细胞特征和坐标
    if(n_classes > 1):
        features_list, coord_list, img_name_list = collect_features_by_class(model, simple_train_loader, feature_indx_list, n_classes)
    else:
        features_list, coord_list, img_name_list = collect_features(model, simple_train_loader, feature_indx_list)

    # 执行聚类，得到新的聚类中心以及每个细胞对应的伪子类标签
    pseudo_labels_list, centroids = cluster(features_list, coord_list, n_clusters, prev_centroids)
    # pseudo_labels_list: [n_class, n_data]，存储某个类别某个图像，用K-mean算法预测的子类
    # prev_centroids: [n_class]

    # 将伪标签落盘，供后续训练阶段作为监督信号读取
    create_pseudo_lbl_gt(simple_train_loader, pseudo_labels_list, coord_list, img_name_list, n_clusters, out_dir)
    return centroids

