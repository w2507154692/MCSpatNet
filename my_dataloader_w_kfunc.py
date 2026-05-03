from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
import glob
import skimage.io as io


class CellsDataset(Dataset):
    def __init__(self,img_root, gt_dmap_root, gt_dots_root, class_indx, gt_dmap_subclasses_root, gt_dots_subclasses_root, gt_kmap_root, split_filepath=None, phase='train', fixed_size=-1, max_side=-1, max_scale=-1):
        super(CellsDataset, self).__init__()
        '''
        img_root: 输入图像所在根目录。
        gt_dmap_root: 真实膨胀点图所在根目录。
        gt_dots_root: 真实中心点图所在根目录。
        class_indx: 需要从真实标注中返回的通道索引，格式为逗号分隔字符串。
        gt_dmap_subclasses_root: 聚类生成的伪子类膨胀点图所在根目录。
        gt_dots_subclasses_root: 聚类生成的伪子类中心点图所在根目录。
        gt_kmap_root: 真实 cross K-function 图所在根目录。
        split_filepath: 若不为 None，则只读取该文件中列出的图像。
        phase: 当前阶段，通常为 train 或 test。
        fixed_size: 若大于 0，则训练时返回边长为 fixed_size 的随机裁剪块。
        max_side: 训练时允许的最大边长，超过则先随机裁剪到该范围内。
        max_scale: 为使 patch 边长可被 max_scale 整除而补边。
        '''
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_dots_root=gt_dots_root
        self.gt_dmap_subclasses_root=gt_dmap_subclasses_root
        self.gt_dots_subclasses_root=gt_dots_subclasses_root
        self.gt_kmap_root = gt_kmap_root
        self.phase=phase

        if(split_filepath is None):
            self.img_names=[filename for filename in os.listdir(img_root) \
                               if os.path.isfile(os.path.join(img_root,filename))]
        else:
            self.img_names=np.loadtxt(split_filepath, dtype=str).tolist()
            
        self.n_samples=len(self.img_names)

        self.fixed_size = fixed_size
        self.max_side = max_side
        self.max_scale = max_scale
        self.class_indx = class_indx
        self.class_indx_list = [int(x) for x in self.class_indx.split(',')]


    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'

        # 读取图像并归一化到 [0,1]，同时确保最终是三通道 RGB 形式
        img_name=self.img_names[index]
        print('img_name',img_name)
        img=io.imread(os.path.join(self.img_root,img_name))/255# convert from [0,255] to [0,1]
        if len(img.shape)==2: # 如果是灰度图，则复制成三通道，保持网络输入格式一致
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        # 读取真实膨胀点图；若文件不存在，则构造全 0 占位张量
        gt_path = os.path.join(self.gt_dmap_root,img_name.replace('.png','.npy'));
        if(os.path.isfile(gt_path)):
            gt_dmap=np.load(gt_path, allow_pickle=True)[:,:,self.class_indx_list].squeeze()
        else:
            gt_dmap=np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)))

        # 读取真实中心点图；若文件不存在，则构造全 0 占位张量
        gt_dots_path = os.path.join(self.gt_dots_root,img_name.replace('.png','_gt_dots.npy'));
        if(os.path.isfile(gt_dots_path)):
            gt_dots=np.load(gt_dots_path, allow_pickle=True)[:,:,self.class_indx_list].squeeze()
        else:
            gt_dots=np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)))

        # 读取真实 cross K-function 图；若不存在，则用固定通道数的 0 张量代替
        gt_kmap_path = os.path.join(self.gt_kmap_root,img_name.replace('.png','_gt_kmap.npy'));
        if(os.path.isfile(gt_kmap_path)):
            gt_kmap=np.load(gt_kmap_path, allow_pickle=True).squeeze()
        else:
            gt_kmap=np.zeros((img.shape[0], img.shape[1], 21))
        
        # 读取伪子类膨胀点图。
        # 这里兼容两种命名：xxx.npy 和 xxx.png.npy，读取后丢弃第 0 个背景通道。
        gt_subclasses_path = os.path.join(self.gt_dmap_subclasses_root,img_name.replace('.png','.npy'));
        gt_subclasses_path2 = os.path.join(self.gt_dmap_subclasses_root,img_name.replace('.png','.png.npy'));
        print(gt_subclasses_path)
        if(os.path.isfile(gt_subclasses_path)):
            gt_subclasses_dmap=np.load(gt_subclasses_path, allow_pickle=True)[:,:,1:].squeeze()
        elif(os.path.isfile(gt_subclasses_path2)):
            gt_subclasses_dmap=np.load(gt_subclasses_path2, allow_pickle=True)[:,:,1:].squeeze()
        else:
            gt_subclasses_dmap=np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)*5))

        # 读取伪子类中心点图；同样去掉第 0 个背景通道
        gt_subclasses_dots_path = os.path.join(self.gt_dots_subclasses_root,img_name.replace('.png','_gt_dots.npy'));
        if(os.path.isfile(gt_subclasses_dots_path)):
            gt_subclasses_dots=np.load(gt_subclasses_dots_path, allow_pickle=True)[:,:,1:].squeeze()
        else:
            gt_subclasses_dots=np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)*5))

        # 训练阶段随机进行左右翻转增强，图像与所有监督信号保持同步翻转
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[:,::-1].copy() # horizontal flip
            gt_dmap=gt_dmap[:,::-1].copy() # horizontal flip
            gt_dots=gt_dots[:,::-1].copy() # horizontal flip
            gt_subclasses_dmap=gt_subclasses_dmap[:,::-1].copy() # horizontal flip
            gt_subclasses_dots=gt_subclasses_dots[:,::-1].copy() # horizontal flip
            gt_kmap=gt_kmap[:,::-1].copy() # horizontal flip
        
        # 训练阶段随机进行上下翻转增强，同样保证所有标注严格对齐
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[::-1,:].copy() # vertical flip
            gt_dmap=gt_dmap[::-1,:].copy() # vertical flip
            gt_dots=gt_dots[::-1,:].copy() # vertical flip
            gt_subclasses_dmap=gt_subclasses_dmap[::-1,:].copy() # vertical flip
            gt_subclasses_dots=gt_subclasses_dots[::-1,:].copy() # vertical flip
            gt_kmap=gt_kmap[::-1,:].copy() # vertical flip

        # 训练阶段若图像尺寸超过 max_side，则先随机裁出一个不超过该上限的区域
        if(self.phase=='train' and self.max_side > 0):
            h = img.shape[0]
            w = img.shape[1]
            h2 = h
            w2 = w
            crop = False
            if(h > self.max_side):
                h2 = self.max_side
                crop = True
            if(w > self.max_side):
                w2 = self.max_side
                crop = True
            if(crop):
                y=0
                x=0
                if(not (h2 ==h)):
                    y = np.random.randint(0, high = h-h2)
                if(not (w2 ==w)):
                    x = np.random.randint(0, high = w-w2)
                img = img[y:y+h2, x:x+w2, :]
                gt_dmap = gt_dmap[y:y+h2, x:x+w2]
                gt_dots = gt_dots[y:y+h2, x:x+w2]
                gt_subclasses_dmap = gt_subclasses_dmap[y:y+h2, x:x+w2]
                gt_subclasses_dots = gt_subclasses_dots[y:y+h2, x:x+w2]
                gt_kmap = gt_kmap[y:y+h2, x:x+w2]

        
            # 训练阶段再做一次随机裁剪：
            # fixed_size < 0 时取原图 1/4 尺寸；否则尽量裁成 fixed_size 大小
        if self.phase=='train':
            i = -1
            img_pil = Image.fromarray(img.astype(np.uint8)*255);
            if(self.fixed_size < 0):
                i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(img.shape[0]//4, img.shape[1]//4))
            elif(self.fixed_size < img.shape[0] or self.fixed_size < img.shape[1]):
                i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(min(self.fixed_size,img.shape[0]), min(self.fixed_size,img.shape[1])))
            if(i >= 0):
                img = img[i:i+h, j:j+w, :]
                gt_dmap = gt_dmap[i:i+h, j:j+w]
                gt_dots = gt_dots[i:i+h, j:j+w]
                gt_subclasses_dmap = gt_subclasses_dmap[i:i+h, j:j+w]
                gt_subclasses_dots = gt_subclasses_dots[i:i+h, j:j+w]
                gt_kmap = gt_kmap[i:i+h, j:j+w]



        # 如有需要，在四周补边，使图像尺寸可以被 max_scale 整除，
        # 以适配网络中的下采样倍率
        if self.max_scale>1: # 这样下采样后的图像和监督图尺寸能够与深度模型对齐
            ds_rows=int(img.shape[0]//self.max_scale)*self.max_scale
            ds_cols=int(img.shape[1]//self.max_scale)*self.max_scale
            pad_y1 = 0
            pad_y2 = 0
            pad_x1 = 0
            pad_x2 = 0
            if(ds_rows < img.shape[0]):
                pad_y1 = (self.max_scale - (img.shape[0] - ds_rows))//2
                pad_y2 = (self.max_scale - (img.shape[0] - ds_rows)) - pad_y1
            if(ds_cols < img.shape[1]):
                pad_x1 = (self.max_scale - (img.shape[1] - ds_cols))//2
                pad_x2 = (self.max_scale - (img.shape[1] - ds_cols)) - pad_x1
            img = np.pad(img, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(1,) )# 图像补边常数取决于数据集背景颜色，这里使用 1
            gt_dmap = np.pad(gt_dmap, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# 监督图补边统一填 0，表示新增区域没有目标
            gt_dots = np.pad(gt_dots, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# 监督图补边统一填 0，表示新增区域没有目标
            gt_subclasses_dmap = np.pad(gt_subclasses_dmap, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# 监督图补边统一填 0，表示新增区域没有目标
            gt_subclasses_dots = np.pad(gt_subclasses_dots, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# 监督图补边统一填 0，表示新增区域没有目标
            gt_kmap = np.pad(gt_kmap, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# K-function 图补边也填 0，避免引入伪响应


        # 将 numpy 数组转换为 PyTorch 常用的通道优先格式 (C, H, W)
        img=img.transpose((2,0,1)) # 从 (H, W, C) 转成 (C, H, W)
        gt_kmap=gt_kmap.transpose((2,0,1)) # 从 (H, W, C) 转成 (C, H, W)
        if(len(self.class_indx_list) > 1):
            gt_dmap=gt_dmap.transpose((2,0,1)) # 多通道监督图直接转成 (C, H, W)
            gt_dots=gt_dots.transpose((2,0,1)) # 多通道监督图直接转成 (C, H, W)
            gt_subclasses_dmap=gt_subclasses_dmap.transpose((2,0,1)) # 多通道监督图直接转成 (C, H, W)
            gt_subclasses_dots=gt_subclasses_dots.transpose((2,0,1)) # 多通道监督图直接转成 (C, H, W)
        else:
            # 单类别任务时人为补一个通道维，保证返回张量形状一致
            gt_dmap=gt_dmap[np.newaxis,:,:]
            gt_dots=gt_dots[np.newaxis,:,:]
            gt_subclasses_dmap=gt_subclasses_dmap[np.newaxis,:,:]
            gt_subclasses_dots=gt_subclasses_dots[np.newaxis,:,:]

        # 最终统一转换为 float 类型张量，供模型直接读取
        img_tensor=torch.tensor(img,dtype=torch.float)
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        gt_dots_tensor=torch.tensor(gt_dots,dtype=torch.float)
        gt_subclasses_dmap_tensor=torch.tensor(gt_subclasses_dmap,dtype=torch.float)
        gt_subclasses_dots_tensor=torch.tensor(gt_subclasses_dots,dtype=torch.float)
        gt_kmap_tensor=torch.tensor(gt_kmap,dtype=torch.float)

        # 返回图像、各类监督信号以及文件名
        return img_tensor,gt_dmap_tensor,gt_dots_tensor, gt_subclasses_dmap_tensor, gt_subclasses_dots_tensor, gt_kmap_tensor, img_name


