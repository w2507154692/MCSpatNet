import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import collections
# from distutils.util import strtobool;
import numpy as np  

from sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch;


class UnetVggMultihead(nn.Module):
    def __init__(self, load_weights=False, kwargs=None):
        super(UnetVggMultihead,self).__init__()

        # 预定义参数列表；若外部通过 kwargs 传入同名参数，则会覆盖这里的默认值
        args = {'conv_init': 'he', 'block_size':3, 'pool_size':2
            , 'dropout_prob' : 0, 'initial_pad':0, 'n_classes':1, 'n_channels':3, 'n_heads':2, 'head_classes':[1,1]
 
        };

        if(not(kwargs is None)):
            args.update(kwargs);

        # conv_init 可选初始化方式：'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'he'

        # 读取并保存外部传入的结构参数
        self.n_channels = int(args['n_channels']);
        self.n_classes = int(args['n_classes']);
        self.conv_init = str(args['conv_init']).lower();
        self.n_heads = int(args['n_heads']);
        self.head_classes = np.array(args['head_classes']).astype(int);
    
        self.block_size = int(args['block_size']);
        self.pool_size = int(args['pool_size']);
        self.dropout_prob = float(args['dropout_prob'])
        self.initial_pad = int(args['initial_pad']);

        # print('self.initial_pad',self.initial_pad)

    # U-Net 收缩路径（编码器 + 瓶颈层）
        self.encoder = nn.Sequential()
        layer_index = 0;
        layer = nn.Sequential();
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_0', nn.Conv2d(self.n_channels, 64, kernel_size=self.block_size, padding=self.initial_pad));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(64, 64, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer);

        layer_index = 1;
        layer = nn.Sequential();
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        layer.add_module('encoder_dropout_l_'+str(layer_index), nn.Dropout(p=self.dropout_prob));
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_0', nn.Conv2d(64, 128, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(128, 128, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer);

        layer_index = 2;
        layer = nn.Sequential();
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        layer.add_module('encoder_dropout_l_'+str(layer_index), nn.Dropout(p=self.dropout_prob));
        layer.add_module('encoder_conv_l_'+str(layer_index) + '_0', nn.Conv2d(128, 256, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(256, 256, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_2', nn.Conv2d(256, 256, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_2', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer);

        layer_index = 3;
        layer = nn.Sequential();
        layer.add_module('encoder_maxpool_l_'+str(layer_index), nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        layer.add_module('encoder_dropout_l_'+str(layer_index), nn.Dropout(p=self.dropout_prob));
        layer.add_module('encoder_conv_l_'+str(layer_index) + '_0', nn.Conv2d(256, 512, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_1', nn.Conv2d(512, 512, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(inplace=True))
        layer.add_module('encoder_conv_l_'+str(layer_index)+ '_2', nn.Conv2d(512, 512, kernel_size=self.block_size));
        layer.add_module('encoder_relu_l_'+str(layer_index)+'_2', nn.ReLU(inplace=True))
        self.encoder.add_module('encoder_l_'+str(layer_index), layer);

        self.bottleneck = nn.Sequential();
        self.bottleneck.add_module('bottleneck_maxpool', nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size));
        self.bottleneck.add_module('bottleneck_dropout_l_'+str(layer_index), nn.Dropout(p=self.dropout_prob));
        self.bottleneck.add_module('bottleneck_conv'+ '_0', nn.Conv2d(512, 512, kernel_size=self.block_size));
        self.bottleneck.add_module('bottleneck_relu'+'_0', nn.ReLU(inplace=True))
        self.bottleneck.add_module('bottleneck_conv'+ '_1', nn.Conv2d(512, 512, kernel_size=self.block_size));
        self.bottleneck.add_module('bottleneck_relu'+'_1', nn.ReLU(inplace=True))
        self.bottleneck.add_module('bottleneck_conv'+ '_2', nn.Conv2d(512, 512, kernel_size=self.block_size));
        self.bottleneck.add_module('bottleneck_relu'+'_2', nn.ReLU(inplace=True))
     

        # U-Net 扩张路径（解码器）
        self.decoder = nn.Sequential()
        layer_index = 3;
        layer = nn.Sequential();
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(512, 512, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(1024, 512, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(512, 512, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True));
        self.decoder.add_module('decoder_l_'+str(layer_index), layer);

        layer_index = 2;
        layer = nn.Sequential();
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(512, 256, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(512, 256, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(256, 256, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True));
        self.decoder.add_module('decoder_l_'+str(layer_index), layer);

        layer_index = 1;
        layer = nn.Sequential();
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(256, 128, stride=self.pool_size, kernel_size=self.pool_size))
        layer.add_module('decoder_conv_l_s_'+str(layer_index)+'_0', nn.Conv2d(256, 128, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_0', nn.ReLU(inplace=True))
        layer.add_module('decoder_conv_l_'+str(layer_index)+'_1', nn.Conv2d(128, 128, kernel_size=self.block_size));
        layer.add_module('decoder_relu_l_'+str(layer_index)+'_1', nn.ReLU(True));
        self.decoder.add_module('decoder_l_'+str(layer_index), layer);

        layer_index = 0;
        layer = nn.Sequential();
        layer.add_module('decoder_deconv_l_'+str(layer_index), nn.ConvTranspose2d(128, 96, stride=self.pool_size, kernel_size=self.pool_size))
        self.decoder.add_module('decoder_l_'+str(layer_index), layer);

        self.final_layers_lst=nn.ModuleList()
        # 理想情况下模型包含 4 个输出头：
        # 1. 细胞检测
        # 2. 细胞分类
        # 3. 细胞子类/聚类分类
        # 4. 细胞 cross K-function 回归
        for i in range(self.n_heads):
            block = nn.Sequential();
            feat_subblock = nn.Sequential();
            pred_subblock = nn.Sequential();
            feat_subblock.add_module('final_block_'+str(i)+'_conv3_0', nn.Conv2d(96, 64, kernel_size=self.block_size));
            feat_subblock.add_module('final_block_'+str(i)+'_relu_0', nn.ReLU(inplace=True))
            feat_subblock.add_module('final_block_'+str(i)+'_conv3_1', nn.Conv2d(64, 64, kernel_size=self.block_size));
            feat_subblock.add_module('final_block_'+str(i)+'_relu_1', nn.ReLU(True));
            pred_subblock.add_module('final_block_'+str(i)+'_conv1_2', nn.Conv2d(64, self.head_classes[i], kernel_size=1))
            block.add_module('final_block_'+str(i) +'feat', feat_subblock)
            block.add_module('final_block_'+str(i) +'pred', pred_subblock)
            self.final_layers_lst.append(block)

        # self.final_final_block = nn.Sequential();
        # self.final_final_block.add_module('conv_final', nn.Conv2d(64*self.n_heads, self.n_classes, kernel_size=1));


        self._initialize_weights()

        self.zero_grad() ;

        print('self.encoder',self.encoder)
        print('self.bottleneck',self.bottleneck)
        print('self.decoder',self.decoder)


    def forward(self,x, feat_indx_list=[], feat_as_dict=False):
        '''
            x: 输入图像，默认已完成除以 255 的归一化。
            feat_indx_list: 指定需要返回哪些层的特征。
                若该列表非空，则除预测结果外，还会返回对应编号的特征。
                feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4}
            feat_as_dict: 当 feat_indx_list 非空时，
                若为 True，则将特征按字典形式返回，键为特征编号，值为特征张量；
                否则返回拼接后的特征数组。
        '''
        feat = None
        feat_dict = {}
        feat_indx = 0
        encoder_out = [];
        for l in self.encoder:     
            # 依次通过编码器各层，并缓存中间特征用于后续跳跃连接
            x = l(x);
            encoder_out.append(x);
        x = self.bottleneck(x);
        j = len(self.decoder);
        for l in self.decoder:            
            # 先反卷积上采样，再与对应编码器层特征做裁剪后拼接
            x = l[0](x);
            j -= 1;
            corresponding_layer_indx = j;

            ## 对编码器特征进行裁剪并与当前解码器特征拼接
            if(j > 0):
                cropped = CNNArchUtilsPyTorch.crop_a_to_b(encoder_out[corresponding_layer_indx],  x);
                x = torch.cat((cropped, x), 1) ;
            for i in range(1, len(l)):
                x = l[i](x);


        # 检查是否需要把解码器最终特征作为附加输出返回
        if(feat_indx in feat_indx_list):
            if(feat_as_dict):
                feat_dict[feat_indx] = x.detach().cpu().numpy()
            else:
                feat = x.detach().cpu().numpy()
        
        c=[]
        f=None
        for layer in self.final_layers_lst:
            feat_indx += 1
            f1 = layer[0](x) # 当前输出头的中间特征
            c.append(layer[1](f1)) # 当前输出头的最终预测结果
            if(f is None):
                f = f1
            else:
                f = torch.cat((f1, f), 1) ;

            # 检查当前输出头的特征是否也需要返回
            if(feat_indx in feat_indx_list):
                if(feat_as_dict):
                    feat_dict[feat_indx] = f1.detach().cpu().numpy()
                else:
                    if(feat is None):
                        feat = f1.detach().cpu().numpy()
                    else:
                        feat= np.concatenate((feat, f1.detach().cpu().numpy()), axis=1)

        # 若没有请求返回中间特征，则仅返回各输出头预测结果
        if(len(feat_indx_list) == 0):
            return c

        # 若请求了中间特征，则连同预测结果一起返回
        if(feat_as_dict):
            return c,feat_dict;
        return c,feat;
            

    def _initialize_weights(self):
        # 初始化编码器卷积层参数
        for l in self.encoder:
            for layer in l:
                if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                    if(self.conv_init == 'normal'):
                        torch.nn.init.normal_(layer.weight) ;
                    elif(self.conv_init == 'xavier_uniform'):
                        torch.nn.init.xavier_uniform_(layer.weight) ;
                    elif(self.conv_init == 'xavier_normal'):
                        torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                    elif(self.conv_init == 'he'):
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 

        # 初始化瓶颈层卷积参数
        for layer in self.bottleneck:
            if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                if(self.conv_init == 'normal'):
                    torch.nn.init.normal_(layer.weight) ;
                elif(self.conv_init == 'xavier_uniform'):
                    torch.nn.init.xavier_uniform_(layer.weight) ;
                elif(self.conv_init == 'xavier_normal'):
                    torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                elif(self.conv_init == 'he'):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 


        # 初始化解码器卷积参数
        for l in self.decoder:
            for layer in l:
                if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                    if(self.conv_init == 'normal'):
                        torch.nn.init.normal_(layer.weight) ;
                    elif(self.conv_init == 'xavier_uniform'):
                        torch.nn.init.xavier_uniform_(layer.weight) ;
                    elif(self.conv_init == 'xavier_normal'):
                        torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                    elif(self.conv_init == 'he'):
                        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 


        # 初始化各个输出头中的卷积参数
        for layer in self.final_layers_lst:
            if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
                if(self.conv_init == 'normal'):
                    torch.nn.init.normal_(layer.weight) ;
                elif(self.conv_init == 'xavier_uniform'):
                    torch.nn.init.xavier_uniform_(layer.weight) ;
                elif(self.conv_init == 'xavier_normal'):
                    torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
                elif(self.conv_init == 'he'):
                    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 


        # 用预训练 VGG-16 参数初始化编码器和瓶颈层，借助分类模型的底层视觉特征加速收敛
        vgg_model = models.vgg16(pretrained = True)
        fsd=collections.OrderedDict()
        i = 0
        for m in self.encoder.state_dict().items():
            temp_key=m[0]
            print('temp_key', temp_key)
            print('vgg_key', list(vgg_model.state_dict().items())[i][0])
            fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
            i += 1
        self.encoder.load_state_dict(fsd)

        fsd=collections.OrderedDict()
        for m in self.bottleneck.state_dict().items():
            temp_key=m[0]
            print('temp_key', temp_key)
            print('vgg_key', list(vgg_model.state_dict().items())[i][0])
            fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
            i += 1
        self.bottleneck.load_state_dict(fsd)

        #del vgg_model



