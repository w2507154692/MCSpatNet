### 环境配置 

代码已在基于 nvidia pytorch [18.09 release](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_18.09.html) 的 docker 环境中完成测试。  
安装环境如下：  
- python 3.6.5  
- pytorch 0.5.0a0（更高版本如 1.0.0 或 1.4.0 预计也可以运行）  
- Numpy 1.19.4  
- scikit-image 0.15.0  
- OpenCV 4.1.0（这里只使用基础功能，其他版本通常也可以）   
- Scipy 1.1.0  
- Pillow 6.1.0  
- tqdm 4.25.0  
另外，为了在训练中生成 K function maps，还需要：    
- R 4.0.3（安装 spatstat 包）  
- Pandas 1.1.5
- pyper  
- rpy2 


docker 环境可通过以下命令获取：   
    
	docker pull shahira/pytorch_plus_r 

