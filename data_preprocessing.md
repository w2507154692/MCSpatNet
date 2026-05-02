### 生成真值标签 
1. 下载并解压 CoNSeP 数据集到目录 `../MCSpatNet_datasets`
	

	     wget https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep.zip -P ../MCSpatNet_datasets
	     unzip ../datasets/consep.zip -d ../MCSpatNet_datasets

2. `cd data_prepare/`
3. 编辑 `1_generate_dot_maps_consep.py` <br/>
设置以下变量：<br/>
`in_dir` 指向 CoNSeP 的训练/测试目录，<br/>
`out_root_dir` 分别指向训练/测试数据的输出目录。<br/>
默认值如下：

	     in_dir = '../../MCSpatNet_datasets/CoNSeP/Train' 
	     out_root_dir = '../../MCSpatNet_datasets/CoNSeP_train' 
         
4. 运行 `1_generate_dot_maps_consep.py`


		python 1_generate_dot_maps_consep.py

	它会在输出目录中创建两个子目录：`images` 和 `gt_custom`。<br/>
	生成的文件如下：	<br/>		

	- images/: <br/>
		- `<img_name>.png`：按 0.5 缩放后的图像（20x）。<br/>
	- gt\_custom/: <br/>
		- `<img_name>_gt_dots.npy`：分类点标注图。<br/>
		- `<img_name>_gt_dots_all.npy`：检测点标注图。<br/>
		- `<img_name>.npy`：分类二值掩码。<br/>
		- `<img_name>_all.npy`：检测二值掩码。<br/>
		- `<img_name>_s<class id>_binary.png`：各类别二值掩码的可视化结果（默认：1=炎症细胞，2=上皮细胞，3=基质细胞）。<br/>
		- `<img_name>_binary.png`：检测二值掩码的可视化结果。<br/>
		- `<img_name>_img_with_dots.jpg`：带有细胞点标注可视化的图像，不同点颜色表示不同类别。（默认：蓝=炎症细胞，红=上皮细胞，绿=基质细胞）。<br/>
			
5. 编辑 `2_calc_kmaps.py` <br/>
设置以下变量：<br/>
`root_dir` 指向上一步生成的 CoNSeP 训练/测试目录。<br/>
默认值如下：

	     root_dir = '../../MCSpatNet_datasets/CoNSeP_train' 

6. 运行 `2_calc_kmaps.py`

		python 2_calc_kmaps.py

	它会在输出目录中创建子目录：`k_func_maps`。<br/>
	该脚本会生成 cross k function maps，文件名格式为 `k_func_maps/<img_name>_gt_kmap.npy` <br/>		

7. 使用测试数据目录重复执行第 3-6 步：<br/>
	将 `CoNSeP/Train` 替换为 `CoNSeP/Test` <br/>
	将 `CoNSeP_train` 替换为 `CoNSeP_test`
	
