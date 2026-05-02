
### 模型训练

1. 进入仓库根目录：
	
	`cd ..`

2. 编辑 `01_train_mcspat.py` <br/>
设置以下变量：
	- `checkpoints_root_dir `：所有训练输出的根目录。<br/>
	- `checkpoints_folder_name`：训练输出文件夹名称，其路径为 `<checkpoints_root_dir>/<checkpoints_folder_name>`。<br/>
	- `model_param_path`：用于继续训练的历史 checkpoint 路径。<br/>
	- `clustering_pseudo_gt_root`：训练过程中生成的所有伪真值聚类标签的根目录。当前训练生成的聚类伪标签会保存在 `<clustering_pseudo_gt_root>/<checkpoints_folder_name>` <br/>
	- `train_data_root`：训练数据集根目录。<br/>
	- `test_data_root`：验证数据集根目录。<br/>
	- `train_split_filepath`：包含训练所用文件列表的文本文件路径。如果设为 None，则使用训练图像文件夹中的所有图像。<br/>
	- `test_split_filepath`：包含验证所用文件列表的文本文件路径。如果设为 None，则使用验证图像文件夹中的所有图像。<br/>
	
 
	默认值如下：

	    checkpoints_root_dir = '../MCSpatNet_checkpoints' 
		checkpoints_folder_name = 'mcspatnet_consep_1'
		model_param_path = None
		clustering_pseudo_gt_root = '../MCSpatNet_epoch_subclasses'
		train_data_root = '../MCSpatNet_datasets/CoNSeP_train'
		test_data_root = '../MCSpatNet_datasets/CoNSeP_train'
		train_split_filepath = './data_splits/consep/train_split.txt'
		test_split_filepath = './data_splits/consep/val_split.txt'

 
6. 开始训练
		
		CUDA_VISIBLE_DEVICES='1' nohup python 01_train_mcspat.py > tmp_log.txt &

	请根据实际情况修改 `CUDA_VISIBLE_DEVICES` 和 `tmp_log.txt`。
 
### 模型测试

1. 进入仓库根目录。

2. 编辑 `02_test_vis_mcspat.py` <br/>
设置以下变量：
	- `checkpoints_root_dir `：所有训练输出的根目录。<br/>
	- `checkpoints_folder_name`：训练输出文件夹名称。保存的 checkpoint 位于 `<checkpoints_root_dir>/<checkpoints_folder_name>` 文件夹中。<br/>
	- `eval_root_dir`：所有测试输出的根目录。预测结果会保存在 `<eval_root_dir>/<checkpoints_folder_name>_e<epoch>`。<br/>
	- `epoch`：要测试的 checkpoint 轮次。
	- `visualize`：布尔值，表示是否输出预测结果的可视化。<br/>
	- `test_data_root`：测试数据集根目录。<br/>
	- `test_split_filepath`：包含测试所用文件列表的文本文件路径。如果设为 None，则使用测试图像文件夹中的所有图像。<br/>
	
 
	默认值如下：

	    checkpoints_root_dir = '../MCSpatNet_checkpoints' 
		checkpoints_folder_name = 'mcspatnet_consep_1'
		eval_root_dir = '../MCSpatNet_eval'
		epoch = 100
		visualize = True
		test_data_root = '../MCSpatNet_datasets/CoNSeP_test'
		test_split_filepath = None

3. 运行 `02_test_vis_mcspat.py`

		CUDA_VISIBLE_DEVICES='1' nohup python 02_test_vis_mcspat.py > tmp_test_log.txt &

	输出为预测结果，以及可选的可视化结果。

	- `<img_name>_gt_dots_class.npy`：真值分类点图。  
	- `<img_name>_gt_dots_all.npy`：真值检测点图。  
	- `<img_name>_likelihood_class.npy`：预测分类似然图。 
	- `<img_name>_likelihood_all.npy`：预测检测似然图。 
	- `<img_name>_centers_s<class id>.npy`：各细胞类型的预测分类点图。（默认：0=炎症细胞，1=上皮细胞，2=基质细胞） 
	- `<img_name>_centers_all.npy`：预测检测点图。  
	
	（可选）如果 `visualize` 设为 True：
	- `<img_name>.png`：输入图像
	- `<img_name>_centers_det_overlay.png`：将预测细胞检测结果叠加到图像上的可视化结果。 
	- `<img_name>_centers_class_overlay.png`：将预测细胞分类结果叠加到图像上的可视化结果。 
	- `<img_name>_gt_centers_class_overlay.png`：将真值细胞分类结果叠加到图像上的可视化结果。 

4. 编辑 `03_eval_localization_fscore.py`<br/>
设置以下变量：
	- `data_dir `：运行 `02_test_vis_mcspat.py` 后生成的预测输出路径。<br/>
	- `max_dist_thresh`：使用 `[1-<max_dist_thresh>]` 范围内的距离阈值进行评估。距离阈值指预测中心点与真值点之间被判定为真正例的最大距离，单位为像素。

 
	默认值如下：

	    data_dir = '../MCSpatNet_eval/mcspatnet_consep_1_e100'
		max_dist_thresh = 6
 
5. 运行 `03_eval_localization_fscore.py`

		python 03_eval_localization_fscore.py 

	输出文件 `<data_dir>/out_distance_scores.txt`，其中包含在 `[1-<max_dist_thresh>]` 距离阈值范围内分类与检测任务的 precision、recall 和 f-score。
