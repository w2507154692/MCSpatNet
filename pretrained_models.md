
## 预训练模型
  
* 预训练模型位于：./pretrained_models
* 解压分卷压缩的模型：  
	`cd pretrained_models/ `   
	`cat mcspat_brca_m2c_consep_seerlung.zip.* > mcspat_brca_m2c_consep_seerlung.zip`  
	`unzip mcspat_brca_m2c_consep_seerlung.zip`

* 当前可用的预训练模型包括：
	* `mcspat_brca-m2c`：在 BRCA-M2C 数据集上训练。
	* `mcspat_consep`：在 ConSeP 数据集上训练。
	* `mcspat_seer-lung`：在 SEER-Lung 数据集上训练。
	* `mcspat_brca_m2c_consep_seerlung`：在 BRCA-M2C、ConSeP 和 SEER-Lung 数据集上训练。
	
