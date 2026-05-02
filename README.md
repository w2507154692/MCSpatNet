# MCSpatNet
### 用于 Spatial Context Representation 多类别细胞检测的代码仓库，ICCV 2021 ###

[**Shahira Abousamra, David Belinsky, John Van Arnam, Felicia Allard, Eric Yee, Rajarsi Gupta, Tahsin Kurc, Dimitris Samaras, Joel Saltz, Chao Chen, Multi-Class Cell Detection Using Spatial Context Representation, ICCV 2021.**](https://arxiv.org/pdf/2110.04886.pdf)

<figure>
  <img src="./arch5.png" alt="MCSpatNet Architecture" style="width:90%">
<p align="center"> 
  <figcaption style="font-weight:bold;text-align:center">Multi-Class Spatial Network (MCSpatNet)</figcaption>
</p>
</figure>

<br/>
  
  
  
- **环境配置**： 
请参考 `environment.md`。

- **生成真值标签**：请参考 `data_preprocessing.md`。

- **模型训练与评估**：请参考 `train_and_test.md`。

- **预处理后的数据集**：位于 `datasets` 目录下。包括：<br/>
CoNSeP 数据集：<br/>
S. Graham, Q. D. Vu, S. E. A. Raza, A. Azam, Y-W. Tsang, J. T. Kwak and N. Rajpoot. "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images." Medical Image Analysis, Sept. 2019 .(https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/) <br/><br/> 
BRCA-M2C 数据集：<br/>
本文配套数据集：<br/>
S. Abousamra, D. Belinsky, J. V. Arnam, F. Allard, E. Yee, R. Gupta, T. Kurc, D. Samaras, J. Saltz, C. Chen, "Multi-Class Cell Detection Using Spatial Context Representation", ICCV 2021. <br/>
(https://github.com/TopoXLab/Dataset-BRCA-M2C)

- **训练好的模型**：位于 `pretrained_models` 目录下。请参考 `pretrained_models.md`。

- **训练模型的测试结果**：位于 `pretrained_results` 目录下。

### 引用 ###
	@InProceedings{Abousamra_2021_ICCV,
    author    = {Abousamra, Shahira and Belinsky, David and Van Arnam, John and Allard, Felicia and Yee, Eric and Gupta, Rajarsi and Kurc, Tahsin and Samaras, Dimitris and Saltz, Joel and Chen, Chao},  
    title     = {Multi-Class Cell Detection Using Spatial Context Representation},  
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},  
    year      = {2021},  
	}
