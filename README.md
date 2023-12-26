# STGAGRTN
This is a PyTorch implementation of STGAGRTN in the Spatial-Temporal Graph Attention Gated Recurrent Transformer Network for Traffic Flow Forecasting. [Spatial-Temporal Graph Attention Gated Recurrent Transformer Network for Traffic Flow Forecasting](https://ieeexplore.ieee.org/document/10347394) has been published in the **IEEE Internet of Things Journal**. 

![figure1](https://github.com/wudi2001/STGAGRTN/blob/main/img.png)

## 核心文件介绍
* prepareData.py 负责处理原始数据，并将其划分为训练集、验证集与测试集。运行完成后得到npz文件，data文件夹下包含处理得到的npz文件，无需重复运行。
* train.py 负责训练模型，并生成params参数文件。04.params 与 08.params 分别是模型在PEMS04与PEMS08两个数据集下训练得到的参数文件。
* predict.py 负责得到预测结果。
* model.py 与 GAT.py 两个文件是模型主体部分。

# Requirements
* Python 3.8
* numpy == 1.21.0
* pandas == 1.2.4
* torch == 1.10.2+cu113
* torch-geometric == 2.0.4

