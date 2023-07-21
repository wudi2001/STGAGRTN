# STGAGRTN
This is a PyTorch implementation of STGAGRTN in the Spatial-Temporal Graph Attention Gated Recurrent Transformer Network for Traffic Flow Forecasting.

## 核心文件介绍
* prepareData.py 负责处理原始数据，并将其划分为训练集、验证集与测试集。运行完成后得到npz文件，data文件夹下包含处理得到的npz文件，无需重复运行。
* train_batch.py 负责训练模型，并生成params参数文件。04_gru_epoch_10.params与08_gru_epoch_24.params分别为PEMS04与PEMS08两个数据集下训练得到的参数文件。
* predict_batch.py 负责得到预测结果。

# Requirements
* Python 3.8
* numpy == 1.21.0
* pandas == 1.2.4
* torch == 1.10.2+cu113
* torch-geometric == 2.0.4

