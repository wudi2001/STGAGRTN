1.prepareData.py 负责处理原始数据，并将其划分为训练集、验证集与测试集。运行完成后得到npz文件，data文件夹下包含处理得到的npz文件，无需重复运行。
2.train_batch.py 负责训练模型，并生成params参数文件。04_gru_epoch_10.params与08_gru_epoch_24.params分别为PEMS04与PEMS08两个数据集下训练得到的参数文件。
3.predict_batch.py 负责得到预测结果。