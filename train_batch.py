import torch
import torch.nn as nn
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import load_graphdata_channel_my, compute_val_loss_STGAGRTN
from time import time
import argparse
import os
from model import STGAGRTN


# 对原始邻接矩阵进行处理
def compute_adj(adj):
    adj = adj.numpy()
    std_data = []  # 收集所有具有记录距离的节点
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] > 1:
                std_data.append(adj[i, j])
    std_data = np.array(std_data)
    adj_mean = np.mean(std_data)
    adj_std = np.std(std_data)
    w = np.zeros((len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] > 0:
                w[i, j] = np.exp(-(adj[i, j] / adj_std) ** 2)
    return torch.Tensor(w)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def mkdir(path):

    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)



if __name__ == '__main__':

    # 设置随机数种子
    setup_seed(0)

    parser = argparse.ArgumentParser(description='STGAGRTN')
    parser.add_argument('--out_dir', type=str, default='./out_dir')
    parser.add_argument('--dataset', type=str, default='./data/PEMS08/r1_d1_w1_PEMS08.npz', help='options: [./data/PEMS04/r1_d1_w1_PEMS04.npz, ./data/PEMS08/r1_d1_w1_PEMS08.npz]')
    parser.add_argument('--adj', type=str, default='./data/PEMS08/adj.csv', help='adjacency matrix, options: [./data/PEMS04/adj.csv, ./data/PEMS08/adj.csv]')
    args = parser.parse_args()
    params_path = args.out_dir
    filename = args.dataset
    adj_mx = pd.read_csv(args.adj, header=None)


    num_of_hours, num_of_days, num_of_weeks = 1, 1, 1  ## The same setting as prepareData.py, 1表示使用一个小时的数据

    ### Training Hyparameter
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'

    DEVICE = device
    batch_size = 100
    learning_rate = 0.01
    epochs = 40

    ### Generate Data Loader
    train_loader, val_loader, test_loader, _mean, _std = load_graphdata_channel_my(
        filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)

    ### Adjacency Matrix Import
    adj_mx = np.array(adj_mx)
    A = adj_mx
    A = torch.Tensor(A)
    A = compute_adj(A)

    ### Training Hyparameter
    in_channels = 1  # Channels of input
    embed_size = 64  # Dimension of hidden embedding features
    time_num = 288   # 一天共有288个时间步
    T_dim = 36  # Input length, should be the same as prepareData.py
    output_T_dim = 6  # Output Expected length
    heads = 2  # Number of Heads in MultiHeadAttention
    forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    dropout = 0.1
    iters = len(train_loader)

    ### Construct Network
    net = STGAGRTN(
        A,
        in_channels,
        embed_size,
        time_num,
        T_dim,
        output_T_dim,
        heads,
        forward_expansion,
        dropout,
        device)

    net.to(device)


    #### Loss Function Setting
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=iters / 8)



    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    start_time = time()


    #### train model
    for epoch in tqdm(range(epochs)):
        net.train()  # ensure dropout layers are in train mode
        for batch_index, batch_data in tqdm(enumerate(train_loader), desc='train'):
            encoder_inputs, time_encoder, labels = batch_data
            optimizer.zero_grad()
            outputs = net(encoder_inputs.permute(0, 2, 1, 3), time_encoder)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            training_loss = loss.item()
            print(training_loss)
            global_step += 1
            if global_step % 100 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs' % (
                global_step, training_loss, time() - start_time))

        ##### Parameter Saving
        mkdir(params_path)
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        ##### Evaluate on Validation Set
        val_loss = compute_val_loss_STGAGRTN(net, val_loader, criterion, epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)
    print('best epoch:', best_epoch)











