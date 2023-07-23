import numpy as np
import torch
import torch.utils.data


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def load_data(filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):

    file_data = np.load(filename, allow_pickle=True)
    train_x = file_data['train_x'].astype(float)
    train_target = file_data['train_target'].astype(float)
    train_time_features = file_data['train_time_features'].astype(float)

    val_x = file_data['val_x'].astype(float)
    val_target = file_data['val_target'].astype(float)
    val_time_features = file_data['val_time_features'].astype(float)

    test_x = file_data['test_x'].astype(float)
    test_target = file_data['test_target'].astype(float)
    test_time_features = file_data['test_time_features'].astype(float)

    mean = file_data['mean'].astype(float)
    std = file_data['std'].astype(float)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    train_time_features_tensor = torch.from_numpy(train_time_features).type(torch.FloatTensor).to(DEVICE)

    ### torch.utils.data.TensorDataset生成TensorDataset并方便生成后续的Loader，第一维要求是样本维
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_time_features_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    val_time_features_tensor = torch.from_numpy(val_time_features).type(torch.FloatTensor).to(DEVICE)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_time_features_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)
    test_time_features_tensor = torch.from_numpy(test_time_features).type(torch.FloatTensor).to(DEVICE)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_time_features_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, val_loader, test_loader, mean, std


def compute_val_loss_STGAGRTN(net, val_loader, criterion, epoch, limit=None):

    net.eval()  # ensure dropout layers are in evaluation mode
    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            input, time_features, ground_truth = batch_data
            output = net(input.permute(0, 2, 1, 3), time_features)
            loss = criterion(output, ground_truth)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 10 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)

    return validation_loss