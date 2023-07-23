import os
import numpy as np
import pandas as pd


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.day


class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofyear


def time_features():
    # 由于我们的所有数据都是以分钟为最小单位，故我们提取以下五个时间特征
    attr = [
        MinuteOfHour,
        HourOfDay,
        DayOfWeek,
        DayOfMonth,
        DayOfYear,
    ]
    return [cls() for cls in attr]


def search_data(Q, sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data
    num_of_depend: int,
    label_start_idx: int, the first index of predicting target
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    if units > 1:
        # 对于日周期与周周期数据含有时间窗口。他们所包含的时间节点为预测六个时刻对应周期节点加左右两个窗口各三个时刻数据，共计12个时刻值
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - points_per_hour * units * i - Q
            end_idx = start_idx + num_for_predict + 2 * Q
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None
    else:
        # 近期数据，近期的12个时刻值
        for i in range(1, num_of_depend + 1):
            start_idx = label_start_idx - points_per_hour * units * i
            end_idx = start_idx + points_per_hour * units * i
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None

    if len(x_idx) != num_of_depend:
        return None

    return x_idx[::-1]


def get_sample_indices(Q, data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):

    week_sample, day_sample, hour_sample = None, None, None

    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    if num_of_weeks > 0:
        week_indices = search_data(Q, data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        if not week_indices:
            return None, None, None, None

        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    if num_of_days > 0:
        day_indices = search_data(Q, data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        if not day_indices:
            return None, None, None, None

        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    if num_of_hours > 0:
        hour_indices = search_data(Q, data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        if not hour_indices:
            return None, None, None, None

        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]  # 获取本组数据的预测值

    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(Q, time_data_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, save=False):

    x = pd.read_csv(time_data_filename, header=None)
    data_seq = np.array(x)
    if len(data_seq.shape) == 2:
        data_seq = np.expand_dims(data_seq, 2)
        print('Dim Expansion')  # 表示其只有流量这一个属性

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(Q, data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []  # [(week_sample),(day_sample),(hour_sample),target,time_sample]

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)


        time_predict = data_seq[idx, 0]
        sample.append(time_predict)

        all_samples.append(sample)  # sampe：[(week_sample),(day_sample),(hour_sample),target,time_predict] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,)]

    split_line1 = int(len(all_samples) * 0.8)
    split_line2 = int(len(all_samples) * 0.9)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')  沿着时间维度进行扩充，以构建时间步
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    # 获得所属时间
    train_x_time = train_x[:, 0, :, :]
    train_x_time = np.squeeze(train_x_time)
    val_x_time = val_x[:, 0, :, :]
    val_x_time = np.squeeze(val_x_time)
    test_x_time = test_x[:, 0, :, :]
    test_x_time = np.squeeze(test_x_time)

    # 对原有数据进行更新
    train_x = train_x[:, 1:, :, :].astype('float')
    val_x = val_x[:, 1:, :, :].astype('float')
    test_x = test_x[:, 1:, :, :].astype('float')

    train_target = training_set[-2][:, 1:, :].astype('float')  # (B,N,T)
    val_target = validation_set[-2][:, 1:, :].astype('float')
    test_target = testing_set[-2][:, 1:, :].astype('float')


    train_predict = training_set[-1]
    val_predict = validation_set[-1]
    test_predict = testing_set[-1]

    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    # 计算时间编码
    # 由于训练所用时间为一个矩阵，转为向量后送入转化为datetime
    train_time_list = []  # 最终得到的格式为[总数，时间步，五个特征]
    val_time_list = []
    test_time_list = []
    # 遍历每一组数据，一次计算该组数据的所有时间步
    for i in range(train_x_time.shape[0]):
        train_time_list.append(
            np.vstack([feat(pd.to_datetime(train_x_time[i, :])) for feat in time_features()]).transpose(1, 0))

    for i in range(val_x_time.shape[0]):
        val_time_list.append(
            np.vstack([feat(pd.to_datetime(val_x_time[i, :])) for feat in time_features()]).transpose(1, 0))

    for i in range(test_x_time.shape[0]):
        test_time_list.append(
            np.vstack([feat(pd.to_datetime(test_x_time[i, :])) for feat in time_features()]).transpose(1, 0))

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'train_predict': train_predict,
            'train_time_features': train_time_list,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'val_predict': val_predict,
            'val_time_features': val_time_list,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'test_predict': test_predict,
            'test_time_features': test_time_list,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }

    if save:
        dir = './data/PEMS04'
        filename = ('r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_PEMS04.npz'
        filename = os.path.join(dir, filename)
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_predict=all_data['train']['train_predict'],
                            train_time_features=all_data['train']['train_time_features'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_predict=all_data['val']['val_predict'],
                            val_time_features=all_data['val']['val_time_features'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_predict=all_data['test']['test_predict'],
                            test_time_features=all_data['test']['test_time_features'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                            )
    return all_data


def normalization(train, val, test):

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)
    print('mean.shape:', mean.shape)
    print('std.shape:', std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm


if __name__ == '__main__':
    num_of_vertices = 307  # 传感器数量
    num_of_hours = 1  # 使用前一个小时的数据进行预测
    points_per_hour = 12
    num_for_predict = 6
    num_of_weeks = 1
    num_of_days = 1
    Q = 3  # 时间窗口
    time_data_filename = './data/PEMS04/PEMS04_data.csv'
    all_data = read_and_generate_dataset(Q, time_data_filename, num_of_weeks, num_of_days, num_of_hours,
                                         num_for_predict, points_per_hour=points_per_hour, save=True)





