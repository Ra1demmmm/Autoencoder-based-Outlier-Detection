import os
import pandas as pd
import numpy as np
import torch


def load_data(rootDir, filename, train_ratio=0.4, valid_ratio=0.3, test_ratio=0.3):
    data = pd.read_csv(os.path.join(rootDir, filename + '_data.txt'), delim_whitespace=True, header=None, prefix="dim")
    data = torch.tensor(data.values)
    label = pd.read_csv(os.path.join(rootDir, filename + '_label.txt'), delim_whitespace=True, header=None, prefix="dim")
    label = 1 - torch.tensor(label.values) # 0:Normal, 1:Outlier

    mask = (label == 0)
    mask = mask.repeat(1,data.shape[1])
    normals = data[mask].reshape(-1,data.shape[1])
    n = len(normals)
    index = [i for i in range(n)]
    np.random.shuffle(index)
    normals = normals[index]
    normal_train, normal_valid, normal_test = [], [], []
    for i in range(n):
        if i < train_ratio * n:
            normal_train.append(normals[i])
        elif i >= train_ratio * n and i < (train_ratio + valid_ratio) * n:
            normal_valid.append(normals[i])
        elif i >= (train_ratio + valid_ratio) * n:
            normal_test.append(normals[i])

    normal_train, normal_valid, normal_test = torch.stack(normal_train), torch.stack(normal_valid), torch.stack(normal_test)

    mask = (label == 1)
    mask = mask.repeat(1, data.shape[1])
    outliers = data[mask].reshape(-1, data.shape[1])
    n = len(outliers)
    index = [i for i in range(n)]
    np.random.shuffle(index)
    outliers = outliers[index]
    outlier_train, outlier_valid, outlier_test = [], [], []
    for i in range(n):
        if i < train_ratio * n:
            outlier_train.append(outliers[i])
        elif i >= train_ratio * n and i < (train_ratio + valid_ratio) * n:
            outlier_valid.append(outliers[i])
        elif i >= (train_ratio + valid_ratio) * n:
            outlier_test.append(outliers[i])

    outlier_train, outlier_valid, outlier_test = torch.stack(outlier_train), torch.stack(outlier_valid), torch.stack(outlier_test)

    train_data = torch.cat((normal_train, outlier_train), dim=0)
    train_label = torch.tensor([0] * len(normal_train) + [1] * len(outlier_train))
    valid_data = torch.cat((normal_valid, outlier_valid), dim=0)
    valid_label = torch.tensor([0] * len(normal_valid) + [1] * len(outlier_valid))
    test_data = torch.cat((normal_test, outlier_test), dim=0)
    test_label = torch.tensor([0] * len(normal_test) + [1] * len(outlier_test))

    train_shuffle_index = np.random.choice(range(len(train_data)), len(train_data), replace=False)
    train_data = train_data[train_shuffle_index]
    train_label = train_label[train_shuffle_index]

    valid_shuffle_index = np.random.choice(range(len(valid_data)), len(valid_data), replace=False)
    valid_data = valid_data[valid_shuffle_index]
    valid_label = valid_label[valid_shuffle_index]

    test_shuffle_index = np.random.choice(range(len(test_data)), len(test_data), replace=False)
    test_data = test_data[test_shuffle_index]
    test_label = test_label[test_shuffle_index]

    all_data = torch.cat((train_data, valid_data, test_data), dim=0)
    all_label = torch.cat((train_label, valid_label, test_label), dim=0)

    return [all_data, all_label], [train_data, train_label], [valid_data, valid_label], [test_data, test_label]
