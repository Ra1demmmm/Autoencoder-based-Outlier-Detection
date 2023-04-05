import sys
import os
import numpy as np
import torch

from libs.data import load_data
from libs.dataset_shift import data_mean_shift

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def mean_sd(data, mean=None, std=None):
    if mean is None:
        mean = torch.mean(data, dim=0)
    if std is None:
        std = torch.std(data, dim=0)

    return (data - mean) / (std + 1e-5), mean, std

if __name__ == "__main__":
    data_dir = './dataset/original/Optdigits'
    target_dir = './dataset/processed'
    dataset = 'Optdigits'

    print('Processing "'+dataset+'" ...')

    setup_seed(42)
    all, train, valid, test = load_data(data_dir, dataset, 0.5, 0.25, 0.25)

    train_data = train[0]
    train_label = train[1]
    valid_data = valid[0]
    valid_label = valid[1]
    test_data = test[0]
    test_label = test[1]

    tn = len(train_data)

    train_data = torch.cat((train_data, valid_data), dim=0)
    train_label = torch.cat((train_label, valid_label), dim=0)

    ms = True
    if ms:
        train_data, mean, std = mean_sd(train_data)
        valid_data = train_data[tn:]
        test_data, _, _ = mean_sd(test_data, mean, std)

    dim = train_data.shape[1]

    if not os.path.exists(os.path.join(target_dir, dataset)):
        os.makedirs(os.path.join(target_dir, dataset))

    np.savetxt(os.path.join(target_dir, dataset, 'train_data.txt'), train_data.numpy())
    np.savetxt(os.path.join(target_dir, dataset, 'train_label.txt'), train_label.numpy())
    np.savetxt(os.path.join(target_dir, dataset, 'valid_data.txt'), valid_data.numpy())
    np.savetxt(os.path.join(target_dir, dataset, 'valid_label.txt'), valid_label.numpy())
    np.savetxt(os.path.join(target_dir, dataset, 'test_data.txt'), test_data.numpy())
    np.savetxt(os.path.join(target_dir, dataset, 'test_label.txt'), test_label.numpy())

    data_mean_shift(target_dir, dataset, train_data, tn, test_data)