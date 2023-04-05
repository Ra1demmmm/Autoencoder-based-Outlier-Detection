import os
import warnings
warnings.filterwarnings("ignore")

import argparse

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import  roc_auc_score
import copy

from libs.model import AE, PAE, get_model_args
from libs.train import AE_train_step1, AE_train_step2


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def picking_k(ks, scores, labels):
    s = torch.mean(scores, dim=1, keepdim=False)
    m = np.zeros(len(ks))
    for i in range(len(m)):
        m[i] = roc_auc_score(labels, s[i])
    index = torch.argmax(torch.tensor(m))

    return ks[index]


def min_max(data, min=None, max=None):
    if min is None:
        min = torch.min(data)
    if max is None:
        max = torch.max(data)

    return (data - min) / (max - min), min, max

def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset = args.dataset
    data_dir = args.data_dir
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(os.path.join(result_dir, 'config.txt'), "w") as file:
        print(args, sep='\n', file=file)


    result_auc = pd.DataFrame()
    result_auc['dataset'] = [dataset]

    result_auc['non-MSS'] = [-1.]
    result_auc['MSS1'] = [-1.]
    result_auc['MSS2'] = [-1.]
    result_auc['MSS3'] = [-1.]


    result_k = pd.DataFrame()
    result_k['dataset'] = dataset

    result_k['MSS1'] = [-1]
    result_k['MSS2'] = [-1]
    result_k['MSS3'] = [-1]


    net = args.net
    alpha = args.alpha
    beta = args.beta
    num_epochs = args.epochs
    seeds = range(args.inits)
    init_method = None


    setup_seed(42)
    print('Training on "'+dataset+'" ...')

    train_data = torch.tensor(np.loadtxt(os.path.join(data_dir, dataset, 'train_data.txt')))
    train_label = torch.tensor(np.loadtxt(os.path.join(data_dir, dataset, 'train_label.txt')))
    valid_data = torch.tensor(np.loadtxt(os.path.join(data_dir, dataset, 'valid_data.txt')))
    valid_label = torch.tensor(np.loadtxt(os.path.join(data_dir, dataset, 'valid_label.txt')))
    test_data = torch.tensor(np.loadtxt(os.path.join(data_dir, dataset, 'test_data.txt')))
    test_label = torch.tensor(np.loadtxt(os.path.join(data_dir, dataset, 'test_label.txt')))

    n = train_data.shape[0]
    d = train_data.shape[1]

    net_args_list = get_model_args(d)

    # pick best k in 5 seeds
    print('..step1')

    seeds_for_pickk = [0, 10, 20, 30, 40]

    # pick k mean
    k_xy_score = torch.zeros((3, len(range(2, min(101, n))), len(seeds_for_pickk), len(valid_data)))

    valid_data_path = []
    valid_data_list = []
    for k in range(2, min(101, n)):
        for shifting_way in range(1, 4):
            valid_data_path.append(
                os.path.join(data_dir, dataset, str(k), 'shifted_valid_mean_t{}.txt'.format(shifting_way)))
            valid_data_list.append(torch.tensor(np.loadtxt(valid_data_path[-1])))

    for seed in range(len(seeds_for_pickk)):

        setup_seed(seeds_for_pickk[seed])

        if net == 'AE':
            model_source = AE(*net_args_list)
            loss_type = 'mse'

        elif net == 'PAE':
            model_source = PAE(*net_args_list)
            loss_type = 'pre'

        model = copy.deepcopy(model_source)
        _, training_result = AE_train_step1(model,
                                           train_data,
                                           valid_data,
                                           valid_label,
                                           valid_target=valid_data_list,
                                           init_method=init_method,
                                           shuffle=True,
                                           num_epoch=num_epochs,
                                           loss_type=loss_type,
                                           alpha=alpha,
                                           beta=beta,
                                           lr=1e-3,
                                           lr_update=True,
                                           output_score=True)

        i = 0
        for k in range(2, min(101, n)):
            for shifting_way in range(3):
                score = training_result[i]
                score, _, _ = min_max(score)
                k_xy_score[shifting_way, k - 2, seed] = score
                i += 1


    best_k_xy = torch.zeros(3)

    valid_data_path2 = []
    valid_data_list2 = []
    test_data_path2 = []
    test_data_list2 = []
    for i in range(3):
        best_k_xy[i] = picking_k(list(range(2, min(101, n))), k_xy_score[i], valid_label)
        valid_data_path2.append(
            os.path.join(data_dir, dataset, str(int(best_k_xy[i].item())), 'shifted_valid_mean_t{}.txt'.format(i+1)))
        valid_data_list2.append(torch.tensor(np.loadtxt(valid_data_path2[-1])))
        test_data_path2.append(
            os.path.join(data_dir, dataset, str(int(best_k_xy[i].item())), 'shifted_test_mean_t{}.txt'.format(i + 1)))
        test_data_list2.append(torch.tensor(np.loadtxt(test_data_path2[-1])))

    # original data
    valid_data_list2.append(valid_data)
    test_data_list2.append(test_data)

    result_k['MSS1'] = [int(best_k_xy[0].item())]
    result_k['MSS2'] = [int(best_k_xy[1].item())]
    result_k['MSS3'] = [int(best_k_xy[2].item())]


    result_k.to_csv(os.path.join(result_dir, 'result_k.txt'), sep='\t', index=False)
    result_k.to_excel(os.path.join(result_dir, 'result_k.xlsx'), index=False)


    # Compute over all seeds
    print('..step2')

    xy_score = torch.zeros((3, args.inits, len(test_data)))
    xx_score = torch.zeros((args.inits, len(test_data)))

    xy_auc = torch.zeros((3, args.inits))
    xx_auc = torch.zeros(args.inits)

    for random_seed in seeds:

        setup_seed(random_seed)
        if net == 'AE':
            model_source = AE(*net_args_list)
            loss_type = 'mse'

        elif net == 'PAE':
            model_source = PAE(*net_args_list)
            loss_type = 'pre'

        model = copy.deepcopy(model_source)
        setup_seed(random_seed)

        aucs, _, training_result = AE_train_step2(model,
                                                  train_data,
                                                  valid_data,
                                                  valid_label,
                                                  test_data,
                                                  test_label,
                                                  valid_target=valid_data_list2,
                                                  test_target=test_data_list2,
                                                  init_method=init_method,
                                                  shuffle=True,
                                                  num_epoch=num_epochs,
                                                  loss_type=loss_type,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  lr=1e-3,
                                                  lr_update=True,
                                                  output_score=True)


        for shifting_way in range(3):
            score = training_result[shifting_way]
            score, _, _ = min_max(score)
            xy_score[shifting_way, random_seed] = score
            xy_auc[shifting_way, random_seed] = aucs[shifting_way]

        score = training_result[-1]
        score, _, _ = min_max(score)
        xx_score[random_seed] = score
        xx_auc[random_seed] = aucs[-1]


    xy = np.zeros(3)

    for i in range(len(xy)):
        _, ind = torch.topk(xy_auc[i], k=5, dim=0, largest=True)
        xy_s = xy_score[i, ind[0]]
        xy[i] = roc_auc_score(test_label, xy_s)


    _, ind = torch.topk(xx_auc, k=5, dim=0, largest=True)
    xx_s = xx_score[ind[0]]
    xx = roc_auc_score(test_label, xx_s)

    result_auc['non-MSS'] = [xx]
    result_auc['MSS1'] = [xy[0]]
    result_auc['MSS2'] = [xy[1]]
    result_auc['MSS3'] = [xy[2]]



    result_auc.to_csv(os.path.join(result_dir, 'result_auc.txt'), sep='\t', index=False)
    result_auc.to_excel(os.path.join(result_dir, 'result_auc.xlsx'), index=False)


    print(result_auc)
    print('The results are in ' + result_dir)

if __name__ == "__main__":
    setup_seed(42)
    parser = argparse.ArgumentParser(
        description=
        "Command to start Autoencoder training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu",
                        type=str,
                        default="0",
                        help="Training on which GPU")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="Name of the dataset")
    parser.add_argument("--data_dir",
                        type=str,
                        required=True,
                        help="Directory of data")
    parser.add_argument("--epochs",
                        type=int,
                        default=1000,
                        help="Number of training epochs")
    parser.add_argument("--result_dir",
                        type=str,
                        required=True,
                        help="Directory to dump results")
    parser.add_argument("--net",
                        type=str,
                        default='AE',
                        help="Which network to use: AE; PAE")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.5,
                        help="Hyper-paramter alpha for PAE")
    parser.add_argument("--beta",
                        type=float,
                        default=2,
                        help="Hyper-paramter beta for PAE")
    parser.add_argument("--inits",
                        type=int,
                        default=20,
                        help="How many random initial states")
    args = parser.parse_args()
    # logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    run(args)
