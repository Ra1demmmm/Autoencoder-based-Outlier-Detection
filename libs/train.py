import os
import torch
from torch import optim
from torch.autograd import Variable
import copy
from torch.optim.lr_scheduler import StepLR


import numpy as np

from libs.loss import mse, mse_samples, apre, apre_samples
from libs.net_init import initNetParams
from sklearn.metrics import roc_auc_score


def AE_train_step1(model,
                   # dump_dir,
                   train_data,
                   valid_data,
                   valid_label,
                   train_target=None,
                   valid_target=None,
                   init_method=None,
                   shuffle=True,
                   num_epoch=200,
                   loss_type='mse',
                   alpha=1,
                   beta=1,
                   lr=1e-3,
                   lr_update=False,
                   output_score=False):

    if init_method is not None:
        initNetParams(model, init_method)

    if train_target == None:
        train_target = train_data.clone()

    # if valid_target == None:
    #     valid_target = [valid_data.clone()]
    num_targets = len(valid_target)

    if torch.cuda.is_available():
        model.cuda()
        train_data = Variable(train_data).to(torch.float32).cuda()
        train_target = Variable(train_target).to(torch.float32).cuda()
        for i in range(num_targets):
            valid_target[i] = Variable(valid_target[i]).to(torch.float32).cuda()
        valid_data = Variable(valid_data).to(torch.float32).cuda()

    if loss_type == 'mse':
        lossfn = mse
        loss_evalfn = mse_samples
    elif loss_type == 'pre':
        lossfn = apre
        loss_evalfn = apre_samples


    optimizer = optim.Adam(model.parameters(), lr=lr)
    if lr_update:
        scheduler = StepLR(optimizer,
                           step_size=1000,
                           gamma=0.5
                           )


    auc_all_epochs = torch.zeros(num_targets, num_epoch)
    if output_score:
        best_aucs = torch.zeros(num_targets)
        best_scores = torch.zeros(num_targets, len(valid_data))

    for epoch in range(num_epoch):
        # print(epoch)

        if shuffle:
            index = [i for i in range(len(train_data))]
            np.random.shuffle(index)
            train_data_shuffled = train_data[index]
            train_target_shuffled = train_target[index]
        else:
            train_data_shuffled = train_data
            train_target_shuffled = train_target

        # forward
        model.train()
        optimizer.zero_grad()
        output = model(train_data_shuffled)

        if loss_type == 'pre':
            loss = lossfn(output, train_target_shuffled, 1., 1.) # pre
        else:
            loss = lossfn(output, train_target_shuffled)

        # backward
        loss.backward()
        optimizer.step()
        if lr_update:
            scheduler.step()

        #evaluation
        model.eval()
        with torch.no_grad():
            output = model(valid_data)

            for i in range(num_targets):
                if loss_type == 'pre':
                    loss_eval = loss_evalfn(output, valid_target[i], alpha, beta)
                else:
                    loss_eval = loss_evalfn(output, valid_target[i])
                loss_eval = loss_eval.cpu()
                roc_auc_eval = roc_auc_score(valid_label, loss_eval)
                auc_all_epochs[i, epoch] = roc_auc_eval

                if output_score:
                    if roc_auc_eval > best_aucs[i]:
                        best_aucs[i] = roc_auc_eval
                        best_scores[i] = loss_eval

    # torch.save(model, os.path.join(dump_dir, 'last.pt'))

    if output_score:
        return auc_all_epochs, best_scores
    else:
        return auc_all_epochs

def AE_train_step2(model,
                   # dump_dir,
                   train_data,
                   valid_data,
                   valid_label,
                   test_data,
                   test_label,
                   train_target=None,
                   valid_target=None,
                   test_target=None,
                   init_method=None,
                   shuffle=True,
                   num_epoch=200,
                   loss_type='mse',
                   alpha=1,
                   beta=1,
                   lr=1e-3,
                   lr_update=False,
                   output_score=False):

    if init_method is not None:
        initNetParams(model, init_method)

    if train_target == None:
        train_target = train_data.clone()

    num_targets = len(valid_target)

    if torch.cuda.is_available():
        model.cuda()
        train_data = Variable(train_data).to(torch.float32).cuda()
        train_target = Variable(train_target).to(torch.float32).cuda()
        valid_data = Variable(valid_data).to(torch.float32).cuda()
        for i in range(num_targets):
            valid_target[i] = Variable(valid_target[i]).to(torch.float32).cuda()
        test_data = Variable(test_data).to(torch.float32).cuda()
        for i in range(num_targets):
            test_target[i] = Variable(test_target[i]).to(torch.float32).cuda()

    if loss_type == 'mse':
        lossfn = mse
        loss_evalfn = mse_samples
    elif loss_type == 'pre':
        lossfn = apre
        loss_evalfn = apre_samples

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if lr_update:
        scheduler = StepLR(optimizer,
                           step_size=1000,
                           gamma=0.5
                           )

    auc_all_epochs = torch.zeros(num_targets, num_epoch)
    best_valid_aucs = torch.zeros(num_targets)

    best_model_list = [copy.deepcopy(model) for i in range(num_targets)]
    for epoch in range(num_epoch):
        # print(epoch)

        if shuffle:
            index = [i for i in range(len(train_data))]
            np.random.shuffle(index)
            train_data_shuffled = train_data[index]
            train_target_shuffled = train_target[index]
        else:
            train_data_shuffled = train_data
            train_target_shuffled = train_target

        # forward
        model.train()
        optimizer.zero_grad()
        output = model(train_data_shuffled)

        if loss_type == 'pre':
            loss = lossfn(output, train_target_shuffled, 1., 1.)  # pre
        else:
            loss = lossfn(output, train_target_shuffled)

        # backward
        loss.backward()
        optimizer.step()
        if lr_update:
            scheduler.step()

        #evaluation
        model.eval()
        with torch.no_grad():
            output = model(valid_data)

            for i in range(num_targets):
                if loss_type == 'pre':
                    loss_eval = loss_evalfn(output, valid_target[i], alpha, beta)
                else:
                    loss_eval = loss_evalfn(output, valid_target[i])
                loss_eval = loss_eval.cpu()
                roc_auc_eval = roc_auc_score(valid_label, loss_eval)
                auc_all_epochs[i, epoch] = roc_auc_eval

                if roc_auc_eval > best_valid_aucs[i]:
                    best_valid_aucs[i] = roc_auc_eval
                    best_model_list[i] = copy.deepcopy(model)


    # test
    test_aucs = torch.zeros(num_targets)
    if output_score:
        test_scores = torch.zeros(num_targets, len(test_data))

    for i in range(num_targets):
        best_model_list[i].eval()
        with torch.no_grad():
            output = best_model_list[i](test_data)

            if loss_type == 'pre':
                loss_eval = loss_evalfn(output, test_target[i], alpha, beta)
            else:
                loss_eval = loss_evalfn(output, test_target[i])
            loss_eval = loss_eval.cpu()
            roc_auc_eval = roc_auc_score(test_label, loss_eval)
            test_aucs[i] = roc_auc_eval

        if output_score:
            test_scores[i] = loss_eval


    # torch.save(model, os.path.join(dump_dir, 'last.pt'))

    if output_score:
        return best_valid_aucs, test_aucs, test_scores
    else:
        return best_valid_aucs, test_aucs

