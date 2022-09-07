import copy
import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse

from utils import *
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import torch.nn.functional as F
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
from model import lstm, ml
import itertools
import functools
from Dataset_LSTM import LSTM_Dataset
from Vocab import *
from sklearn.model_selection import KFold

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', type=str, choices=['apcd', 'hidd', 'khin'], default='apcd')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--feature_space', type=str, choices=['combined', 'local'], default='combined')
    parser.add_argument('--code_topk', type=int, default=300)
    # parser.add_argument('--run_model', choices=['LSTM', 'MLP'], default='MLP')

    # Deep PSModels
    parser.add_argument('--batch_size', type=int, default=256)  # 768)  # 64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-6)  # )0001)
    parser.add_argument('--epochs', type=int, default=10)  # 30
    # LSTM
    parser.add_argument('--diag_emb_size', type=int, default=128)
    # parser.add_argument('--med_emb_size', type=int, default=128)
    # parser.add_argument('--med_hidden_size', type=int, default=64)
    parser.add_argument('--diag_hidden_size', type=int, default=64)
    parser.add_argument('--lstm_hidden_size', type=int, default=100)
    # MLP
    # parser.add_argument('--hidden_size', type=str, default='', help=', delimited integers')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/5fold/lstm/')
    args = parser.parse_args()

    # Modifying args
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.random_seed >= 0:
        rseed = args.random_seed
    else:
        from datetime import datetime
        rseed = datetime.now()
    args.random_seed = rseed
    args.save_model_filename = os.path.join(args.output_dir, '{}/lstm_S{}.model'.format(args.dataset,
                                                                                        args.random_seed))
    check_and_mkdir(args.save_model_filename)
    #
    # args.hidden_size = [int(x.strip()) for x in args.hidden_size.split(',')
    #                     if (x.strip() not in ('', '0'))]

    return args


if __name__ == '__main__':
    # python main_lstm.py --dataset hidd --random_seed 0 2>&1 | tee  log/lstm_hidd_r0.txt

    start_time = time.time()
    args = parse_args()
    print('args: ', args)
    print('random_seed: ', args.random_seed)
    print('device: ', args.device)
    print('Strategy: 5-Fold')

    # reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    #
    print('Dataset:', args.dataset)
    encode2namefile = r'dataset/icd9encode2name.pkl'
    majordatafile = r'dataset/final_pats_1st_neg_triples_{}-{}_new_labeled.pkl'.format(args.dataset, 'icd9')
    with open(encode2namefile, 'rb') as f:
        dx_name = pickle.load(f)
        print('len(dx_name):', len(dx_name))

    with open(majordatafile, 'rb') as f:
        data_1st_neg = pickle.load(f)
        print('len(data_1st_neg):', len(data_1st_neg))

    if args.feature_space == 'combined':
        # Load combined features
        with open('Pre_train/selected_features_apcd.obj', 'rb') as f:
            vocab_apcd = pickle.load(f)

        with open('Pre_train/selected_features_hidd.obj', 'rb') as f:
            vocab_hidd = pickle.load(f)

        with open('Pre_train/selected_features_khin.obj', 'rb') as f:
            vocab_khin = pickle.load(f)

        vocab_combined = copy.deepcopy(vocab_apcd)
        vocab_combined.extend_vocab(vocab_hidd)
        vocab_combined.extend_vocab(vocab_khin)

        print('Using combined feature space')
        my_dataset = LSTM_Dataset(data_1st_neg, diag_name=dx_name, diag_code_vocab=vocab_combined)

    else:
        print('Using local feature space, top k:', args.code_topk)
        my_dataset = LSTM_Dataset(data_1st_neg, diag_name=dx_name, diag_code_topk=args.code_topk)


    n_feature = my_dataset.DIM_OF_CONFOUNDERS
    feature_name = my_dataset.FEATURE_NAME
    print('n_feature: ', n_feature, ':')

    train_ratio = 0.8
    print('train_ratio: ', train_ratio,
          'test_ratio: ', 1 - train_ratio)

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))

    # data shuffle in here
    np.random.shuffle(indices)

    train_set_indices, test_set_indices = indices[:train_index], indices[train_index:]

    test_sampler = SubsetRandomSampler(test_set_indices)
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, sampler=test_sampler)
    data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(indices))

    paras_grid = {
        'hidden_size': [32, 64],  # 64, 128
        'diag_hidden_size': [32, 64],
        'diag_embedding_size': [32, 64],
        'lr': [1e-3, 1e-4],
        'weight_decay': [1e-4, 1e-5, ],
        'batch_size': [1024],  # 50
    }
    hyper_paras_names, hyper_paras_v = zip(*paras_grid.items())
    hyper_paras_list = list(itertools.product(*hyper_paras_v))
    print('Model lstm Searching Space N={}: '.format(len(hyper_paras_list)), paras_grid)

    best_hyper_paras = None
    best_model = None
    best_auc = float('-inf')
    best_balance = float('inf')
    best_model_epoch = -1
    best_result_9 = None
    best_result_95 = None
    results = []
    i = -1
    i_iter = -1
    for hyper_paras in tqdm(hyper_paras_list):
        kf = KFold(random_state=0, shuffle=True)
        auc_list = []
        for train_index, test_index in kf.split(train_set_indices):
            train_sampler = SubsetRandomSampler(train_index)
            val_sampler = SubsetRandomSampler(test_index)
            val_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size, sampler=val_sampler)
            hidden_size, diag_hidden_size, diag_embedding_size, lr, weight_decay, batch_size = hyper_paras
            train_loader_shuffled = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, sampler=train_sampler)
            model_params = dict(
                diag_hidden_size=diag_hidden_size,  # 64
                hidden_size=hidden_size,  # 100,
                bidirectional=True,
                diag_vocab_size=len(my_dataset.diag_code_vocab),
                diag_embedding_size=diag_embedding_size,
            )
            model = lstm.LSTMModel(**model_params)
            if args.cuda:
                model = model.to('cuda')
            print(model)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            for epoch in tqdm(range(args.epochs)):
                i_iter += 1
                epoch_losses = []
                uid_list = []
                for confounder, labels, outcome, uid in train_loader_shuffled:
                    model.train()
                    # train IPW
                    optimizer.zero_grad()
                    uid_list.extend(uid)
                    if args.cuda:  # confounder = (diag, med, sex, age)
                        for ii in range(len(confounder)):
                            confounder[ii] = confounder[ii].to('cuda')
                        flag = labels.to('cuda').float()

                    treatment_logits, _ = model(confounder)
                    loss = F.binary_cross_entropy_with_logits(treatment_logits, flag)

                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())

                # just finish 1 epoch
                # scheduler.step()
                epoch_losses = np.mean(epoch_losses)

            auc_val, loss_val, Y_val, Y_pred_val, uid_val = transfer_data(model, val_loader, cuda=args.cuda, normalized=False)
            auc_list.append(auc_val)

        auc_theta = np.mean(auc_list)

        if auc_theta > best_auc:
            best_auc = auc_theta
            best_hyper_paras = hyper_paras

    hidden_size, diag_hidden_size, diag_embedding_size, lr, weight_decay, batch_size = best_hyper_paras
    train_loader_shuffled = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_set_indices))
    model_params = dict(
        diag_hidden_size=diag_hidden_size,  # 64
        hidden_size=hidden_size,  # 100,
        bidirectional=True,
        diag_vocab_size=len(my_dataset.diag_code_vocab),
        diag_embedding_size=diag_embedding_size,
    )
    best_model = lstm.LSTMModel(**model_params)
    if args.cuda:
        best_model = best_model.to('cuda')
    print(best_model)

    optimizer = torch.optim.Adam(best_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.epochs)):
        i_iter += 1
        epoch_losses = []
        uid_list = []
        for confounder, labels, outcome, uid in train_loader_shuffled:
            best_model.train()
            # train IPW
            optimizer.zero_grad()
            uid_list.extend(uid)
            if args.cuda:  # confounder = (diag, med, sex, age)
                for ii in range(len(confounder)):
                    confounder[ii] = confounder[ii].to('cuda')
                flag = labels.to('cuda').float()

            treatment_logits, _ = best_model(confounder)
            loss = F.binary_cross_entropy_with_logits(treatment_logits, flag)

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_losses = np.mean(epoch_losses)

    save_model(best_model, args.save_model_filename, model_params=best_hyper_paras)


    auc_test, loss_test, Y_test, Y_pred_test, uid_test = transfer_data(best_model, test_loader, cuda=args.cuda, normalized=False)

    print('Final results', "test-auc", auc_test)

    result_1 = ml.MLModels._performance_at_specificity_or_threshold(Y_pred_test, Y_test, specificity=0.9)
    print('......Results at specificity 0.95:')
    result_2 = ml.MLModels._performance_at_specificity_or_threshold(Y_pred_test, Y_test, specificity=0.95)

    df1 = pd.DataFrame([result_1 + (best_hyper_paras,), result_2 + (best_hyper_paras,)],
                       columns=["AUC", "threshold", "Specificity", "Sensitivity/recall", "PPV/precision",
                                "n_negative", "n_positive", "precision_recall_fscore_support", 'best_hyper_paras'],
                       index=['r_9', 'r_95'])
    df1.to_csv(os.path.join(os.path.dirname(args.save_model_filename),
                            'test_results_{}r{}.csv'.format('lstm', args.random_seed)))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
